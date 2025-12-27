"""Router agent - decides which workflow path to execute."""

import json
import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

from diabetic_api.agents.state import ChatState, RouteDecision
from diabetic_api.agents.prompts.router import ROUTER_SYSTEM_PROMPT
from diabetic_api.services.usage import UsageLimitExceeded, extract_token_usage

logger = logging.getLogger(__name__)

# Agent name for usage tracking
AGENT_NAME = "router"


class RouterAgent:
    """
    Router agent that decides which workflow path to execute.
    
    Analyzes the user's message and chat history to determine:
    - Whether a MongoDB query is needed
    - Whether interpretation by the Research agent is required
    - What data strategy to use
    
    Maps directly to the N8N Router Decision Agent workflow.
    """

    def __init__(self, llm: BaseChatModel):
        """
        Initialize router agent.
        
        Args:
            llm: Language model to use for routing decisions
        """
        self.llm = llm

    async def __call__(self, state: ChatState) -> dict:
        """
        Execute routing decision.
        
        Args:
            state: Current graph state
            
        Returns:
            Dict with route_decision to merge into state
        """
        logger.info(f"Router processing: {state['message'][:50]}...")
        
        # Build context from chat history
        history_context = self._format_history(state.get("history", []))
        
        # Create the user message with context
        user_content = f"""Chat History:
{history_context}

Current User Message: {state['message']}"""

        messages = [
            SystemMessage(content=ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ]

        try:
            # Check usage limits before making LLM call
            usage_service = state.get("usage_service")
            if usage_service:
                await usage_service.check_limit()

            # Get routing decision from LLM
            response = await self.llm.ainvoke(messages)

            # Record usage after successful call (including tokens)
            if usage_service:
                model_name = getattr(self.llm, "model", "unknown")
                input_tokens, output_tokens = extract_token_usage(response)
                await usage_service.record_call(
                    model=str(model_name),
                    agent=AGENT_NAME,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
            response_text = response.content.strip()
            
            # Clean up response if wrapped in markdown
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                response_text = "\n".join(lines[1:-1])
            
            # Parse JSON response
            decision_data = json.loads(response_text)
            
            route_decision = RouteDecision(
                need_mongo_query=decision_data.get("need_mongo_query", "no"),
                need_research_agent=decision_data.get("need_research_agent", "yes"),
                query_results_useful=decision_data.get("query_results_useful", "n/a"),
                data_pass_strategy=decision_data.get("data_pass_strategy", "n/a"),
                workflow=decision_data.get("workflow", "research"),
                workflow_index=decision_data.get("workflow_index", 4),
            )
            
            logger.info(f"Router decision: {route_decision['workflow']} (index {route_decision['workflow_index']})")
            
        except UsageLimitExceeded:
            # Re-raise usage limit exceptions - don't swallow them
            raise
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse router response: {e}")
            # Default to research_query (safest option - will try query and interpret)
            route_decision = RouteDecision(
                need_mongo_query="yes",
                need_research_agent="yes",
                query_results_useful="yes",
                data_pass_strategy="query_only",
                workflow="research_query",
                workflow_index=1,
            )
        except Exception as e:
            logger.error(f"Router error: {e}")
            # Default to research only on error
            route_decision = RouteDecision(
                need_mongo_query="no",
                need_research_agent="yes",
                query_results_useful="n/a",
                data_pass_strategy="n/a",
                workflow="research",
                workflow_index=4,
            )

        return {"route_decision": route_decision}

    def _format_history(self, history: list, max_messages: int = 6) -> str:
        """Format chat history for context."""
        if not history:
            return "(No previous messages)"
        
        formatted = []
        for msg in history[-max_messages:]:
            role = msg.get("role", "user").capitalize()
            text = msg.get("text", "")[:300]  # Truncate long messages
            formatted.append(f"{role}: {text}")
        
        return "\n".join(formatted)


def route_decision_edge(state: ChatState) -> str:
    """
    Conditional edge function for LangGraph routing.
    
    Determines the next node based on the router's decision.
    
    Args:
        state: Current graph state
        
    Returns:
        Name of the next node to execute:
        - "query_gen" for workflows that need a MongoDB query
        - "research" for workflows that go directly to research
    """
    decision = state.get("route_decision")
    
    if decision is None:
        logger.warning("No route decision found, defaulting to research")
        return "research"
    
    workflow = decision.get("workflow", "research")
    need_query = decision.get("need_mongo_query", "no")
    
    # Determine next node
    if need_query == "yes" or workflow in ("query", "research_query", "research_full_query"):
        return "query_gen"
    else:
        return "research"
