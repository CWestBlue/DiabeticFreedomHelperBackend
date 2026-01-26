"""Query generator agent - creates MongoDB aggregation pipelines from natural language."""

import json
import logging
import re

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

from diabetic_api.agents.state import ChatState
from diabetic_api.agents.prompts.query_gen import QUERY_GEN_SYSTEM_PROMPT
from diabetic_api.services.usage import UsageLimitExceeded, extract_token_usage

logger = logging.getLogger(__name__)

# Agent name for usage tracking
AGENT_NAME = "query_gen"


class QueryGenAgent:
    """
    Query generator agent that creates MongoDB aggregation pipelines.
    
    Takes natural language questions about diabetic data and generates
    valid MongoDB aggregation pipelines to answer them.
    
    Features:
    - Full PumpData schema awareness
    - Error recovery with retry capability
    - Strict JSON output parsing
    - Automatic query execution via UoW
    
    Maps directly to the N8N MongoDB Query Generator workflow.
    """

    def __init__(self, llm: BaseChatModel, max_retries: int = 2):
        """
        Initialize query generator agent.
        
        Args:
            llm: Language model for query generation
            max_retries: Maximum retry attempts on query errors
        """
        self.llm = llm
        self.max_retries = max_retries

    def _format_history(self, history: list, max_messages: int = 5) -> str:
        """Format chat history for context."""
        if not history:
            return "(No previous messages)"
        
        formatted = []
        for msg in history[-max_messages:]:
            role = msg.get("role", "user").capitalize()
            text = msg.get("text", "")[:250]  # Truncate long messages
            formatted.append(f"{role}: {text}")
        
        return "\n".join(formatted)

    async def __call__(self, state: ChatState) -> dict:
        """
        Generate and execute MongoDB aggregation pipeline.
        
        Args:
            state: Current graph state with message and optional last_error
            
        Returns:
            Dict with query_results, generated_query, and/or last_error
        """
        last_error = state.get("last_error", "") or ""
        retry_count = state.get("_retry_count", 0)
        
        logger.info(f"QueryGen processing: {state['message'][:50]}...")
        if last_error:
            logger.info(f"Retrying due to error: {last_error}")
        
        # Build prompt with error context
        prompt = QUERY_GEN_SYSTEM_PROMPT.format(
            last_error=last_error if last_error else "None"
        )
        
        # Include chat history for context (helps with follow-up queries)
        history = state.get("history", [])
        history_context = self._format_history(history)
        
        user_content = f"""Chat History:
{history_context}

Current User Message: {state["message"]}"""
        
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=user_content),
        ]

        try:
            # Check usage limits before making LLM call
            usage_service = state.get("usage_service")
            if usage_service:
                await usage_service.check_limit()

            # Generate query
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

            query_text = response.content.strip()
            
            # Clean up response
            query_text = self._clean_response(query_text)
            
            logger.debug(f"Generated query: {query_text[:200]}...")
            
            # Check for error response (e.g., user asked to write data)
            # This is a legitimate refusal, not a retry-able error
            if query_text.startswith("ERROR:"):
                logger.warning(f"Query generator refused: {query_text}")
                return {
                    "generated_query": None,
                    "query_results": None,
                    "last_error": query_text,
                    # Set to max to prevent retries - this is intentional refusal
                    "_retry_count": self.max_retries,
                }
            
            # Parse pipeline
            pipeline = json.loads(query_text)
            
            if not isinstance(pipeline, list):
                raise ValueError("Pipeline must be a JSON array")
            
            # Execute query via UoW
            uow = state.get("uow")
            if uow is None:
                logger.error("No UoW in state - cannot execute query")
                return {
                    "generated_query": query_text,
                    "query_results": None,
                    "last_error": "Database connection not available",
                }
            
            # Log the pipeline for debugging
            logger.debug(f"Executing pipeline: {json.dumps(pipeline, indent=2)}")
            
            # Execute the aggregation
            try:
                results = await uow.run_aggregation("PumpData", pipeline, limit=100)
                logger.info(f"Query returned {len(results)} results")
            except Exception as mongo_error:
                # MongoDB execution error - capture for retry
                error_msg = f"MongoDB error: {str(mongo_error)}"
                logger.warning(f"Query failed: {error_msg}")
                logger.warning(f"Failed pipeline: {query_text[:500]}")
                return {
                    "generated_query": query_text,
                    "query_results": None,
                    "last_error": error_msg,
                    "_retry_count": retry_count + 1,
                }
            
            return {
                "generated_query": query_text,
                "query_results": results,
                "last_error": None,
                "_retry_count": 0,  # Reset on success
            }
            
        except UsageLimitExceeded:
            # Re-raise usage limit exceptions - don't swallow them
            raise
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON in generated query: {str(e)}"
            logger.warning(error_msg)
            return {
                "generated_query": query_text if 'query_text' in locals() else None,
                "query_results": None,
                "last_error": error_msg,
                "_retry_count": retry_count + 1,
            }
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Query execution error: {error_msg}")
            return {
                "generated_query": query_text if 'query_text' in locals() else None,
                "query_results": None,
                "last_error": error_msg,
                "_retry_count": retry_count + 1,
            }

    def _clean_response(self, text: str) -> str:
        """
        Clean LLM response to extract JSON pipeline.
        
        Handles:
        - Markdown code blocks
        - Leading/trailing whitespace
        - Common formatting issues
        """
        text = text.strip()
        
        # Remove markdown code blocks
        if text.startswith("```"):
            # Find the end of the opening fence
            first_newline = text.find("\n")
            if first_newline != -1:
                text = text[first_newline + 1:]
            # Remove closing fence
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
        
        # Remove any leading text before the array
        bracket_match = re.search(r'\[', text)
        if bracket_match and bracket_match.start() > 0:
            text = text[bracket_match.start():]
        
        return text


def should_retry_query(state: ChatState) -> str:
    """
    Conditional edge for query retry logic.

    IMPORTANT: This function has safeguards to prevent infinite loops:
    - Hard cap of 2 retries (3 total attempts max)
    - Only retries if there's both an error AND retry_count is below max

    Args:
        state: Current graph state

    Returns:
        "retry" if should retry query_gen, "continue" to proceed to research
    """
    last_error = state.get("last_error")
    retry_count = state.get("_retry_count", 0)

    # Hard cap on retries to prevent infinite loops (billing protection)
    MAX_RETRIES = 2  # Maximum 2 retries = 3 total attempts

    # Safety check: if retry_count is somehow invalid, don't retry
    if not isinstance(retry_count, int) or retry_count < 0:
        logger.warning(f"Invalid retry_count: {retry_count}, proceeding to research")
        return "continue"

    # Only retry if there's an error AND we haven't hit the cap
    if last_error and retry_count < MAX_RETRIES:
        logger.info(f"Will retry query (attempt {retry_count + 1}/{MAX_RETRIES})")
        return "retry"

    if last_error:
        logger.warning(f"Max retries ({MAX_RETRIES}) reached, proceeding with error: {last_error[:100]}")

    return "continue"
