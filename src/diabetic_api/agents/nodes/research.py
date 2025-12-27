"""Research agent - interprets query results and provides insights."""

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

from diabetic_api.agents.state import ChatState
from diabetic_api.agents.prompts.research import (
    RESEARCH_SYSTEM_PROMPT,
    format_research_prompt,
)

logger = logging.getLogger(__name__)


class ResearchAgent:
    """
    Research agent that interprets data and provides health insights.
    
    Takes query results, chat history, and user questions to generate
    helpful, contextual responses about diabetic health data.
    
    Features:
    - Supportive, educational tone
    - Markdown-formatted responses
    - Handles both successful queries and error states
    - Provides context and suggestions
    
    Maps to the N8N Research Agent workflow.
    """

    def __init__(self, llm: BaseChatModel):
        """
        Initialize research agent.
        
        Args:
            llm: Language model for generating responses
        """
        self.llm = llm

    async def __call__(self, state: ChatState) -> dict:
        """
        Generate research response based on available context.

        Args:
            state: Current graph state with message, query_results, history, full_data

        Returns:
            Dict with response to merge into state
        """
        logger.info("Research agent generating response...")

        # Get available context
        query_results = state.get("query_results")
        chat_history = state.get("history", [])
        last_error = state.get("last_error")
        route_decision = state.get("route_decision")
        full_data = state.get("full_data")

        # Log context
        if query_results:
            logger.info(f"Using {len(query_results)} query results")
        if full_data:
            sensor_lines = len(full_data.get("sensorData", "").split("\n")) if full_data.get("sensorData") else 0
            logger.info(f"Using full dataset with ~{sensor_lines} sensor readings")
        if last_error:
            logger.info(f"Query had error: {last_error[:50]}...")

        # Format the prompt with all context (including full_data)
        user_prompt = format_research_prompt(
            question=state["message"],
            query_results=query_results,
            chat_history=chat_history,
            last_error=last_error,
            full_data=full_data,
        )

        # Build messages
        messages = [
            SystemMessage(content=RESEARCH_SYSTEM_PROMPT.format(context="")),
            HumanMessage(content=user_prompt),
        ]

        try:
            # Generate response
            response = await self.llm.ainvoke(messages)
            response_text = response.content

            logger.info(f"Generated response: {len(response_text)} chars")

            return {"response": response_text}

        except Exception as e:
            logger.error(f"Research agent error: {e}")

            # Generate fallback response
            fallback = self._generate_fallback(state, str(e))
            return {"response": fallback}

    def _generate_fallback(self, state: ChatState, error: str) -> str:
        """Generate a fallback response when the LLM fails."""
        query_results = state.get("query_results")
        
        response_parts = [
            "I apologize, but I encountered an issue generating a detailed response.",
        ]
        
        # If we have query results, at least show them
        if query_results and len(query_results) > 0:
            response_parts.append("\n\nHowever, here's the data from your query:\n")
            
            import json
            # Format results nicely
            for i, result in enumerate(query_results[:5]):
                formatted = json.dumps(result, indent=2, default=str)
                response_parts.append(f"```json\n{formatted}\n```\n")
            
            if len(query_results) > 5:
                response_parts.append(f"\n... and {len(query_results) - 5} more results.")
        
        response_parts.append(
            "\n\nPlease try rephrasing your question, or ask me something else!"
        )
        
        return "".join(response_parts)


class StreamingResearchAgent(ResearchAgent):
    """
    Research agent with streaming support.
    
    Yields response chunks as they're generated for real-time display.
    """

    async def stream(self, state: ChatState):
        """
        Stream response chunks.

        Args:
            state: Current graph state

        Yields:
            Response text chunks
        """
        logger.info("Research agent streaming response...")

        # Get context
        query_results = state.get("query_results")
        chat_history = state.get("history", [])
        last_error = state.get("last_error")
        full_data = state.get("full_data")

        # Format prompt (including full_data)
        user_prompt = format_research_prompt(
            question=state["message"],
            query_results=query_results,
            chat_history=chat_history,
            last_error=last_error,
            full_data=full_data,
        )

        messages = [
            SystemMessage(content=RESEARCH_SYSTEM_PROMPT.format(context="")),
            HumanMessage(content=user_prompt),
        ]

        try:
            # Stream response
            async for chunk in self.llm.astream(messages):
                if hasattr(chunk, 'content') and chunk.content:
                    yield chunk.content

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield self._generate_fallback(state, str(e))
