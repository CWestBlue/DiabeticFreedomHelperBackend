"""Query generator agent - creates MongoDB aggregation pipelines from natural language."""

import json
import logging
import re

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

from diabetic_api.agents.state import ChatState
from diabetic_api.agents.prompts.query_gen import QUERY_GEN_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


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
        
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=state["message"]),
        ]

        try:
            # Generate query
            response = await self.llm.ainvoke(messages)
            query_text = response.content.strip()
            
            # Clean up response
            query_text = self._clean_response(query_text)
            
            logger.debug(f"Generated query: {query_text[:200]}...")
            
            # Check for error response
            if query_text.startswith("ERROR:"):
                return {
                    "generated_query": None,
                    "query_results": None,
                    "last_error": query_text,
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
    
    Args:
        state: Current graph state
        
    Returns:
        "retry" if should retry query_gen, "continue" to proceed to research
    """
    last_error = state.get("last_error")
    retry_count = state.get("_retry_count", 0)
    max_retries = 2
    
    if last_error and retry_count < max_retries:
        logger.info(f"Will retry query (attempt {retry_count + 1}/{max_retries})")
        return "retry"
    
    return "continue"
