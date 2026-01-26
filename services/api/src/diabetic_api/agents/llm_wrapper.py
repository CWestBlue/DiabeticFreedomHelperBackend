"""Usage-aware LLM wrapper for tracking and limiting API calls."""

import logging
from typing import Any, AsyncIterator

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatResult

from diabetic_api.services.usage import UsageService, UsageLimitExceeded

logger = logging.getLogger(__name__)


class UsageAwareLLM:
    """
    Wrapper around LangChain LLM that tracks usage and enforces limits.
    
    Intercepts all LLM calls to:
    1. Check usage limits before making API calls
    2. Record successful calls for tracking
    3. Raise UsageLimitExceeded when limits are hit
    
    This provides a centralized safeguard against runaway API costs.
    
    Usage:
        llm = get_llm()
        usage_service = UsageService(db)
        wrapped_llm = UsageAwareLLM(llm, usage_service, agent_name="router")
        
        # Use like normal - tracking happens automatically
        response = await wrapped_llm.ainvoke(messages)
    """
    
    def __init__(
        self,
        llm: BaseChatModel,
        usage_service: UsageService,
        agent_name: str = "unknown",
    ):
        """
        Initialize usage-aware LLM wrapper.
        
        Args:
            llm: The underlying LangChain LLM to wrap
            usage_service: Service for tracking/limiting usage
            agent_name: Name of the agent using this LLM (for tracking)
        """
        self._llm = llm
        self._usage_service = usage_service
        self._agent_name = agent_name
        
        # Get model name from the underlying LLM
        self._model_name = getattr(llm, "model_name", None) or getattr(llm, "model", "unknown")
    
    async def _check_and_record(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        """Check limits and record usage."""
        await self._usage_service.record_call(
            model=str(self._model_name),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            agent=self._agent_name,
        )
    
    async def ainvoke(
        self,
        messages: list[BaseMessage],
        **kwargs: Any,
    ) -> AIMessage:
        """
        Invoke the LLM with usage tracking.
        
        Args:
            messages: List of messages to send
            **kwargs: Additional arguments for the LLM
            
        Returns:
            AI message response
            
        Raises:
            UsageLimitExceeded: If usage limits are exceeded
        """
        # Check limits before making the call
        await self._usage_service.check_limit()
        
        logger.debug(f"LLM call by {self._agent_name} using {self._model_name}")
        
        # Make the actual API call
        response = await self._llm.ainvoke(messages, **kwargs)
        
        # Extract token counts if available
        input_tokens = 0
        output_tokens = 0
        
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            input_tokens = response.usage_metadata.get("input_tokens", 0)
            output_tokens = response.usage_metadata.get("output_tokens", 0)
        
        # Record the successful call
        await self._check_and_record(input_tokens, output_tokens)
        
        return response
    
    async def astream(
        self,
        messages: list[BaseMessage],
        **kwargs: Any,
    ) -> AsyncIterator[AIMessageChunk]:
        """
        Stream LLM response with usage tracking.
        
        Args:
            messages: List of messages to send
            **kwargs: Additional arguments for the LLM
            
        Yields:
            AI message chunks
            
        Raises:
            UsageLimitExceeded: If usage limits are exceeded
        """
        # Check limits before making the call
        await self._usage_service.check_limit()
        
        logger.debug(f"LLM stream by {self._agent_name} using {self._model_name}")
        
        # Track tokens during streaming (approximate)
        chunk_count = 0
        
        async for chunk in self._llm.astream(messages, **kwargs):
            chunk_count += 1
            yield chunk
        
        # Record the call after streaming completes
        # Note: Token counts are approximate for streaming
        await self._check_and_record(output_tokens=chunk_count * 4)  # Rough estimate
    
    def __getattr__(self, name: str) -> Any:
        """
        Delegate attribute access to the underlying LLM.
        
        This allows the wrapper to be used as a drop-in replacement.
        """
        return getattr(self._llm, name)


def create_usage_aware_llm(
    llm: BaseChatModel,
    usage_service: UsageService,
    agent_name: str = "unknown",
) -> UsageAwareLLM:
    """
    Factory function to create a usage-aware LLM wrapper.
    
    Args:
        llm: The LangChain LLM to wrap
        usage_service: Usage tracking service
        agent_name: Name of the agent for tracking
        
    Returns:
        UsageAwareLLM wrapper instance
    """
    return UsageAwareLLM(llm, usage_service, agent_name)

