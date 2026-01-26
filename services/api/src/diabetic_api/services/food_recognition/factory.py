"""
Factory for creating food recognition service instances.

Reads configuration from environment variables and returns the appropriate provider.
"""

import logging
import os
from functools import lru_cache

from .base import FoodRecognitionService, FoodRecognitionError
from .ollama_provider import OllamaFoodRecognition

logger = logging.getLogger(__name__)


# Supported providers
PROVIDERS = {
    "ollama": OllamaFoodRecognition,
    # Future providers:
    # "openai": OpenAIFoodRecognition,
    # "clarifai": ClarifaiFoodRecognition,
    # "huggingface": HuggingFaceFoodRecognition,
}


@lru_cache(maxsize=1)
def get_food_recognition_service() -> FoodRecognitionService:
    """
    Get the configured food recognition service.
    
    Configuration is read from environment variables:
    - FOOD_RECOGNITION_PROVIDER: Provider name (default: "ollama")
    - OLLAMA_BASE_URL: Ollama API URL (default: "http://localhost:11434")
    - OLLAMA_MODEL: Model to use (default: "llava:7b")
    
    Returns:
        Configured FoodRecognitionService instance
        
    Raises:
        FoodRecognitionError: If provider is not supported or configuration is invalid
    """
    provider_name = os.getenv("FOOD_RECOGNITION_PROVIDER", "ollama").lower()
    
    logger.info(f"Initializing food recognition provider: {provider_name}")
    
    if provider_name not in PROVIDERS:
        raise FoodRecognitionError(
            message=f"Unknown food recognition provider: {provider_name}",
            error_code="INVALID_PROVIDER",
            provider=provider_name,
            details={"supported_providers": list(PROVIDERS.keys())},
        )
    
    if provider_name == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        model = os.getenv("OLLAMA_MODEL", "llava:7b")
        timeout = float(os.getenv("OLLAMA_TIMEOUT", "60"))
        
        logger.info(f"Configuring Ollama provider: {base_url}, model={model}")
        
        return OllamaFoodRecognition(
            base_url=base_url,
            model=model,
            timeout=timeout,
        )
    
    # Add other providers here as they're implemented
    raise FoodRecognitionError(
        message=f"Provider {provider_name} is not yet implemented",
        error_code="NOT_IMPLEMENTED",
        provider=provider_name,
    )


def clear_service_cache():
    """Clear the cached service instance (useful for testing)."""
    get_food_recognition_service.cache_clear()
