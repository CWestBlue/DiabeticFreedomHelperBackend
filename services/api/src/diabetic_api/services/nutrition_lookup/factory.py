"""
Factory for creating nutrition lookup service instances.

Reads configuration from settings and returns the appropriate provider.
"""

import logging
from functools import lru_cache

from diabetic_api.core.config import get_settings

from .base import NutritionLookupError, NutritionLookupService
from .usda_provider import USDANutritionLookup

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_nutrition_lookup_service() -> NutritionLookupService | None:
    """
    Get the configured nutrition lookup service.
    
    Configuration is read from settings:
    - usda_api_key: USDA FoodData Central API key
    - usda_api_base_url: API base URL
    - usda_enabled: Whether USDA lookup is enabled
    
    Returns:
        Configured NutritionLookupService instance, or None if not configured
    """
    settings = get_settings()
    
    if not settings.is_usda_configured:
        logger.warning("USDA nutrition lookup not configured (missing API key or disabled)")
        return None
    
    logger.info(f"Initializing USDA nutrition lookup service")
    
    return USDANutritionLookup(
        api_key=settings.usda_api_key,
        base_url=settings.usda_api_base_url,
        timeout=30.0,
    )


def clear_service_cache():
    """Clear the cached service instance (useful for testing)."""
    get_nutrition_lookup_service.cache_clear()
