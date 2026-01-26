"""
Nutrition Lookup Service - Facade pattern for nutrition database APIs.

MVP-2.7: Provides abstraction layer for nutrition lookup with USDA FoodData Central as initial provider.
"""

from .base import (
    NutritionLookupService,
    NutritionResult,
    NutrientInfo,
    NutritionLookupError,
    MacroSource,
)
from .factory import get_nutrition_lookup_service

__all__ = [
    "NutritionLookupService",
    "NutritionResult",
    "NutrientInfo",
    "NutritionLookupError",
    "MacroSource",
    "get_nutrition_lookup_service",
]
