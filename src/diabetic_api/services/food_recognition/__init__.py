"""
Food Recognition Service - Facade pattern for food identification APIs.

MVP-2.4: Provides abstraction layer for food recognition with Ollama/LLaVA as initial provider.
"""

from .base import (
    FoodRecognitionService,
    FoodRecognitionResult,
    RecognizedFood,
    EstimatedMacros,
    FoodRecognitionError,
)
from .factory import get_food_recognition_service

__all__ = [
    "FoodRecognitionService",
    "FoodRecognitionResult",
    "RecognizedFood",
    "EstimatedMacros",
    "FoodRecognitionError",
    "get_food_recognition_service",
]
