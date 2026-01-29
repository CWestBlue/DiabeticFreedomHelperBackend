"""
Food Recognition Service - Facade pattern for food identification APIs.

MVP-2.4: Provides abstraction layer for food recognition with Ollama/LLaVA as initial provider.
MVP-2.9: Adds per-mask recognition for improved multi-item detection.
"""

from .base import (
    FoodRecognitionService,
    FoodRecognitionResult,
    RecognizedFood,
    RecognizedFoodWithMask,
    EstimatedMacros,
    FoodRecognitionError,
)
from .factory import get_food_recognition_service
from .per_mask import PerMaskFoodRecognition, create_per_mask_service

__all__ = [
    "FoodRecognitionService",
    "FoodRecognitionResult",
    "RecognizedFood",
    "RecognizedFoodWithMask",
    "EstimatedMacros",
    "FoodRecognitionError",
    "get_food_recognition_service",
    "PerMaskFoodRecognition",
    "create_per_mask_service",
]
