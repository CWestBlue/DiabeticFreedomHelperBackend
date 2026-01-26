"""
Base classes and models for food recognition service.

Defines the abstract interface that all providers must implement,
plus standardized response models.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class FoodCategory(str, Enum):
    """Category of food item."""
    
    PROTEIN = "protein"
    CARBOHYDRATE = "carbohydrate"
    VEGETABLE = "vegetable"
    FRUIT = "fruit"
    DAIRY = "dairy"
    FAT = "fat"
    BEVERAGE = "beverage"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class EstimatedMacros(BaseModel):
    """Estimated macronutrients for a food item."""
    
    carbs: float = Field(ge=0, description="Estimated carbohydrates in grams")
    protein: float = Field(ge=0, description="Estimated protein in grams")
    fat: float = Field(ge=0, description="Estimated fat in grams")
    fiber: float = Field(ge=0, default=0, description="Estimated fiber in grams")
    
    @property
    def calories(self) -> float:
        """Estimate calories using 4-4-9-2 formula."""
        return (self.carbs * 4) + (self.protein * 4) + (self.fat * 9) + (self.fiber * 2)


class RecognizedFood(BaseModel):
    """A single recognized food item from the image."""
    
    label: str = Field(..., description="Human-readable food name")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence score 0-1"
    )
    estimated_grams: float | None = Field(
        None, ge=0, description="Estimated weight in grams"
    )
    estimated_macros: EstimatedMacros | None = Field(
        None, description="Estimated macronutrients"
    )
    category: FoodCategory = Field(
        FoodCategory.UNKNOWN, description="Food category"
    )
    is_mixed_dish: bool = Field(
        False, description="Whether this is a mixed/composite dish"
    )
    visible_components: list[str] | None = Field(
        None, description="Visible components for mixed dishes (MVP-3.4)"
    )
    
    # For matching with nutrition database
    possible_usda_matches: list[str] = Field(
        default_factory=list,
        description="Possible USDA FoodData Central matches"
    )


class FoodRecognitionResult(BaseModel):
    """Complete result from food recognition."""
    
    foods: list[RecognizedFood] = Field(
        default_factory=list, description="Identified foods in the image"
    )
    overall_confidence: float = Field(
        ge=0.0, le=1.0, description="Overall confidence in recognition"
    )
    is_multi_food_plate: bool = Field(
        False, description="Whether multiple distinct foods were detected (MVP-3.4)"
    )
    raw_response: str = Field(
        "", description="Raw response from the provider (for debugging)"
    )
    provider: str = Field(
        ..., description="Provider that generated this result"
    )
    processing_time_ms: int = Field(
        0, ge=0, description="Time taken to process in milliseconds"
    )
    
    @property
    def primary_food(self) -> RecognizedFood | None:
        """Get the highest confidence food item."""
        if not self.foods:
            return None
        return max(self.foods, key=lambda f: f.confidence)
    
    @property
    def total_estimated_macros(self) -> EstimatedMacros | None:
        """Sum macros from all recognized foods."""
        foods_with_macros = [f for f in self.foods if f.estimated_macros]
        if not foods_with_macros:
            return None
        
        return EstimatedMacros(
            carbs=sum(f.estimated_macros.carbs for f in foods_with_macros),
            protein=sum(f.estimated_macros.protein for f in foods_with_macros),
            fat=sum(f.estimated_macros.fat for f in foods_with_macros),
            fiber=sum(f.estimated_macros.fiber for f in foods_with_macros),
        )


class FoodRecognitionError(Exception):
    """Error during food recognition."""
    
    def __init__(
        self,
        message: str,
        error_code: str = "RECOGNITION_ERROR",
        provider: str = "unknown",
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.provider = provider
        self.details = details or {}


class FoodRecognitionService(ABC):
    """
    Abstract base class for food recognition services.
    
    All providers (Ollama, OpenAI, Clarifai, etc.) must implement this interface.
    """
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of this provider."""
        ...
    
    @abstractmethod
    async def recognize(
        self,
        image_data: bytes,
        *,
        max_foods: int = 5,
        include_macros: bool = True,
        include_portions: bool = True,
    ) -> FoodRecognitionResult:
        """
        Recognize foods in an image.
        
        Args:
            image_data: Raw image bytes (JPEG or PNG)
            max_foods: Maximum number of food items to identify
            include_macros: Whether to estimate macronutrients
            include_portions: Whether to estimate portion sizes
            
        Returns:
            FoodRecognitionResult with identified foods
            
        Raises:
            FoodRecognitionError: If recognition fails
        """
        ...
    
    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the provider is available and healthy.
        
        Returns:
            True if the provider is ready to accept requests
        """
        ...
