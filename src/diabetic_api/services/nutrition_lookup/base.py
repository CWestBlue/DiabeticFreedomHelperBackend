"""
Base classes and models for nutrition lookup service.

Defines the abstract interface that all providers must implement,
plus standardized response models.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MacroSource(str, Enum):
    """Source of macro nutrient data."""
    
    USDA = "usda"  # USDA FoodData Central
    LLAVA = "llava"  # AI/LLaVA estimate
    USER_OVERRIDE = "user_override"  # User-provided values
    UNKNOWN = "unknown"


class NutrientInfo(BaseModel):
    """Detailed nutrient information from database."""
    
    # Primary macros (grams per serving)
    carbs: float = Field(ge=0, description="Carbohydrates in grams")
    protein: float = Field(ge=0, description="Protein in grams")
    fat: float = Field(ge=0, description="Total fat in grams")
    fiber: float = Field(ge=0, default=0, description="Dietary fiber in grams")
    
    # Additional nutrients (optional)
    calories: float | None = Field(None, ge=0, description="Energy in kcal")
    sugar: float | None = Field(None, ge=0, description="Total sugars in grams")
    saturated_fat: float | None = Field(None, ge=0, description="Saturated fat in grams")
    sodium: float | None = Field(None, ge=0, description="Sodium in mg")
    
    # Serving info
    serving_size_g: float = Field(100, ge=0, description="Serving size in grams")
    serving_description: str | None = Field(None, description="Human-readable serving description")
    
    def scale_to_grams(self, grams: float) -> "NutrientInfo":
        """Scale nutrients to a specific gram amount."""
        if self.serving_size_g == 0:
            return self
        
        scale = grams / self.serving_size_g
        return NutrientInfo(
            carbs=self.carbs * scale,
            protein=self.protein * scale,
            fat=self.fat * scale,
            fiber=self.fiber * scale,
            calories=self.calories * scale if self.calories else None,
            sugar=self.sugar * scale if self.sugar else None,
            saturated_fat=self.saturated_fat * scale if self.saturated_fat else None,
            sodium=self.sodium * scale if self.sodium else None,
            serving_size_g=grams,
            serving_description=f"{grams}g",
        )


class NutritionResult(BaseModel):
    """Complete result from nutrition lookup."""
    
    # Match info
    found: bool = Field(..., description="Whether a match was found")
    food_id: str | None = Field(None, description="Database food ID (e.g., USDA FDC ID)")
    food_name: str | None = Field(None, description="Official food name from database")
    
    # Nutrients
    nutrients: NutrientInfo | None = Field(None, description="Nutrient information")
    
    # Match quality
    match_score: float = Field(
        0.0, ge=0.0, le=1.0, description="How well the search matched (0-1)"
    )
    search_query: str = Field("", description="Original search query")
    
    # Provider info
    provider: str = Field(..., description="Provider that generated this result")
    data_type: str | None = Field(
        None, description="USDA data type: 'Branded', 'Foundation', 'SR Legacy', etc."
    )
    
    # Additional matches (for user selection)
    alternative_matches: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Other potential matches with id, name, score"
    )


class NutritionLookupError(Exception):
    """Error during nutrition lookup."""
    
    def __init__(
        self,
        message: str,
        error_code: str = "LOOKUP_ERROR",
        provider: str = "unknown",
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.provider = provider
        self.details = details or {}


class NutritionLookupService(ABC):
    """
    Abstract base class for nutrition lookup services.
    
    All providers (USDA, Nutritionix, etc.) must implement this interface.
    """
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of this provider."""
        ...
    
    @abstractmethod
    async def search_food(
        self,
        query: str,
        *,
        max_results: int = 5,
    ) -> NutritionResult:
        """
        Search for a food and return nutrition information.
        
        Args:
            query: Food name/description to search for
            max_results: Maximum number of alternative matches to include
            
        Returns:
            NutritionResult with best match and alternatives
            
        Raises:
            NutritionLookupError: If lookup fails
        """
        ...
    
    @abstractmethod
    async def get_food_by_id(
        self,
        food_id: str,
    ) -> NutritionResult:
        """
        Get nutrition info for a specific food by database ID.
        
        Args:
            food_id: Database-specific food identifier
            
        Returns:
            NutritionResult with nutrition data
            
        Raises:
            NutritionLookupError: If lookup fails
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
