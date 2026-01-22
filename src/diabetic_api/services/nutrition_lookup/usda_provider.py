"""
USDA FoodData Central provider for nutrition lookup.

Uses the USDA FDC API to search for foods and retrieve nutrition information.
API Documentation: https://fdc.nal.usda.gov/api-guide.html
"""

import logging
from typing import Any

import httpx

from .base import (
    MacroSource,
    NutrientInfo,
    NutritionLookupError,
    NutritionLookupService,
    NutritionResult,
)

logger = logging.getLogger(__name__)


# USDA nutrient IDs for macros we care about
NUTRIENT_IDS = {
    "energy": 1008,  # Energy (kcal)
    "protein": 1003,  # Protein
    "fat": 1004,  # Total lipid (fat)
    "carbs": 1005,  # Carbohydrate, by difference
    "fiber": 1079,  # Fiber, total dietary
    "sugar": 2000,  # Total Sugars
    "saturated_fat": 1258,  # Fatty acids, total saturated
    "sodium": 1093,  # Sodium, Na
}


class USDANutritionLookup(NutritionLookupService):
    """
    Nutrition lookup using USDA FoodData Central API.
    """
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.nal.usda.gov/fdc/v1",
        timeout: float = 30.0,
    ):
        """
        Initialize USDA provider.
        
        Args:
            api_key: USDA FoodData Central API key
            base_url: API base URL
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)
    
    @property
    def provider_name(self) -> str:
        return "usda"
    
    async def search_food(
        self,
        query: str,
        *,
        max_results: int = 5,
    ) -> NutritionResult:
        """
        Search USDA FoodData Central for a food.
        
        Prefers Foundation and SR Legacy data types for accuracy.
        """
        try:
            # Search endpoint
            search_url = f"{self.base_url}/foods/search"
            
            params = {
                "api_key": self.api_key,
                "query": query,
                "pageSize": max_results + 5,  # Get extra to filter
                # Prefer foundation/standard reference data over branded
                "dataType": ["Foundation", "SR Legacy", "Survey (FNDDS)"],
            }
            
            logger.info(f"Searching USDA for: {query}")
            
            response = await self._client.get(search_url, params=params)
            
            if response.status_code != 200:
                logger.error(f"USDA search failed: {response.status_code} - {response.text}")
                return NutritionResult(
                    found=False,
                    provider=self.provider_name,
                    search_query=query,
                )
            
            data = response.json()
            foods = data.get("foods", [])
            
            if not foods:
                logger.info(f"No USDA results for: {query}")
                return NutritionResult(
                    found=False,
                    provider=self.provider_name,
                    search_query=query,
                )
            
            # Get best match (first result after filtering)
            best_match = foods[0]
            
            # Extract nutrients from search result
            nutrients = self._extract_nutrients_from_search(best_match)
            
            # Build alternative matches
            alternatives = []
            for food in foods[1:max_results]:
                alternatives.append({
                    "id": str(food.get("fdcId")),
                    "name": food.get("description", "Unknown"),
                    "data_type": food.get("dataType"),
                    "brand": food.get("brandOwner"),
                })
            
            # Calculate match score based on search score or position
            search_score = best_match.get("score", 0)
            # Normalize score (USDA scores vary widely)
            match_score = min(1.0, search_score / 1000) if search_score else 0.8
            
            return NutritionResult(
                found=True,
                food_id=str(best_match.get("fdcId")),
                food_name=best_match.get("description"),
                nutrients=nutrients,
                match_score=match_score,
                search_query=query,
                provider=self.provider_name,
                data_type=best_match.get("dataType"),
                alternative_matches=alternatives,
            )
            
        except httpx.RequestError as e:
            logger.error(f"USDA request failed: {e}")
            raise NutritionLookupError(
                message=f"Failed to connect to USDA API: {e}",
                error_code="CONNECTION_ERROR",
                provider=self.provider_name,
            ) from e
        except Exception as e:
            logger.exception("Unexpected error in USDA search")
            raise NutritionLookupError(
                message=f"Unexpected error: {e}",
                error_code="UNEXPECTED_ERROR",
                provider=self.provider_name,
            ) from e
    
    async def get_food_by_id(
        self,
        food_id: str,
    ) -> NutritionResult:
        """
        Get detailed nutrition info for a specific USDA food.
        """
        try:
            url = f"{self.base_url}/food/{food_id}"
            params = {"api_key": self.api_key}
            
            logger.info(f"Fetching USDA food: {food_id}")
            
            response = await self._client.get(url, params=params)
            
            if response.status_code == 404:
                return NutritionResult(
                    found=False,
                    food_id=food_id,
                    provider=self.provider_name,
                    search_query=food_id,
                )
            
            if response.status_code != 200:
                raise NutritionLookupError(
                    message=f"USDA API error: {response.status_code}",
                    error_code="API_ERROR",
                    provider=self.provider_name,
                )
            
            food = response.json()
            nutrients = self._extract_nutrients_from_detail(food)
            
            return NutritionResult(
                found=True,
                food_id=food_id,
                food_name=food.get("description"),
                nutrients=nutrients,
                match_score=1.0,  # Exact ID match
                search_query=food_id,
                provider=self.provider_name,
                data_type=food.get("dataType"),
            )
            
        except NutritionLookupError:
            raise
        except Exception as e:
            logger.exception(f"Error fetching USDA food {food_id}")
            raise NutritionLookupError(
                message=f"Failed to fetch food: {e}",
                error_code="FETCH_ERROR",
                provider=self.provider_name,
            ) from e
    
    def _extract_nutrients_from_search(self, food: dict[str, Any]) -> NutrientInfo:
        """Extract nutrients from search result format."""
        nutrients = food.get("foodNutrients", [])
        return self._parse_nutrients(nutrients, is_search_format=True)
    
    def _extract_nutrients_from_detail(self, food: dict[str, Any]) -> NutrientInfo:
        """Extract nutrients from detail/full food format."""
        nutrients = food.get("foodNutrients", [])
        
        # Get serving size if available
        serving_size = 100.0  # Default per 100g
        serving_desc = "100g"
        
        portions = food.get("foodPortions", [])
        if portions:
            # Use first portion as reference
            portion = portions[0]
            serving_size = portion.get("gramWeight", 100.0)
            serving_desc = portion.get("portionDescription") or f"{serving_size}g"
        
        nutrient_info = self._parse_nutrients(nutrients, is_search_format=False)
        nutrient_info.serving_size_g = serving_size
        nutrient_info.serving_description = serving_desc
        
        return nutrient_info
    
    def _parse_nutrients(
        self, 
        nutrients: list[dict[str, Any]], 
        is_search_format: bool = True
    ) -> NutrientInfo:
        """Parse nutrients list into NutrientInfo model."""
        values: dict[str, float | None] = {
            "carbs": 0.0,
            "protein": 0.0,
            "fat": 0.0,
            "fiber": 0.0,
            "calories": None,
            "sugar": None,
            "saturated_fat": None,
            "sodium": None,
        }
        
        for nutrient in nutrients:
            # Search results use nutrientId, detail uses nutrient.id
            if is_search_format:
                nutrient_id = nutrient.get("nutrientId")
                value = nutrient.get("value", 0)
            else:
                nutrient_obj = nutrient.get("nutrient", {})
                nutrient_id = nutrient_obj.get("id")
                value = nutrient.get("amount", 0)
            
            if nutrient_id is None or value is None:
                continue
            
            # Map to our fields
            if nutrient_id == NUTRIENT_IDS["carbs"]:
                values["carbs"] = float(value)
            elif nutrient_id == NUTRIENT_IDS["protein"]:
                values["protein"] = float(value)
            elif nutrient_id == NUTRIENT_IDS["fat"]:
                values["fat"] = float(value)
            elif nutrient_id == NUTRIENT_IDS["fiber"]:
                values["fiber"] = float(value)
            elif nutrient_id == NUTRIENT_IDS["energy"]:
                values["calories"] = float(value)
            elif nutrient_id == NUTRIENT_IDS["sugar"]:
                values["sugar"] = float(value)
            elif nutrient_id == NUTRIENT_IDS["saturated_fat"]:
                values["saturated_fat"] = float(value)
            elif nutrient_id == NUTRIENT_IDS["sodium"]:
                values["sodium"] = float(value)
        
        return NutrientInfo(
            carbs=values["carbs"] or 0.0,
            protein=values["protein"] or 0.0,
            fat=values["fat"] or 0.0,
            fiber=values["fiber"] or 0.0,
            calories=values["calories"],
            sugar=values["sugar"],
            saturated_fat=values["saturated_fat"],
            sodium=values["sodium"],
            serving_size_g=100.0,  # USDA data is per 100g
            serving_description="per 100g",
        )
    
    async def health_check(self) -> bool:
        """Check if USDA API is available."""
        try:
            # Simple search to verify API key works
            url = f"{self.base_url}/foods/search"
            params = {
                "api_key": self.api_key,
                "query": "apple",
                "pageSize": 1,
            }
            
            response = await self._client.get(url, params=params)
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"USDA health check failed: {e}")
            return False
    
    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()
