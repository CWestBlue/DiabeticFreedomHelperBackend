"""Food Search API routes.

MVP-3.2: Endpoints for searching USDA FoodData Central and fetching nutrition info.
Enables users to correct food identification and get accurate macros.
"""

import logging
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from diabetic_api.services.nutrition_lookup import (
    NutritionLookupError,
    get_nutrition_lookup_service,
)

router = APIRouter()
logger = logging.getLogger(__name__)


# =============================================================================
# Response Models
# =============================================================================


class MacrosPer100g(BaseModel):
    """Macronutrients per 100 grams."""

    carbs: float = Field(..., ge=0, description="Carbohydrates in grams per 100g")
    protein: float = Field(..., ge=0, description="Protein in grams per 100g")
    fat: float = Field(..., ge=0, description="Fat in grams per 100g")
    fiber: float = Field(..., ge=0, description="Fiber in grams per 100g")
    calories: float | None = Field(None, ge=0, description="Calories per 100g")


class SearchedFood(BaseModel):
    """A food item from search results."""

    id: str = Field(..., description="USDA FoodData Central ID")
    name: str = Field(..., description="Food name/description")
    macros_per_100g: MacrosPer100g = Field(..., description="Macros per 100 grams")
    data_type: str | None = Field(
        None, description="USDA data type (Branded, Foundation, SR Legacy, etc.)"
    )
    brand: str | None = Field(None, description="Brand name for branded foods")


class FoodSearchResponse(BaseModel):
    """Response from food search endpoint."""

    foods: list[SearchedFood] = Field(
        default_factory=list, description="List of matching foods"
    )
    total_results: int = Field(..., description="Total number of results found")
    query: str = Field(..., description="Original search query")


class FoodDetailResponse(BaseModel):
    """Response from food detail endpoint."""

    id: str = Field(..., description="USDA FoodData Central ID")
    name: str = Field(..., description="Food name/description")
    macros_per_100g: MacrosPer100g = Field(..., description="Macros per 100 grams")
    data_type: str | None = Field(None, description="USDA data type")
    serving_description: str | None = Field(
        None, description="Default serving description"
    )


class FoodSearchError(BaseModel):
    """Error response for food search."""

    error: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Error code")


# =============================================================================
# Endpoints
# =============================================================================


@router.get(
    "/search",
    response_model=FoodSearchResponse,
    responses={
        200: {"description": "Search results"},
        400: {"description": "Invalid query", "model": FoodSearchError},
        503: {"description": "USDA service unavailable", "model": FoodSearchError},
    },
    summary="Search USDA FoodData Central",
    description="""
Search the USDA FoodData Central database for foods matching the query.
Returns foods with their macros per 100g for client-side portion scaling.

Use this to let users correct AI food identification with verified USDA data.
""",
)
async def search_foods(
    q: Annotated[
        str,
        Query(
            min_length=2,
            max_length=200,
            description="Search query (food name or description)",
        ),
    ],
    limit: Annotated[
        int,
        Query(ge=1, le=25, description="Maximum number of results to return"),
    ] = 10,
) -> FoodSearchResponse:
    """Search for foods in USDA database."""
    logger.info(f"Food search request: q='{q}', limit={limit}")

    # Get nutrition lookup service
    nutrition_service = get_nutrition_lookup_service()
    if nutrition_service is None:
        logger.error("USDA nutrition service not configured")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "Nutrition lookup service not available",
                "error_code": "SERVICE_UNAVAILABLE",
            },
        )

    try:
        # Search USDA
        result = await nutrition_service.search_food(q, max_results=limit)

        foods: list[SearchedFood] = []

        # Add primary result if found
        if result.found and result.nutrients:
            foods.append(
                SearchedFood(
                    id=result.food_id or "",
                    name=result.food_name or "Unknown",
                    macros_per_100g=MacrosPer100g(
                        carbs=result.nutrients.carbs,
                        protein=result.nutrients.protein,
                        fat=result.nutrients.fat,
                        fiber=result.nutrients.fiber,
                        calories=result.nutrients.calories,
                    ),
                    data_type=result.data_type,
                    brand=None,  # Primary result doesn't include brand in current impl
                )
            )

        # Add alternative matches
        for alt in result.alternative_matches:
            # Skip if we already have this food
            if any(f.id == str(alt.get("id")) for f in foods):
                continue

            # For alternatives, we need to fetch full details to get macros
            # For MVP, we'll include them without macros and let client fetch on selection
            # Or we can make a follow-up call - but that's slow
            # For now, just include basic info - client will call GET /{id} when selected
            foods.append(
                SearchedFood(
                    id=str(alt.get("id", "")),
                    name=alt.get("name", "Unknown"),
                    macros_per_100g=MacrosPer100g(
                        carbs=0,
                        protein=0,
                        fat=0,
                        fiber=0,
                        calories=None,
                    ),  # Will be fetched when selected
                    data_type=alt.get("data_type"),
                    brand=alt.get("brand"),
                )
            )

        total_results = len(foods)
        logger.info(f"Food search returned {total_results} results for '{q}'")

        return FoodSearchResponse(
            foods=foods[:limit],
            total_results=total_results,
            query=q,
        )

    except NutritionLookupError as e:
        logger.error(f"USDA lookup error: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": e.message,
                "error_code": e.error_code,
            },
        )
    except Exception as e:
        logger.exception(f"Unexpected error in food search: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Internal server error",
                "error_code": "INTERNAL_ERROR",
            },
        )


@router.get(
    "/{food_id}",
    response_model=FoodDetailResponse,
    responses={
        200: {"description": "Food details"},
        404: {"description": "Food not found", "model": FoodSearchError},
        503: {"description": "USDA service unavailable", "model": FoodSearchError},
    },
    summary="Get food details by USDA ID",
    description="""
Get detailed nutrition information for a specific food by its USDA FoodData Central ID.
Returns macros per 100g for client-side portion scaling.
""",
)
async def get_food_by_id(food_id: str) -> FoodDetailResponse:
    """Get food details by USDA FoodData Central ID."""
    logger.info(f"Food detail request: food_id={food_id}")

    # Get nutrition lookup service
    nutrition_service = get_nutrition_lookup_service()
    if nutrition_service is None:
        logger.error("USDA nutrition service not configured")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "Nutrition lookup service not available",
                "error_code": "SERVICE_UNAVAILABLE",
            },
        )

    try:
        # Fetch from USDA
        result = await nutrition_service.get_food_by_id(food_id)

        if not result.found or not result.nutrients:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": f"Food with ID '{food_id}' not found",
                    "error_code": "NOT_FOUND",
                },
            )

        return FoodDetailResponse(
            id=result.food_id or food_id,
            name=result.food_name or "Unknown",
            macros_per_100g=MacrosPer100g(
                carbs=result.nutrients.carbs,
                protein=result.nutrients.protein,
                fat=result.nutrients.fat,
                fiber=result.nutrients.fiber,
                calories=result.nutrients.calories,
            ),
            data_type=result.data_type,
            serving_description=result.nutrients.serving_description,
        )

    except HTTPException:
        raise
    except NutritionLookupError as e:
        logger.error(f"USDA lookup error: {e.message}")
        if "not found" in e.message.lower() or e.error_code == "NOT_FOUND":
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": f"Food with ID '{food_id}' not found",
                    "error_code": "NOT_FOUND",
                },
            )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": e.message,
                "error_code": e.error_code,
            },
        )
    except Exception as e:
        logger.exception(f"Unexpected error fetching food {food_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Internal server error",
                "error_code": "INTERNAL_ERROR",
            },
        )
