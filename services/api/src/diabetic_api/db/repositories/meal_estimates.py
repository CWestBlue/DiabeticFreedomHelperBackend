"""Repository for MealEstimate collection (food scan results)."""

from datetime import datetime
from typing import Any

from motor.motor_asyncio import AsyncIOMotorCollection

from diabetic_api.models.food_scan import MealEstimate

from .base import BaseRepository


class MealEstimateRepository(BaseRepository[MealEstimate]):
    """
    Repository for meal estimates from food scans.
    
    Stored in dedicated `meal_estimates` collection (separate from `pump_data`).
    This separation ensures CareLink sync workflows are not affected.
    """

    model_class = MealEstimate

    def __init__(self, collection: AsyncIOMotorCollection):
        super().__init__(collection)

    async def create_from_scan(
        self,
        scan_id: str,
        user_id: str,
        canonical_food_id: str,
        food_label: str,
        macros: dict[str, float],
        confidence: float,
        macro_ranges: dict[str, float] | None = None,
        uncertainty_reasons: list[str] | None = None,
    ) -> str:
        """
        Create a meal estimate from a food scan result.
        
        Args:
            scan_id: Reference to the original scan
            user_id: User identifier
            canonical_food_id: Food database ID
            food_label: Human-readable food name
            macros: Macronutrient values (carbs, protein, fat, fiber)
            confidence: Confidence score (0-1)
            macro_ranges: Optional P10-P90 confidence ranges
            uncertainty_reasons: List of uncertainty reason codes
            
        Returns:
            Inserted document ID
        """
        document = {
            "scan_id": scan_id,
            "user_id": user_id,
            "source": "vision",
            "canonical_food_id": canonical_food_id,
            "food_label": food_label,
            "macros": macros,
            "confidence": confidence,
            "created_at": datetime.utcnow(),
        }
        
        if macro_ranges:
            document["macro_ranges"] = macro_ranges
        
        if uncertainty_reasons:
            document["uncertainty_reasons"] = uncertainty_reasons
        else:
            document["uncertainty_reasons"] = []
        
        return await self.insert_one(document)

    async def get_by_scan_id(self, scan_id: str) -> MealEstimate | None:
        """
        Get meal estimate by scan ID.
        
        Args:
            scan_id: The scan identifier
            
        Returns:
            MealEstimate or None if not found
        """
        return await self.find_one({"scan_id": scan_id})

    async def get_user_meals(
        self,
        user_id: str,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 100,
    ) -> list[MealEstimate]:
        """
        Get meal estimates for a user within a date range.
        
        Args:
            user_id: User identifier
            start: Start datetime (inclusive)
            end: End datetime (inclusive)
            limit: Maximum results to return
            
        Returns:
            List of MealEstimate objects
        """
        filter_query: dict[str, Any] = {"user_id": user_id}
        
        if start or end:
            filter_query["created_at"] = {}
            if start:
                filter_query["created_at"]["$gte"] = start
            if end:
                filter_query["created_at"]["$lte"] = end
        
        return await self.find_many(
            filter=filter_query,
            sort=[("created_at", -1)],
            limit=limit,
        )

    async def get_timeline_data(
        self,
        user_id: str,
        start: datetime,
        end: datetime,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """
        Get meal estimates formatted for timeline display.
        
        Returns simplified data for combining with pump_data in timeline views.
        
        Args:
            user_id: User identifier
            start: Start datetime
            end: End datetime
            limit: Maximum results
            
        Returns:
            List of timeline-formatted meal entries
        """
        pipeline = [
            {
                "$match": {
                    "user_id": user_id,
                    "created_at": {"$gte": start, "$lte": end},
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "id": {"$toString": "$_id"},
                    "timestamp": "$created_at",
                    "type": {"$literal": "meal_scan"},
                    "food_label": 1,
                    "carbs": "$macros.carbs",
                    "protein": "$macros.protein",
                    "fat": "$macros.fat",
                    "fiber": "$macros.fiber",
                    "confidence": 1,
                    "scan_id": 1,
                    "source": 1,
                }
            },
            {"$sort": {"timestamp": -1}},
            {"$limit": limit},
        ]
        return await self.aggregate(pipeline, limit=limit)

    async def update_user_overrides(
        self,
        meal_id: str,
        overrides: dict[str, Any],
    ) -> bool:
        """
        Update user corrections/overrides on a meal estimate.
        
        Args:
            meal_id: Meal estimate document ID
            overrides: User override data (corrected macros, food selection, etc.)
            
        Returns:
            True if updated successfully
        """
        return await self.update_one(
            meal_id,
            {"user_overrides": overrides},
        )

    async def get_daily_summary(
        self,
        user_id: str,
        date: datetime,
    ) -> dict[str, Any]:
        """
        Get daily meal summary for a user.
        
        Args:
            user_id: User identifier
            date: Date to summarize
            
        Returns:
            Dictionary with daily totals
        """
        start = datetime(date.year, date.month, date.day, 0, 0, 0)
        end = datetime(date.year, date.month, date.day, 23, 59, 59)
        
        pipeline = [
            {
                "$match": {
                    "user_id": user_id,
                    "created_at": {"$gte": start, "$lte": end},
                }
            },
            {
                "$group": {
                    "_id": None,
                    "total_carbs": {"$sum": "$macros.carbs"},
                    "total_protein": {"$sum": "$macros.protein"},
                    "total_fat": {"$sum": "$macros.fat"},
                    "total_fiber": {"$sum": "$macros.fiber"},
                    "meal_count": {"$sum": 1},
                    "avg_confidence": {"$avg": "$confidence"},
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "total_carbs": {"$round": ["$total_carbs", 1]},
                    "total_protein": {"$round": ["$total_protein", 1]},
                    "total_fat": {"$round": ["$total_fat", 1]},
                    "total_fiber": {"$round": ["$total_fiber", 1]},
                    "meal_count": 1,
                    "avg_confidence": {"$round": ["$avg_confidence", 2]},
                }
            },
        ]
        
        results = await self.aggregate(pipeline, limit=1)
        if results:
            return results[0]
        return {
            "total_carbs": 0,
            "total_protein": 0,
            "total_fat": 0,
            "total_fiber": 0,
            "meal_count": 0,
            "avg_confidence": None,
        }
