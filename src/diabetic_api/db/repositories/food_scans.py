"""Repository for FoodScan collection (scan metadata + results)."""

from datetime import datetime
from typing import Any

from motor.motor_asyncio import AsyncIOMotorCollection

from diabetic_api.models.food_scan import FoodScan

from .base import BaseRepository


class FoodScanRepository(BaseRepository[FoodScan]):
    """
    Repository for food scan metadata and results.
    
    Stored in dedicated `food_scans` collection (separate from `pump_data`).
    This stores the complete scan record including request metadata and results.
    """

    model_class = FoodScan

    def __init__(self, collection: AsyncIOMotorCollection):
        super().__init__(collection)

    async def create_scan(
        self,
        scan_id: str,
        user_id: str,
        device: dict[str, Any],
        intrinsics: dict[str, Any],
        orientation: dict[str, Any] | None = None,
        opt_in_store_artifacts: bool = False,
        scan_version: str = "1.0",
    ) -> str:
        """
        Create a new scan record when scan is initiated.
        
        Args:
            scan_id: Unique scan identifier
            user_id: User identifier
            device: Device information
            intrinsics: Camera intrinsics
            orientation: Device orientation (optional)
            opt_in_store_artifacts: Whether user opted in to store artifacts
            scan_version: API contract version
            
        Returns:
            Inserted document ID
        """
        document = {
            "scan_id": scan_id,
            "user_id": user_id,
            "device": device,
            "intrinsics": intrinsics,
            "scan_version": scan_version,
            "opt_in_store_artifacts": opt_in_store_artifacts,
            "created_at": datetime.utcnow(),
            # Results will be updated after processing
            "food_candidates": [],
            "selected_food": None,
            "volume_ml": None,
            "grams_est": None,
            "macros": None,
            "macro_ranges": None,
            "confidence_score": None,
            "scan_quality": None,
            "uncertainty_reasons": [],
            "processing_time_ms": None,
            "processed_at": None,
        }
        
        if orientation:
            document["orientation"] = orientation
        
        return await self.insert_one(document)

    async def update_results(
        self,
        scan_id: str,
        food_candidates: list[dict[str, Any]],
        selected_food: dict[str, Any] | None,
        volume_ml: float | None,
        grams_est: float | None,
        macros: dict[str, float] | None,
        macro_ranges: dict[str, float] | None,
        confidence_score: float | None,
        scan_quality: str | None,
        uncertainty_reasons: list[str],
        processing_time_ms: int,
    ) -> bool:
        """
        Update scan record with processing results.
        
        Args:
            scan_id: Scan identifier
            food_candidates: List of food candidates
            selected_food: Selected food candidate
            volume_ml: Estimated volume
            grams_est: Estimated weight
            macros: Estimated macros
            macro_ranges: Macro confidence ranges
            confidence_score: Overall confidence
            scan_quality: Scan quality indicator
            uncertainty_reasons: List of uncertainty reasons
            processing_time_ms: Processing time in milliseconds
            
        Returns:
            True if updated successfully
        """
        result = await self.collection.update_one(
            {"scan_id": scan_id},
            {
                "$set": {
                    "food_candidates": food_candidates,
                    "selected_food": selected_food,
                    "volume_ml": volume_ml,
                    "grams_est": grams_est,
                    "macros": macros,
                    "macro_ranges": macro_ranges,
                    "confidence_score": confidence_score,
                    "scan_quality": scan_quality,
                    "uncertainty_reasons": uncertainty_reasons,
                    "processing_time_ms": processing_time_ms,
                    "processed_at": datetime.utcnow(),
                }
            },
        )
        return result.modified_count > 0

    async def get_by_scan_id(self, scan_id: str) -> FoodScan | None:
        """
        Get scan record by scan ID.
        
        Args:
            scan_id: The scan identifier
            
        Returns:
            FoodScan or None if not found
        """
        return await self.find_one({"scan_id": scan_id})

    async def get_user_scans(
        self,
        user_id: str,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 100,
    ) -> list[FoodScan]:
        """
        Get scan records for a user within a date range.
        
        Args:
            user_id: User identifier
            start: Start datetime (inclusive)
            end: End datetime (inclusive)
            limit: Maximum results to return
            
        Returns:
            List of FoodScan objects
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

    async def get_recent_scans(
        self,
        user_id: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Get recent scans for a user with summary info.
        
        Args:
            user_id: User identifier
            limit: Maximum results
            
        Returns:
            List of scan summaries
        """
        pipeline = [
            {"$match": {"user_id": user_id}},
            {"$sort": {"created_at": -1}},
            {"$limit": limit},
            {
                "$project": {
                    "_id": 0,
                    "scan_id": 1,
                    "selected_food.label": 1,
                    "macros": 1,
                    "confidence_score": 1,
                    "scan_quality": 1,
                    "created_at": 1,
                }
            },
        ]
        return await self.aggregate(pipeline, limit=limit)

    async def delete_user_scans(self, user_id: str) -> int:
        """
        Delete all scans for a user (for account deletion/privacy).
        
        Args:
            user_id: User identifier
            
        Returns:
            Number of deleted documents
        """
        result = await self.collection.delete_many({"user_id": user_id})
        return result.deleted_count
