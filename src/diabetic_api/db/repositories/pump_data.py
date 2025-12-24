"""Repository for PumpData collection (glucose readings, boluses, etc.)."""

from datetime import datetime
from typing import Any

from motor.motor_asyncio import AsyncIOMotorCollection

from .base import BaseRepository


class PumpDataRepository(BaseRepository):
    """
    Repository for Medtronic pump data.
    
    Handles glucose readings, boluses, basal rates, and other pump data.
    Most fields are stored as strings in MongoDB and require conversion.
    """

    def __init__(self, collection: AsyncIOMotorCollection):
        super().__init__(collection)

    async def get_glucose_readings(
        self,
        start: datetime,
        end: datetime,
        limit: int = 10000,
    ) -> list[dict[str, Any]]:
        """
        Get glucose readings within a date range.
        
        Args:
            start: Start datetime (inclusive)
            end: End datetime (inclusive)
            limit: Maximum readings to return
            
        Returns:
            List of glucose reading documents
        """
        pipeline = [
            {
                "$match": {
                    "Timestamp": {"$gte": start, "$lte": end},
                    "Sensor Glucose (mg/dL)": {"$ne": ""},
                }
            },
            {
                "$addFields": {
                    "glucose_value": {"$toDouble": "$Sensor Glucose (mg/dL)"}
                }
            },
            {
                "$match": {"glucose_value": {"$ne": None}}
            },
            {
                "$project": {
                    "_id": 0,
                    "timestamp": "$Timestamp",
                    "value": "$glucose_value",
                }
            },
            {"$sort": {"timestamp": 1}},
            {"$limit": limit},
        ]
        return await self.aggregate(pipeline, limit=limit)

    async def get_boluses(
        self,
        start: datetime,
        end: datetime,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """
        Get bolus deliveries within a date range.
        
        Args:
            start: Start datetime (inclusive)
            end: End datetime (inclusive)
            limit: Maximum boluses to return
            
        Returns:
            List of bolus documents
        """
        pipeline = [
            {
                "$match": {
                    "Timestamp": {"$gte": start, "$lte": end},
                    "$or": [
                        {"Bolus Volume Delivered (U)": {"$ne": ""}},
                        {"Final Bolus Estimate": {"$ne": ""}},
                    ],
                }
            },
            {
                "$addFields": {
                    "bolus_delivered": {
                        "$toDouble": {
                            "$ifNull": ["$Bolus Volume Delivered (U)", "0"]
                        }
                    },
                    "bolus_estimate": {
                        "$toDouble": {
                            "$ifNull": ["$Final Bolus Estimate", "0"]
                        }
                    },
                    "carbs": {
                        "$toDouble": {
                            "$ifNull": ["$BWZ Carb Input (grams)", "0"]
                        }
                    },
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "timestamp": "$Timestamp",
                    "bolus_delivered": 1,
                    "bolus_estimate": 1,
                    "carbs": 1,
                }
            },
            {"$sort": {"timestamp": 1}},
            {"$limit": limit},
        ]
        return await self.aggregate(pipeline, limit=limit)

    async def get_metrics_aggregation(
        self,
        start: datetime,
        end: datetime,
    ) -> dict[str, Any]:
        """
        Get aggregated metrics for dashboard.
        
        Args:
            start: Start datetime
            end: End datetime
            
        Returns:
            Dictionary with aggregated metrics
        """
        pipeline = [
            {
                "$match": {
                    "Timestamp": {"$gte": start, "$lte": end},
                }
            },
            {
                "$addFields": {
                    "glucose": {
                        "$cond": {
                            "if": {"$ne": ["$Sensor Glucose (mg/dL)", ""]},
                            "then": {"$toDouble": "$Sensor Glucose (mg/dL)"},
                            "else": None,
                        }
                    },
                    "bolus": {
                        "$cond": {
                            "if": {"$ne": ["$Bolus Volume Delivered (U)", ""]},
                            "then": {"$toDouble": "$Bolus Volume Delivered (U)"},
                            "else": 0,
                        }
                    },
                    "carbs": {
                        "$cond": {
                            "if": {"$ne": ["$BWZ Carb Input (grams)", ""]},
                            "then": {"$toDouble": "$BWZ Carb Input (grams)"},
                            "else": 0,
                        }
                    },
                    "date_str": {"$dateToString": {"format": "%Y-%m-%d", "date": "$Timestamp"}},
                }
            },
            {
                "$group": {
                    "_id": None,
                    "avg_glucose": {"$avg": "$glucose"},
                    "min_glucose": {"$min": "$glucose"},
                    "max_glucose": {"$max": "$glucose"},
                    "total_readings": {
                        "$sum": {"$cond": [{"$ne": ["$glucose", None]}, 1, 0]}
                    },
                    "total_bolus": {"$sum": "$bolus"},
                    "total_carbs": {"$sum": "$carbs"},
                    "bolus_count": {
                        "$sum": {"$cond": [{"$gt": ["$bolus", 0]}, 1, 0]}
                    },
                    "unique_days": {"$addToSet": "$date_str"},
                    # Time in range calculations
                    "in_range_count": {
                        "$sum": {
                            "$cond": [
                                {
                                    "$and": [
                                        {"$gte": ["$glucose", 70]},
                                        {"$lte": ["$glucose", 180]},
                                    ]
                                },
                                1,
                                0,
                            ]
                        }
                    },
                    "low_count": {
                        "$sum": {"$cond": [{"$lt": ["$glucose", 70]}, 1, 0]}
                    },
                    "high_count": {
                        "$sum": {"$cond": [{"$gt": ["$glucose", 180]}, 1, 0]}
                    },
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "avg_glucose": {"$round": ["$avg_glucose", 1]},
                    "min_glucose": 1,
                    "max_glucose": 1,
                    "total_readings": 1,
                    "total_bolus": {"$round": ["$total_bolus", 2]},
                    "total_carbs": {"$round": ["$total_carbs", 0]},
                    "bolus_count": 1,
                    "days_count": {"$size": "$unique_days"},
                    "time_in_range": {
                        "$round": [
                            {
                                "$multiply": [
                                    {"$divide": ["$in_range_count", "$total_readings"]},
                                    100,
                                ]
                            },
                            1,
                        ]
                    },
                    "time_low": {
                        "$round": [
                            {
                                "$multiply": [
                                    {"$divide": ["$low_count", "$total_readings"]},
                                    100,
                                ]
                            },
                            1,
                        ]
                    },
                    "time_high": {
                        "$round": [
                            {
                                "$multiply": [
                                    {"$divide": ["$high_count", "$total_readings"]},
                                    100,
                                ]
                            },
                            1,
                        ]
                    },
                }
            },
        ]
        
        results = await self.aggregate(pipeline, limit=1)
        return results[0] if results else {}

    async def get_date_range(self) -> dict[str, datetime | None]:
        """
        Get the earliest and latest data dates.
        
        Returns:
            Dictionary with 'earliest' and 'latest' dates
        """
        pipeline = [
            {
                "$group": {
                    "_id": None,
                    "earliest": {"$min": "$Timestamp"},
                    "latest": {"$max": "$Timestamp"},
                }
            },
            {"$project": {"_id": 0, "earliest": 1, "latest": 1}},
        ]
        
        results = await self.aggregate(pipeline, limit=1)
        if results:
            return results[0]
        return {"earliest": None, "latest": None}

