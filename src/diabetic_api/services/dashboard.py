"""Dashboard service for aggregating diabetic metrics.

Replicates N8N workflow logic for dashboard data aggregation.
"""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from diabetic_api.db.unit_of_work import UnitOfWork
from diabetic_api.models.dashboard import (
    BolusData,
    DashboardData,
    DashboardMetrics,
    GlucoseReading,
    UploadDatesResponse,
)


class DashboardService:
    """
    Service for dashboard data aggregation.

    Replicates N8N dashboard workflow logic for Flutter compatibility.
    """

    def __init__(self, uow: UnitOfWork):
        """Initialize dashboard service."""
        self.uow = uow
        self.tz = ZoneInfo("America/Chicago")

    async def get_dashboard_data(
        self,
        days: int = 7,
        reference_date: datetime | None = None,
    ) -> DashboardData:
        """
        Get complete dashboard data for the specified number of days.

        Replicates N8N's dashboard aggregation logic.

        Args:
            days: Number of days of data to fetch
            reference_date: Reference date (defaults to now)

        Returns:
            Complete dashboard data matching N8N format
        """
        # Use reference_date or now
        if reference_date is None:
            reference_date = datetime.now(self.tz)
        elif reference_date.tzinfo is None:
            reference_date = reference_date.replace(tzinfo=self.tz)

        # Fetch data concurrently-ish
        metrics = await self._get_metrics(days)
        glucose_readings = await self._get_glucose_readings(days)
        bolus_data = await self._get_bolus_data(days)

        return DashboardData(
            metrics=metrics,
            glucoseReadings=glucose_readings,
            bolusData=bolus_data,
        )

    async def _get_metrics(self, days: int) -> DashboardMetrics:
        """
        Get aggregated metrics matching N8N format.

        Replicates the complex N8N aggregation that calculates:
        - Average blood sugar
        - Time in range percentages
        - Trends (comparing thirds of the time period)
        - Daily averages for carbs, bolus, basal
        """
        pipeline = [
            # Stage 1: Add fields with conversions
            {
                "$addFields": {
                    "ts": "$Timestamp",
                    "sg": {"$toDouble": {"$ifNull": ["$Sensor Glucose (mg/dL)", None]}},
                    "carb_input": {
                        "$convert": {
                            "input": "$BWZ Carb Input (grams)",
                            "to": "double",
                            "onError": 0,
                            "onNull": 0,
                        }
                    },
                    "bolus_total_doc": {
                        "$add": [
                            {
                                "$convert": {
                                    "input": "$Bolus Volume Delivered (U)",
                                    "to": "double",
                                    "onError": 0,
                                    "onNull": 0,
                                }
                            },
                            {
                                "$convert": {
                                    "input": "$Final Bolus Estimate",
                                    "to": "double",
                                    "onError": 0,
                                    "onNull": 0,
                                }
                            },
                        ]
                    },
                    "basal_rate": {
                        "$convert": {
                            "input": "$Basal Rate (U/h)",
                            "to": "double",
                            "onError": None,
                            "onNull": None,
                        }
                    },
                }
            },
            # Stage 2: Filter to date range
            {
                "$match": {
                    "$expr": {
                        "$gte": [
                            "$ts",
                            {
                                "$dateSubtract": {
                                    "startDate": "$$NOW",
                                    "unit": "day",
                                    "amount": days,
                                }
                            },
                        ]
                    }
                }
            },
            # Stage 3: Calculate thirds for trend analysis
            {
                "$addFields": {
                    "third": {
                        "$floor": {
                            "$divide": [
                                {
                                    "$subtract": [
                                        {"$toLong": "$$NOW"},
                                        {"$toLong": "$ts"},
                                    ]
                                },
                                days * 24 * 60 * 60 * 1000 / 3,
                            ]
                        }
                    }
                }
            },
            {
                "$addFields": {
                    "third": {"$min": [{"$max": [0, "$third"]}, 2]}
                }
            },
            # Stage 4: Faceted aggregation
            {
                "$facet": {
                    "glucoseStats": [
                        {"$match": {"$expr": {"$and": [{"$ne": ["$sg", None]}, {"$gt": ["$sg", 0]}]}}},
                        {
                            "$group": {
                                "_id": None,
                                "totalReadings": {"$sum": 1},
                                "inRangeCount": {
                                    "$sum": {
                                        "$cond": [
                                            {"$and": [{"$gte": ["$sg", 70]}, {"$lte": ["$sg", 180]}]},
                                            1,
                                            0,
                                        ]
                                    }
                                },
                                "lowCount": {"$sum": {"$cond": [{"$lt": ["$sg", 70]}, 1, 0]}},
                                "highCount": {"$sum": {"$cond": [{"$gt": ["$sg", 180]}, 1, 0]}},
                                "avg_sg": {"$avg": "$sg"},
                                # Stats by thirds for trends
                                "inRange_t0": {
                                    "$sum": {
                                        "$cond": [
                                            {
                                                "$and": [
                                                    {"$eq": ["$third", 0]},
                                                    {"$gte": ["$sg", 70]},
                                                    {"$lte": ["$sg", 180]},
                                                ]
                                            },
                                            1,
                                            0,
                                        ]
                                    }
                                },
                                "inRange_t2": {
                                    "$sum": {
                                        "$cond": [
                                            {
                                                "$and": [
                                                    {"$eq": ["$third", 2]},
                                                    {"$gte": ["$sg", 70]},
                                                    {"$lte": ["$sg", 180]},
                                                ]
                                            },
                                            1,
                                            0,
                                        ]
                                    }
                                },
                                "totalReadings_t0": {
                                    "$sum": {"$cond": [{"$eq": ["$third", 0]}, 1, 0]}
                                },
                                "totalReadings_t2": {
                                    "$sum": {"$cond": [{"$eq": ["$third", 2]}, 1, 0]}
                                },
                                "avg_sg_t0_sum": {
                                    "$sum": {"$cond": [{"$eq": ["$third", 0]}, "$sg", 0]}
                                },
                                "avg_sg_t2_sum": {
                                    "$sum": {"$cond": [{"$eq": ["$third", 2]}, "$sg", 0]}
                                },
                            }
                        },
                    ],
                    "dailyCarbBolus": [
                        {
                            "$group": {
                                "_id": {
                                    "$dateTrunc": {
                                        "date": "$ts",
                                        "unit": "day",
                                        "timezone": "America/Chicago",
                                    }
                                },
                                "carb_sum": {"$sum": "$carb_input"},
                                "bolus_sum": {"$sum": "$bolus_total_doc"},
                            }
                        },
                        {
                            "$group": {
                                "_id": None,
                                "avg_daily_carb": {"$avg": "$carb_sum"},
                                "avg_daily_bolus": {"$avg": "$bolus_sum"},
                            }
                        },
                    ],
                    "dailyBasal": [
                        {"$match": {"basal_rate": {"$ne": None}}},
                        {"$sort": {"ts": 1}},
                        {
                            "$group": {
                                "_id": {
                                    "$dateTrunc": {
                                        "date": "$ts",
                                        "unit": "day",
                                        "timezone": "America/Chicago",
                                    }
                                },
                                "rates": {"$push": "$basal_rate"},
                            }
                        },
                        {
                            "$addFields": {
                                # Simplified: average rate * 24 hours
                                "basal_units_sum": {"$multiply": [{"$avg": "$rates"}, 24]}
                            }
                        },
                        {
                            "$group": {
                                "_id": None,
                                "avg_daily_basal": {"$avg": "$basal_units_sum"},
                            }
                        },
                    ],
                }
            },
            # Stage 5: Flatten results
            {
                "$project": {
                    "glucose": {"$arrayElemAt": ["$glucoseStats", 0]},
                    "carbBolus": {"$arrayElemAt": ["$dailyCarbBolus", 0]},
                    "basal": {"$arrayElemAt": ["$dailyBasal", 0]},
                }
            },
        ]

        results = await self.uow.pump_data.aggregate(pipeline, limit=1)
        result = results[0] if results else {}

        glucose = result.get("glucose") or {}
        carb_bolus = result.get("carbBolus") or {}
        basal = result.get("basal") or {}

        # Calculate metrics
        avg_sg = glucose.get("avg_sg")
        total_readings = glucose.get("totalReadings", 0)
        in_range = glucose.get("inRangeCount", 0)
        low = glucose.get("lowCount", 0)
        high = glucose.get("highCount", 0)

        tir_pct = (in_range / total_readings * 100) if total_readings > 0 else None
        below_pct = (low / total_readings * 100) if total_readings > 0 else None
        above_pct = (high / total_readings * 100) if total_readings > 0 else None

        # Calculate estimated A1C: (avg_glucose + 46.7) / 28.7
        estimated_a1c = None
        if avg_sg and avg_sg > 0:
            estimated_a1c = round((avg_sg + 46.7) / 28.7, 1)

        # Calculate trends
        tir_trend = self._calculate_trend(
            glucose.get("inRange_t0", 0),
            glucose.get("totalReadings_t0", 0),
            glucose.get("inRange_t2", 0),
            glucose.get("totalReadings_t2", 0),
            threshold=1,
        )

        t0_count = glucose.get("totalReadings_t0", 0)
        t2_count = glucose.get("totalReadings_t2", 0)
        avg_t0 = glucose.get("avg_sg_t0_sum", 0) / t0_count if t0_count > 0 else None
        avg_t2 = glucose.get("avg_sg_t2_sum", 0) / t2_count if t2_count > 0 else None
        avg_trend = self._calculate_value_trend(avg_t0, avg_t2, threshold=5)

        return DashboardMetrics(
            averageBloodSugar=round(avg_sg, 1) if avg_sg else None,
            timeInRangePercentage=round(tir_pct, 1) if tir_pct is not None else None,
            timeBelowRangePercentage=round(below_pct, 1) if below_pct is not None else None,
            timeAboveRangePercentage=round(above_pct, 1) if above_pct is not None else None,
            estimatedA1c=estimated_a1c,
            timeInRangeTrend=tir_trend,
            averageBloodSugarTrend=avg_trend,
            averageDailyCarbGrams=round(carb_bolus.get("avg_daily_carb", 0), 1) or None,
            averageDailyBolusUnits=round(carb_bolus.get("avg_daily_bolus", 0), 1) or None,
            averageDailyBasalUnits=round(basal.get("avg_daily_basal", 0), 1) or None,
        )

    def _calculate_trend(
        self,
        t0_count: int,
        t0_total: int,
        t2_count: int,
        t2_total: int,
        threshold: float,
    ) -> str:
        """Calculate trend comparing first third to last third."""
        if t0_total == 0 or t2_total == 0:
            return "Stable"

        t0_pct = t0_count / t0_total * 100
        t2_pct = t2_count / t2_total * 100
        diff = t2_pct - t0_pct

        if diff > threshold:
            return "Up"
        elif diff < -threshold:
            return "Down"
        return "Stable"

    def _calculate_value_trend(
        self,
        t0_val: float | None,
        t2_val: float | None,
        threshold: float,
    ) -> str:
        """Calculate trend comparing first third to last third values."""
        if t0_val is None or t2_val is None:
            return "Stable"

        diff = t2_val - t0_val

        if diff > threshold:
            return "Up"
        elif diff < -threshold:
            return "Down"
        return "Stable"

    async def _get_glucose_readings(self, days: int) -> list[GlucoseReading]:
        """Get glucose readings in N8N format."""
        pipeline = [
            {
                "$addFields": {
                    "SG": {
                        "$convert": {
                            "input": "$Sensor Glucose (mg/dL)",
                            "to": "double",
                            "onError": None,
                            "onNull": None,
                        }
                    }
                }
            },
            {
                "$match": {
                    "$expr": {
                        "$gte": [
                            "$Timestamp",
                            {
                                "$dateSubtract": {
                                    "startDate": "$$NOW",
                                    "unit": "day",
                                    "amount": days,
                                }
                            },
                        ]
                    },
                    "SG": {"$ne": None},
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "GlucoseReading": "$SG",
                    "Timestamp": "$Timestamp",
                }
            },
            {"$sort": {"Timestamp": -1}},
        ]

        results = await self.uow.pump_data.aggregate(pipeline, limit=10000)
        return [GlucoseReading(**r) for r in results]

    async def _get_bolus_data(self, days: int) -> list[BolusData]:
        """Get bolus data in N8N format."""
        pipeline = [
            {"$addFields": {"ts": "$Timestamp"}},
            {
                "$match": {
                    "$expr": {
                        "$gte": [
                            "$ts",
                            {
                                "$dateSubtract": {
                                    "startDate": "$$NOW",
                                    "unit": "day",
                                    "amount": days,
                                }
                            },
                        ]
                    },
                    "$or": [
                        {"Final Bolus Estimate": {"$exists": True, "$nin": [None, "", 0, "0"]}},
                        {"BWZ Estimate (U)": {"$exists": True, "$nin": [None, "", 0, "0"]}},
                        {"BWZ Carb Input (grams)": {"$exists": True, "$nin": [None, "", 0, "0"]}},
                    ],
                }
            },
            {
                "$group": {
                    "_id": "$ts",
                    "FinalBolus": {
                        "$max": {
                            "$cond": [
                                {"$in": ["$Final Bolus Estimate", [None, "", 0, "0"]]},
                                None,
                                {"$toDouble": "$Final Bolus Estimate"},
                            ]
                        }
                    },
                    "ActiveInsulin": {
                        "$max": {
                            "$cond": [
                                {"$in": ["$BWZ Active Insulin (U)", [None, ""]]},
                                None,
                                {"$toDouble": "$BWZ Active Insulin (U)"},
                            ]
                        }
                    },
                    "InsulinNeededEstimate": {
                        "$max": {
                            "$cond": [
                                {"$in": ["$BWZ Estimate (U)", [None, "", 0, "0"]]},
                                None,
                                {"$toDouble": "$BWZ Estimate (U)"},
                            ]
                        }
                    },
                    "CarbInput": {
                        "$max": {
                            "$cond": [
                                {"$in": ["$BWZ Carb Input (grams)", [None, "", 0, "0"]]},
                                None,
                                {"$toDouble": "$BWZ Carb Input (grams)"},
                            ]
                        }
                    },
                    "BloodSugarCorrectionEstimate": {
                        "$max": {
                            "$cond": [
                                {"$in": ["$BWZ Correction Estimate (U)", [None, "", 0, "0"]]},
                                None,
                                {"$toDouble": "$BWZ Correction Estimate (U)"},
                            ]
                        }
                    },
                    "CarbInsulinEstimate": {
                        "$max": {
                            "$cond": [
                                {"$in": ["$BWZ Food Estimate (U)", [None, "", 0, "0"]]},
                                None,
                                {"$toDouble": "$BWZ Food Estimate (U)"},
                            ]
                        }
                    },
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "Timestamp": "$_id",
                    "FinalBolus": 1,
                    "ActiveInsulin": 1,
                    "InsulinNeededEstimate": 1,
                    "CarbInput": 1,
                    "BloodSugarCorrectionEstimate": 1,
                    "CarbInsulinEstimate": 1,
                }
            },
            {"$sort": {"Timestamp": -1}},
        ]

        results = await self.uow.pump_data.aggregate(pipeline, limit=5000)
        return [BolusData(**r) for r in results]

    async def get_upload_dates(self) -> UploadDatesResponse:
        """Get the date range of uploaded data."""
        pipeline = [
            {"$match": {"Timestamp": {"$ne": None}}},
            {
                "$group": {
                    "_id": None,
                    "Earliest": {"$min": "$Timestamp"},
                    "Latest": {"$max": "$Timestamp"},
                }
            },
            {"$project": {"_id": 0, "Earliest": 1, "Latest": 1}},
        ]

        results = await self.uow.pump_data.aggregate(pipeline, limit=1)
        if results:
            return UploadDatesResponse(
                Earliest=results[0].get("Earliest"),
                Latest=results[0].get("Latest"),
            )
        return UploadDatesResponse(Earliest=None, Latest=None)
