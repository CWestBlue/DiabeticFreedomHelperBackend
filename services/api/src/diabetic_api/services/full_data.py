"""Full dataset service for fetching comprehensive diabetic data.

Replicates N8N's 'Get Diabetic Data from Mongo (30 days)' workflow.
Provides aggregated sensor, basal, and bolus data for the Research Agent.
"""

import csv
import io
from datetime import datetime
from typing import Any

from diabetic_api.db.unit_of_work import UnitOfWork


class FullDataService:
    """
    Service for fetching comprehensive diabetic data.

    Retrieves and formats sensor glucose readings, basal rates, and bolus data
    for use by the Research Agent when the router decides on research_full or
    research_full_query workflow paths.
    """

    def __init__(self, uow: UnitOfWork, days: int = 90):
        """
        Initialize full data service.

        Args:
            uow: Unit of Work for database access
            days: Number of days of data to fetch (default: 90)
        """
        self.uow = uow
        self.days = days

    async def get_full_dataset(self) -> dict[str, str]:
        """
        Fetch the complete dataset for research analysis.

        Returns data in CSV format for compatibility with N8N Research Agent.

        Returns:
            Dictionary with:
            - sensorData: CSV of sensor glucose readings
            - basalData: CSV of basal rate changes
            - bolusData: CSV of bolus/carb entries
        """
        # Fetch all three data types
        sensor_data = await self._get_sensor_readings()
        basal_data = await self._get_basal_rates()
        bolus_data = await self._get_bolus_data()

        return {
            "sensorData": self._to_csv(sensor_data, alias={"ts": "Start", "g": "AvgSensorGlucose"}),
            "basalData": self._to_csv(basal_data, alias={"ts": "Timestamp", "r": "BasalRate"}),
            "bolusData": self._to_csv(bolus_data, alias={"ts": "Timestamp"}),
        }

    async def _get_sensor_readings(self) -> list[dict[str, Any]]:
        """
        Get sensor glucose readings aggregated by 3-hour windows.

        Matches N8N's GetSensorReadings aggregation.
        """
        pipeline = [
            {
                "$addFields": {
                    "ts": "$Timestamp",
                    "SG": {
                        "$convert": {
                            "input": "$Sensor Glucose (mg/dL)",
                            "to": "double",
                            "onError": None,
                            "onNull": None,
                        }
                    },
                }
            },
            {
                "$match": {
                    "$expr": {
                        "$gte": [
                            "$ts",
                            {
                                "$subtract": [
                                    "$$NOW",
                                    {"$multiply": [self.days, 24, 60, 60, 1000]},
                                ]
                            },
                        ]
                    },
                    "SG": {"$ne": None},
                }
            },
            {
                "$group": {
                    "_id": {
                        "$dateTrunc": {
                            "date": "$ts",
                            "unit": "hour",
                            "binSize": 3,
                            "timezone": "America/Chicago",
                        }
                    },
                    "AvgSensorGlucose": {"$avg": "$SG"},
                    "Count": {"$sum": 1},
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "Start": "$_id",
                    "AvgSensorGlucose": {"$round": ["$AvgSensorGlucose", 1]},
                    "Count": 1,
                }
            },
            {"$sort": {"Start": -1}},
        ]

        return await self.uow.run_aggregation("PumpData", pipeline, limit=5000)

    async def _get_basal_rates(self) -> list[dict[str, Any]]:
        """
        Get basal rate changes (only rows where rate differs from previous).

        Matches N8N's GetBasalRate aggregation with $setWindowFields.
        """
        pipeline = [
            {"$addFields": {"ts": "$Timestamp"}},
            {
                "$match": {
                    "$expr": {
                        "$gte": [
                            "$ts",
                            {
                                "$subtract": [
                                    "$$NOW",
                                    {"$multiply": [self.days, 24, 60, 60, 1000]},
                                ]
                            },
                        ]
                    },
                    "Basal Rate (U/h)": {"$exists": True, "$nin": [None, "", 0, "0"]},
                }
            },
            {"$sort": {"ts": 1}},
            {
                "$setWindowFields": {
                    "sortBy": {"ts": 1},
                    "output": {
                        "PrevRate": {
                            "$shift": {
                                "output": "$Basal Rate (U/h)",
                                "by": -1,
                            }
                        }
                    },
                }
            },
            {
                "$match": {
                    "$expr": {
                        "$or": [
                            {"$eq": ["$PrevRate", None]},
                            {"$ne": ["$Basal Rate (U/h)", "$PrevRate"]},
                        ]
                    }
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "Timestamp": "$ts",
                    "BasalRate": "$Basal Rate (U/h)",
                }
            },
            {"$sort": {"Timestamp": -1}},
        ]

        return await self.uow.run_aggregation("PumpData", pipeline, limit=5000)

    async def _get_bolus_data(self) -> list[dict[str, Any]]:
        """
        Get bolus and carb input data.

        Matches N8N's GetBolusData aggregation.
        """
        pipeline = [
            {"$addFields": {"ts": "$Timestamp"}},
            {
                "$match": {
                    "$expr": {
                        "$gte": [
                            "$ts",
                            {
                                "$subtract": [
                                    "$$NOW",
                                    {"$multiply": [self.days, 24, 60, 60, 1000]},
                                ]
                            },
                        ]
                    },
                    "$or": [
                        {"Final Bolus Estimate": {"$exists": True, "$nin": [None, "", 0, "0"]}},
                        {"BWZ Estimate (U)": {"$exists": True, "$nin": [None, "", 0, "0"]}},
                        {"BWZ Carb Input (grams)": {"$exists": True, "$nin": [None, "", 0, "0"]}},
                        {"BWZ Correction Estimate (U)": {"$exists": True, "$nin": [None, "", 0, "0"]}},
                        {"BWZ Food Estimate (U)": {"$exists": True, "$nin": [None, "", 0, "0"]}},
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
                                {"$toDouble": {"$ifNull": ["$Final Bolus Estimate", None]}},
                            ]
                        }
                    },
                    "ActiveInsulin": {
                        "$max": {
                            "$cond": [
                                {"$in": ["$BWZ Active Insulin (U)", [None, ""]]},
                                None,
                                {"$toDouble": {"$ifNull": ["$BWZ Active Insulin (U)", None]}},
                            ]
                        }
                    },
                    "InsulinNeededEstimate": {
                        "$max": {
                            "$cond": [
                                {"$in": ["$BWZ Estimate (U)", [None, "", 0, "0"]]},
                                None,
                                {"$toDouble": {"$ifNull": ["$BWZ Estimate (U)", None]}},
                            ]
                        }
                    },
                    "CarbInput": {
                        "$max": {
                            "$cond": [
                                {"$in": ["$BWZ Carb Input (grams)", [None, "", 0, "0"]]},
                                None,
                                {"$toDouble": {"$ifNull": ["$BWZ Carb Input (grams)", None]}},
                            ]
                        }
                    },
                    "BloodSugarCorrectionEstimate": {
                        "$max": {
                            "$cond": [
                                {"$in": ["$BWZ Correction Estimate (U)", [None, "", 0, "0"]]},
                                None,
                                {"$toDouble": {"$ifNull": ["$BWZ Correction Estimate (U)", None]}},
                            ]
                        }
                    },
                    "CarbInsulinEstimate": {
                        "$max": {
                            "$cond": [
                                {"$in": ["$BWZ Food Estimate (U)", [None, "", 0, "0"]]},
                                None,
                                {"$toDouble": {"$ifNull": ["$BWZ Food Estimate (U)", None]}},
                            ]
                        }
                    },
                }
            },
            {
                "$match": {
                    "$or": [
                        {"InsulinNeededEstimate": {"$ne": None}},
                        {"CarbInput": {"$ne": None}},
                        {"BloodSugarCorrectionEstimate": {"$ne": None}},
                        {"CarbInsulinEstimate": {"$ne": None}},
                        {"FinalBolus": {"$ne": None}},
                    ]
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

        return await self.uow.run_aggregation("PumpData", pipeline, limit=5000)

    def _to_csv(self, items: list[dict], alias: dict[str, str] | None = None) -> str:
        """
        Convert list of dictionaries to CSV string.

        Args:
            items: List of data dictionaries
            alias: Optional field name mapping (short -> long)

        Returns:
            CSV-formatted string
        """
        if not items:
            return ""

        alias = alias or {}

        # Prune null/empty values and apply aliases
        def prune_and_alias(row: dict) -> dict:
            result = {}
            for key, value in row.items():
                # Skip null, empty, and zero values
                if value is None or value == "" or value == 0:
                    continue
                # Apply alias if exists
                aliased_key = alias.get(key, key)
                # Format datetime objects
                if isinstance(value, datetime):
                    result[aliased_key] = value.isoformat()
                else:
                    result[aliased_key] = value
            return result

        rows = [prune_and_alias(item) for item in items]

        if not rows:
            return ""

        # Get all unique headers
        headers = list(dict.fromkeys(key for row in rows for key in row.keys()))

        # Write CSV
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

        return output.getvalue()

