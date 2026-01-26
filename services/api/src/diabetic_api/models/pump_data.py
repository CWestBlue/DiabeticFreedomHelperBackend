"""Pydantic models for pump data schemas."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class PumpDataRecord(BaseModel):
    """Single record from Medtronic pump export."""

    timestamp: datetime = Field(..., alias="Timestamp")
    sensor_glucose: float | None = Field(None, alias="Sensor Glucose (mg/dL)")
    bg_reading: float | None = Field(None, alias="BG Reading (mg/dL)")
    bolus_delivered: float | None = Field(None, alias="Bolus Volume Delivered (U)")
    final_bolus_estimate: float | None = Field(None, alias="Final Bolus Estimate")
    carb_input: float | None = Field(None, alias="BWZ Carb Input (grams)")
    carb_ratio: float | None = Field(None, alias="BWZ Carb Ratio (g/U)")
    insulin_sensitivity: float | None = Field(None, alias="BWZ Insulin Sensitivity (mg/dL/U)")
    basal_rate: float | None = Field(None, alias="Basal Rate (U/h)")
    alert: str | None = Field(None, alias="Alert")

    class Config:
        populate_by_name = True


class UploadResult(BaseModel):
    """Result of CSV upload processing."""

    success: bool
    records_processed: int = 0
    records_inserted: int = 0
    records_skipped: int = 0
    errors: list[str] = Field(default_factory=list)
    message: str = ""


class QueryResult(BaseModel):
    """Result of a MongoDB query (for chat responses)."""

    data: list[dict[str, Any]]
    count: int
    query_executed: str | None = None
    error: str | None = None

