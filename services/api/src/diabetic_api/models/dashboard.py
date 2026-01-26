"""Pydantic models for dashboard-related API schemas.

These models match the N8N webhook response format expected by the Flutter app.
"""

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class TimeRange(str, Enum):
    """Time range options for dashboard."""

    WEEK = "week"
    MONTH = "month"
    THREE_MONTHS = "3months"


# ============================================================================
# Request Models (matching N8N webhook format)
# ============================================================================


class DashboardRequest(BaseModel):
    """Request model for dashboard endpoint (N8N format)."""

    unit: str = Field("day", description="Time unit: day, week, month")
    unitAmount: int = Field(7, description="Number of units")
    days: int = Field(7, description="Number of days to fetch")
    requestedAt: datetime | None = Field(None, description="Reference date")


# ============================================================================
# Response Models (matching N8N webhook format for Flutter)
# ============================================================================


class DashboardMetrics(BaseModel):
    """Aggregated metrics for dashboard display (N8N format)."""

    # Primary metrics
    averageBloodSugar: float | None = Field(None, description="Average glucose mg/dL")
    timeInRangePercentage: float | None = Field(None, description="% in 70-180 range")
    timeBelowRangePercentage: float | None = Field(None, description="% below 70")
    timeAboveRangePercentage: float | None = Field(None, description="% above 180")
    estimatedA1c: float | None = Field(None, description="Estimated A1C")

    # Trends (string values: "Up", "Down", "Stable")
    timeInRangeTrend: Literal["Up", "Down", "Stable"] | None = "Stable"
    averageBloodSugarTrend: Literal["Up", "Down", "Stable"] | None = "Stable"

    # Daily averages
    averageDailyCarbGrams: float | None = None
    averageDailyBolusUnits: float | None = None
    averageDailyBasalUnits: float | None = None


class GlucoseReading(BaseModel):
    """Single glucose reading (N8N format)."""

    GlucoseReading: float = Field(..., description="Glucose value in mg/dL")
    Timestamp: datetime = Field(..., description="Reading timestamp")


class BolusData(BaseModel):
    """Single bolus entry (N8N format)."""

    Timestamp: datetime
    FinalBolus: float | None = None
    ActiveInsulin: float | None = None
    InsulinNeededEstimate: float | None = None
    CarbInput: float | None = None
    BloodSugarCorrectionEstimate: float | None = None
    CarbInsulinEstimate: float | None = None


class DashboardData(BaseModel):
    """Complete dashboard data response (N8N format)."""

    metrics: DashboardMetrics
    glucoseReadings: list[GlucoseReading] = Field(default_factory=list)
    bolusData: list[BolusData] = Field(default_factory=list)


# ============================================================================
# Upload Dates Response
# ============================================================================


class UploadDatesResponse(BaseModel):
    """Response for data date range (N8N format)."""

    Earliest: datetime | None = None
    Latest: datetime | None = None
