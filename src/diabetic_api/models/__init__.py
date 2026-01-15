"""Pydantic models for API schemas."""

from .chat import ChatRequest, ChatResponse, ChatMessage, ChatStreamChunk
from .dashboard import (
    DashboardMetrics,
    DashboardData,
    GlucoseReading,
    BolusData,
    TimeRange,
    DashboardRequest,
    UploadDatesResponse,
)
from .session import ChatSession, ChatSessionDetail, SessionCreate, SessionListResponse
from .pump_data import PumpDataRecord, UploadResult, QueryResult
from .food_scan import (
    # Enums
    ArtifactType,
    ScanSource,
    UncertaintyReason,
    ScanQuality,
    ScanErrorCode,
    # Request models
    CameraIntrinsics,
    DeviceOrientation,
    DeviceInfo,
    FoodScanRequest,
    # Response models
    Macros,
    MacroRanges,
    FoodCandidate,
    VolumeEstimate,
    DebugInfo,
    FoodScanResponse,
    FoodScanError,
    # Storage models (MVP-2.2)
    FoodScan,
    ScanArtifact,
    UserOverrides,
    MealEstimate,
)

__all__ = [
    # Chat
    "ChatRequest",
    "ChatResponse",
    "ChatMessage",
    "ChatStreamChunk",
    # Dashboard
    "DashboardMetrics",
    "DashboardData",
    "GlucoseReading",
    "BolusData",
    "TimeRange",
    "DashboardRequest",
    "UploadDatesResponse",
    # Sessions
    "ChatSession",
    "ChatSessionDetail",
    "SessionCreate",
    "SessionListResponse",
    # Pump Data
    "PumpDataRecord",
    "UploadResult",
    "QueryResult",
    # Food Scan (Meal Vision)
    "ArtifactType",
    "ScanSource",
    "UncertaintyReason",
    "ScanQuality",
    "ScanErrorCode",
    "CameraIntrinsics",
    "DeviceOrientation",
    "DeviceInfo",
    "FoodScanRequest",
    "Macros",
    "MacroRanges",
    "FoodCandidate",
    "VolumeEstimate",
    "DebugInfo",
    "FoodScanResponse",
    "FoodScanError",
    "FoodScan",
    "ScanArtifact",
    "UserOverrides",
    "MealEstimate",
]

