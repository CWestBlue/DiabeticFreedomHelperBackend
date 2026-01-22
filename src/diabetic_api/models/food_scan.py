"""Pydantic models for Food Scan API contract.

Meal Vision Feature - MVP-0.1
Defines the request/response contract for the /food/scan endpoint.

This contract is versioned and designed to support future enhancements:
- Multi-frame fusion (V2)
- Barcode scanning (V4)
- iOS support (V5)
"""

from datetime import datetime
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field

# =============================================================================
# Enums
# =============================================================================


class ScanSource(str, Enum):
    """Source of the food scan."""

    VISION = "vision"  # Camera-based food recognition
    BARCODE = "barcode"  # Future: barcode scanning
    MANUAL = "manual"  # User-entered manually


class UncertaintyReason(str, Enum):
    """Reasons for uncertainty in scan results."""

    MIXED_DISH = "mixed_dish"  # Multiple foods detected
    LOW_SEGMENTATION_CONFIDENCE = "low_segmentation_confidence"
    POOR_DEPTH_QUALITY = "poor_depth_quality"
    INSUFFICIENT_TABLE_VISIBLE = "insufficient_table_visible"
    WEAK_FOOD_MAPPING = "weak_food_mapping"
    UNKNOWN_FOOD = "unknown_food"
    PARTIAL_OCCLUSION = "partial_occlusion"
    LOW_LIGHTING = "low_lighting"
    MOTION_BLUR = "motion_blur"
    NO_DEPTH_DATA = "no_depth_data"  # Depth sensor not available
    LOW_RECOGNITION_CONFIDENCE = "low_recognition_confidence"  # AI recognition uncertain
    RECOGNITION_SERVICE_UNAVAILABLE = "recognition_service_unavailable"  # Ollama/LLaVA unavailable


class MacroSource(str, Enum):
    """Source of macro nutrient data."""
    
    USDA = "usda"  # USDA FoodData Central
    LLAVA = "llava"  # AI/LLaVA estimate
    USER_OVERRIDE = "user_override"  # User-provided values
    UNKNOWN = "unknown"


class ScanQuality(str, Enum):
    """Overall scan quality indicator."""

    GOOD = "good"
    OK = "ok"
    POOR = "poor"


class ScanErrorCode(str, Enum):
    """Standardized error codes for scan failures."""

    NEEDS_TABLE_VISIBLE = "NEEDS_TABLE_VISIBLE"
    DEPTH_TOO_SPARSE = "DEPTH_TOO_SPARSE"
    NO_FOOD_DETECTED = "NO_FOOD_DETECTED"
    INVALID_INTRINSICS = "INVALID_INTRINSICS"
    IMAGE_TOO_DARK = "IMAGE_TOO_DARK"
    IMAGE_TOO_BLURRY = "IMAGE_TOO_BLURRY"
    PLATE_NOT_IN_FRAME = "PLATE_NOT_IN_FRAME"
    PROCESSING_ERROR = "PROCESSING_ERROR"


# =============================================================================
# Request Models
# =============================================================================


class CameraIntrinsics(BaseModel):
    """Camera intrinsic parameters for depth projection."""

    fx: float = Field(..., description="Focal length X in pixels")
    fy: float = Field(..., description="Focal length Y in pixels")
    cx: float = Field(..., description="Principal point X in pixels")
    cy: float = Field(..., description="Principal point Y in pixels")
    width: int = Field(..., description="Image width in pixels")
    height: int = Field(..., description="Image height in pixels")


class DeviceOrientation(BaseModel):
    """Device orientation at capture time."""

    pitch: float = Field(..., description="Pitch angle in degrees")
    roll: float = Field(..., description="Roll angle in degrees")
    yaw: float = Field(..., description="Yaw angle in degrees")


class DeviceInfo(BaseModel):
    """Device information for debugging and model selection."""

    platform: str = Field(..., description="Platform: 'android' or 'ios'")
    model: str = Field(..., description="Device model (e.g., 'Pixel 9')")
    os_version: str = Field(..., description="OS version string")
    depth_sensor: str | None = Field(
        None, description="Depth sensor type (e.g., 'arcore_raw', 'arcore_smoothed')"
    )


class FoodScanRequest(BaseModel):
    """
    Request payload for /food/scan endpoint.
    
    Images are uploaded as multipart form data alongside this JSON metadata.
    - rgb: JPEG image file
    - depth_u16: 16-bit PNG depth map (millimeters)
    - confidence_u8: Optional 8-bit PNG confidence map (0-255)
    """

    # Image metadata (actual images sent as multipart files)
    intrinsics: CameraIntrinsics = Field(..., description="Camera intrinsic parameters")
    orientation: DeviceOrientation = Field(..., description="Device orientation at capture")
    device: DeviceInfo = Field(..., description="Device information")

    # Versioning
    scan_version: str = Field(
        default="1.0",
        description="API contract version for backward compatibility",
    )

    # User context
    user_id: str = Field(..., description="User identifier for personalization")
    client_timestamp: datetime = Field(
        ..., description="Client-side timestamp of capture (ISO 8601)"
    )

    # Privacy
    opt_in_store_artifacts: bool = Field(
        default=False,
        description="User consent to store RGB/depth images for debugging",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "intrinsics": {
                    "fx": 1000.0,
                    "fy": 1000.0,
                    "cx": 540.0,
                    "cy": 960.0,
                    "width": 1080,
                    "height": 1920,
                },
                "orientation": {"pitch": -45.0, "roll": 0.0, "yaw": 0.0},
                "device": {
                    "platform": "android",
                    "model": "Pixel 9",
                    "os_version": "15",
                    "depth_sensor": "arcore_raw",
                },
                "scan_version": "1.0",
                "user_id": "user_123",
                "client_timestamp": "2026-01-15T12:30:00Z",
                "opt_in_store_artifacts": False,
            }
        }


# =============================================================================
# Response Models
# =============================================================================


class Macros(BaseModel):
    """Macronutrient values in grams."""

    carbs: float = Field(..., ge=0, description="Carbohydrates in grams")
    protein: float = Field(..., ge=0, description="Protein in grams")
    fat: float = Field(..., ge=0, description="Fat in grams")
    fiber: float = Field(..., ge=0, description="Fiber in grams")


class MacroRanges(BaseModel):
    """P10-P90 confidence ranges for macros."""

    carbs_p10: float = Field(..., ge=0, description="Carbs 10th percentile")
    carbs_p90: float = Field(..., ge=0, description="Carbs 90th percentile")
    protein_p10: float = Field(..., ge=0, description="Protein 10th percentile")
    protein_p90: float = Field(..., ge=0, description="Protein 90th percentile")
    fat_p10: float = Field(..., ge=0, description="Fat 10th percentile")
    fat_p90: float = Field(..., ge=0, description="Fat 90th percentile")
    fiber_p10: float = Field(..., ge=0, description="Fiber 10th percentile")
    fiber_p90: float = Field(..., ge=0, description="Fiber 90th percentile")


class FoodCandidate(BaseModel):
    """A candidate food identification result."""

    canonical_food_id: str = Field(..., description="Canonical food database ID")
    label: str = Field(..., description="Human-readable food name")
    probability: Annotated[float, Field(ge=0, le=1)] = Field(
        ..., description="Confidence probability (0-1)"
    )
    is_mixed_dish: bool = Field(
        default=False, description="Whether this appears to be a mixed dish"
    )
    estimated_grams: float | None = Field(
        default=None, description="Estimated portion size in grams"
    )
    macros: "Macros | None" = Field(
        default=None, description="Estimated macros for this candidate"
    )


class VolumeEstimate(BaseModel):
    """Volume estimation result."""

    volume_ml: float = Field(..., ge=0, description="Estimated volume in milliliters")
    quality_score: Annotated[float, Field(ge=0, le=1)] = Field(
        ..., description="Volume estimation quality (0-1)"
    )
    method: str = Field(
        ..., description="Method used: 'depth_integration', 'reference_object', etc."
    )


class DebugInfo(BaseModel):
    """Debug information for troubleshooting (only included if opt-in)."""

    segmentation_time_ms: int | None = Field(None, description="Segmentation processing time")
    identification_time_ms: int | None = Field(None, description="Food ID processing time")
    volume_time_ms: int | None = Field(None, description="Volume computation time")
    total_time_ms: int | None = Field(None, description="Total processing time")
    depth_valid_ratio: float | None = Field(None, description="Ratio of valid depth pixels")
    plane_fit_inliers: int | None = Field(None, description="Number of plane fit inliers")
    plane_fit_rmse: float | None = Field(None, description="Plane fit RMSE")


class FoodScanResponse(BaseModel):
    """
    Response payload for successful /food/scan request.
    
    Contains food identification, volume estimation, and nutritional analysis.
    """

    # Identification
    scan_id: str = Field(..., description="Unique identifier for this scan")

    # Food candidates (top-K results)
    food_candidates: list[FoodCandidate] = Field(
        ..., min_length=1, max_length=10, description="Ranked food candidates"
    )
    selected_food: FoodCandidate = Field(
        ..., description="Top/selected food candidate"
    )

    # Volume & weight estimation
    volume_ml: float = Field(..., ge=0, description="Estimated volume in milliliters")
    grams_est: float = Field(..., ge=0, description="Estimated weight in grams")

    # Nutritional values
    macros: Macros = Field(..., description="Primary macronutrients (from best source)")
    macro_ranges: MacroRanges = Field(..., description="P10-P90 confidence ranges")
    
    # Macro source tracking (MVP-2.7)
    macro_source: MacroSource = Field(
        MacroSource.LLAVA, description="Source of primary macros: usda, llava, or user_override"
    )
    llava_macros: Macros | None = Field(
        None, description="AI-estimated macros from LLaVA (always included when available)"
    )
    usda_macros: Macros | None = Field(
        None, description="USDA database macros (null if no match found)"
    )
    usda_food_id: str | None = Field(
        None, description="USDA FoodData Central ID for the matched food"
    )
    usda_food_name: str | None = Field(
        None, description="Official USDA food name"
    )

    # Confidence & quality
    confidence_score: Annotated[float, Field(ge=0, le=1)] = Field(
        ..., description="Overall confidence (0-1)"
    )
    scan_quality: ScanQuality = Field(..., description="Scan quality indicator")
    uncertainty_reasons: list[UncertaintyReason] = Field(
        default_factory=list, description="Reasons for uncertainty"
    )

    # Debug (optional)
    debug: DebugInfo | None = Field(
        None, description="Debug info (only if opt_in_store_artifacts=true)"
    )

    # Timestamps
    processed_at: datetime = Field(..., description="Server processing timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "scan_id": "scan_abc123",
                "food_candidates": [
                    {
                        "canonical_food_id": "usda_12345",
                        "label": "Grilled Chicken Breast",
                        "probability": 0.85,
                        "is_mixed_dish": False,
                    },
                    {
                        "canonical_food_id": "usda_12346",
                        "label": "Roasted Turkey Breast",
                        "probability": 0.10,
                        "is_mixed_dish": False,
                    },
                ],
                "selected_food": {
                    "canonical_food_id": "usda_12345",
                    "label": "Grilled Chicken Breast",
                    "probability": 0.85,
                    "is_mixed_dish": False,
                },
                "volume_ml": 180.5,
                "grams_est": 150.0,
                "macros": {"carbs": 0.0, "protein": 31.0, "fat": 3.6, "fiber": 0.0},
                "macro_ranges": {
                    "carbs_p10": 0.0,
                    "carbs_p90": 2.0,
                    "protein_p10": 25.0,
                    "protein_p90": 38.0,
                    "fat_p10": 2.0,
                    "fat_p90": 6.0,
                    "fiber_p10": 0.0,
                    "fiber_p90": 0.5,
                },
                "confidence_score": 0.82,
                "scan_quality": "good",
                "uncertainty_reasons": [],
                "debug": None,
                "processed_at": "2026-01-15T12:30:05Z",
            }
        }


class FoodScanError(BaseModel):
    """Error response for failed /food/scan request."""

    error_code: ScanErrorCode = Field(..., description="Standardized error code")
    message: str = Field(..., description="Human-readable error message")
    details: dict | None = Field(None, description="Additional error details")
    scan_id: str | None = Field(None, description="Scan ID if partially processed")

    class Config:
        json_schema_extra = {
            "example": {
                "error_code": "NEEDS_TABLE_VISIBLE",
                "message": "Table surface must be visible for accurate volume estimation",
                "details": {"depth_valid_ratio": 0.15, "min_required": 0.40},
                "scan_id": None,
            }
        }


# =============================================================================
# FoodScan (for food_scans collection - MVP-2.2)
# =============================================================================


class ArtifactType(str, Enum):
    """Types of scan artifacts."""

    RGB = "rgb"  # RGB camera image
    DEPTH_U16 = "depth_u16"  # 16-bit depth map
    CONFIDENCE_U8 = "confidence_u8"  # 8-bit confidence map
    SEGMENTATION_MASK = "segmentation_mask"  # Food segmentation mask


class FoodScan(BaseModel):
    """
    Scan metadata and results record.
    
    Stored in `food_scans` collection (separate from `pump_data`).
    This stores the scan request metadata and processing results.
    """

    # Identity
    id: str | None = Field(None, description="MongoDB document ID")
    scan_id: str = Field(..., description="Unique scan identifier")
    user_id: str = Field(..., description="User identifier")

    # Device info
    device: DeviceInfo = Field(..., description="Device information")
    intrinsics: CameraIntrinsics = Field(..., description="Camera intrinsics")
    orientation: DeviceOrientation | None = Field(None, description="Device orientation")

    # Scan results
    food_candidates: list["FoodCandidate"] = Field(
        default_factory=list, description="Food identification candidates"
    )
    selected_food: "FoodCandidate | None" = Field(None, description="Selected food")
    volume_ml: float | None = Field(None, description="Estimated volume in ml")
    grams_est: float | None = Field(None, description="Estimated weight in grams")
    macros: "Macros | None" = Field(None, description="Estimated macros")
    macro_ranges: "MacroRanges | None" = Field(None, description="Macro confidence ranges")
    confidence_score: float | None = Field(None, ge=0, le=1, description="Overall confidence")
    scan_quality: ScanQuality | None = Field(None, description="Scan quality indicator")
    uncertainty_reasons: list[UncertaintyReason] = Field(
        default_factory=list, description="Uncertainty reasons"
    )

    # Processing metadata
    processing_time_ms: int | None = Field(None, description="Server processing time")
    scan_version: str = Field(default="1.0", description="API contract version")
    opt_in_store_artifacts: bool = Field(
        default=False, description="User opted in to store artifacts"
    )

    # Timestamps
    created_at: datetime = Field(..., description="Scan timestamp")
    processed_at: datetime | None = Field(None, description="Processing completion time")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "fs_abc123",
                "scan_id": "scan_20260115123005_a1b2c3d4",
                "user_id": "user_123",
                "device": {
                    "platform": "android",
                    "model": "Pixel 9",
                    "os_version": "15",
                    "depth_sensor": "arcore_raw",
                },
                "intrinsics": {
                    "fx": 1000.0,
                    "fy": 1000.0,
                    "cx": 540.0,
                    "cy": 960.0,
                    "width": 1080,
                    "height": 1920,
                },
                "confidence_score": 0.82,
                "scan_quality": "good",
                "processing_time_ms": 1250,
                "opt_in_store_artifacts": False,
                "created_at": "2026-01-15T12:30:00Z",
                "processed_at": "2026-01-15T12:30:05Z",
            }
        }


class ScanArtifact(BaseModel):
    """
    Scan artifact (image/depth data) for debugging and analytics.
    
    Stored in `scan_artifacts` collection with TTL expiration.
    Only created when user opts in via `opt_in_store_artifacts=true`.
    """

    # Identity
    id: str | None = Field(None, description="MongoDB document ID")
    scan_id: str = Field(..., description="Reference to parent scan")
    artifact_type: ArtifactType = Field(..., description="Type of artifact")

    # Storage
    storage_uri: str = Field(..., description="GridFS or blob storage URI")
    size_bytes: int = Field(..., description="Artifact size in bytes")
    content_type: str = Field(..., description="MIME type (image/jpeg, image/png)")

    # Metadata
    width: int | None = Field(None, description="Image width")
    height: int | None = Field(None, description="Image height")
    bit_depth: int | None = Field(None, description="Bit depth (for depth maps)")

    # TTL
    created_at: datetime = Field(..., description="Upload timestamp")
    ttl_expires_at: datetime = Field(
        ..., description="TTL expiration (default 7 days from creation)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "id": "art_xyz789",
                "scan_id": "scan_20260115123005_a1b2c3d4",
                "artifact_type": "rgb",
                "storage_uri": "gridfs://scan_artifacts/art_xyz789",
                "size_bytes": 1234567,
                "content_type": "image/jpeg",
                "width": 1080,
                "height": 1920,
                "bit_depth": None,
                "created_at": "2026-01-15T12:30:00Z",
                "ttl_expires_at": "2026-01-22T12:30:00Z",
            }
        }


# =============================================================================
# MealEstimate (for storage/timeline integration)
# =============================================================================


class UserOverrides(BaseModel):
    """User corrections to scan results."""

    food_id: str | None = Field(None, description="User-selected food ID override")
    food_label: str | None = Field(None, description="User-entered food name")
    grams: float | None = Field(None, description="User-corrected weight")
    carbs: float | None = Field(None, description="User-corrected carbs")
    protein: float | None = Field(None, description="User-corrected protein")
    fat: float | None = Field(None, description="User-corrected fat")
    fiber: float | None = Field(None, description="User-corrected fiber")


class MealEstimate(BaseModel):
    """
    Canonical meal estimate object for storage.
    
    Stored in `meal_estimates` collection (separate from `pump_data`).
    Used for timeline display alongside CareLink data.
    """

    # Identity
    id: str | None = Field(None, description="MongoDB document ID")
    scan_id: str = Field(..., description="Reference to original scan")
    user_id: str = Field(..., description="User identifier")

    # Source
    source: ScanSource = Field(default=ScanSource.VISION, description="How meal was logged")

    # Food identification
    canonical_food_id: str = Field(..., description="Canonical food database ID")
    food_label: str = Field(..., description="Human-readable food name")

    # Nutritional values (final, after any user overrides)
    macros: Macros = Field(..., description="Final macronutrient values")
    macro_ranges: MacroRanges | None = Field(None, description="Confidence ranges (if from scan)")

    # Confidence
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")
    uncertainty_reasons: list[UncertaintyReason] = Field(
        default_factory=list, description="Uncertainty reasons"
    )

    # User corrections
    user_overrides: UserOverrides | None = Field(
        None, description="User corrections (if any)"
    )

    # Timestamps
    created_at: datetime = Field(..., description="When meal was logged")
    updated_at: datetime | None = Field(None, description="Last update timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "meal_xyz789",
                "scan_id": "scan_abc123",
                "user_id": "user_123",
                "source": "vision",
                "canonical_food_id": "usda_12345",
                "food_label": "Grilled Chicken Breast",
                "macros": {"carbs": 0.0, "protein": 31.0, "fat": 3.6, "fiber": 0.0},
                "macro_ranges": {
                    "carbs_p10": 0.0,
                    "carbs_p90": 2.0,
                    "protein_p10": 25.0,
                    "protein_p90": 38.0,
                    "fat_p10": 2.0,
                    "fat_p90": 6.0,
                    "fiber_p10": 0.0,
                    "fiber_p90": 0.5,
                },
                "confidence": 0.82,
                "uncertainty_reasons": [],
                "user_overrides": None,
                "created_at": "2026-01-15T12:30:05Z",
                "updated_at": None,
            }
        }
