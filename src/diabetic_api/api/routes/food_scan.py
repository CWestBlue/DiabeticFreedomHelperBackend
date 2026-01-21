"""Food Scan API routes.

Meal Vision Feature - MVP-2.1 + MVP-2.4
Endpoint for processing food images with depth data.
"""

import json
import logging
import os
from datetime import UTC, datetime
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from diabetic_api.models.food_scan import (
    FoodCandidate,
    FoodScanError,
    FoodScanRequest,
    FoodScanResponse,
    MacroRanges,
    Macros,
    ScanErrorCode,
    ScanQuality,
)
from diabetic_api.services.food_recognition import (
    FoodRecognitionError,
    get_food_recognition_service,
)

router = APIRouter()
logger = logging.getLogger(__name__)

# =============================================================================
# Validation Constants
# =============================================================================

# Minimum valid depth ratio (percentage of non-zero depth pixels)
MIN_DEPTH_VALID_RATIO = 0.40

# Maximum file sizes (bytes)
MAX_RGB_SIZE = 10 * 1024 * 1024  # 10 MB
MAX_DEPTH_SIZE = 20 * 1024 * 1024  # 20 MB (16-bit PNGs are larger)
MAX_CONFIDENCE_SIZE = 5 * 1024 * 1024  # 5 MB

# Intrinsics validation ranges
MIN_FOCAL_LENGTH = 100.0
MAX_FOCAL_LENGTH = 5000.0
MIN_IMAGE_DIMENSION = 480
MAX_IMAGE_DIMENSION = 8192


# =============================================================================
# Validation Helpers
# =============================================================================


def validate_intrinsics(intrinsics: dict) -> list[str]:
    """
    Validate camera intrinsics.
    
    Returns list of validation error messages (empty if valid).
    """
    errors = []
    
    # Check focal length ranges
    fx = intrinsics.get("fx", 0)
    fy = intrinsics.get("fy", 0)
    if not (MIN_FOCAL_LENGTH <= fx <= MAX_FOCAL_LENGTH):
        errors.append(f"fx must be between {MIN_FOCAL_LENGTH} and {MAX_FOCAL_LENGTH}")
    if not (MIN_FOCAL_LENGTH <= fy <= MAX_FOCAL_LENGTH):
        errors.append(f"fy must be between {MIN_FOCAL_LENGTH} and {MAX_FOCAL_LENGTH}")
    
    # Check principal point is within image bounds
    cx = intrinsics.get("cx", 0)
    cy = intrinsics.get("cy", 0)
    width = intrinsics.get("width", 0)
    height = intrinsics.get("height", 0)
    
    if not (0 < cx < width):
        errors.append("cx must be within image width")
    if not (0 < cy < height):
        errors.append("cy must be within image height")
    
    # Check image dimensions
    if not (MIN_IMAGE_DIMENSION <= width <= MAX_IMAGE_DIMENSION):
        errors.append(
            f"width must be between {MIN_IMAGE_DIMENSION} and {MAX_IMAGE_DIMENSION}"
        )
    if not (MIN_IMAGE_DIMENSION <= height <= MAX_IMAGE_DIMENSION):
        errors.append(
            f"height must be between {MIN_IMAGE_DIMENSION} and {MAX_IMAGE_DIMENSION}"
        )
    
    return errors


async def validate_file_size(file: UploadFile, max_size: int, name: str) -> None:
    """Validate file size doesn't exceed maximum."""
    # Read file to check size (we'll need to seek back)
    content = await file.read()
    size = len(content)
    
    # Reset file position for later processing
    await file.seek(0)
    
    if size > max_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=FoodScanError(
                error_code=ScanErrorCode.PROCESSING_ERROR,
                message=f"{name} exceeds maximum size of {max_size // (1024*1024)} MB",
                details={"size": size, "max_size": max_size},
            ).model_dump(),
        )


async def validate_depth_dimensions(
    depth_file: UploadFile, expected_width: int, expected_height: int
) -> dict:
    """
    Validate depth map dimensions match intrinsics.
    
    Returns dict with validation info.
    NOTE: Full PNG parsing would require pillow - using header check for MVP.
    """
    content = await depth_file.read()
    await depth_file.seek(0)
    
    # Check PNG signature
    if content[:8] != b"\x89PNG\r\n\x1a\n":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=FoodScanError(
                error_code=ScanErrorCode.PROCESSING_ERROR,
                message="Depth map is not a valid PNG file",
            ).model_dump(),
        )
    
    # Parse IHDR chunk for dimensions (bytes 16-23)
    if len(content) < 24:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=FoodScanError(
                error_code=ScanErrorCode.PROCESSING_ERROR,
                message="Depth map PNG is too small to be valid",
            ).model_dump(),
        )
    
    # Width and height are 4-byte big-endian integers at offsets 16 and 20
    width = int.from_bytes(content[16:20], byteorder="big")
    height = int.from_bytes(content[20:24], byteorder="big")
    bit_depth = content[24]
    
    validation_info = {
        "width": width,
        "height": height,
        "bit_depth": bit_depth,
        "size_bytes": len(content),
    }
    
    # Check dimensions match
    if width != expected_width or height != expected_height:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=FoodScanError(
                error_code=ScanErrorCode.PROCESSING_ERROR,
                message=f"Depth dimensions ({width}x{height}) don't match "
                        f"intrinsics ({expected_width}x{expected_height})",
                details=validation_info,
            ).model_dump(),
        )
    
    # Check bit depth (should be 16 for depth_u16)
    if bit_depth != 16:
        logger.warning(
            f"Depth map bit depth is {bit_depth}, expected 16. "
            "Proceeding but accuracy may be affected."
        )
    
    return validation_info


def generate_scan_id(user_id: str) -> str:
    """Generate unique scan ID with user context prefix."""
    timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    unique = uuid4().hex[:8]
    # Format: scan_{timestamp}_{unique}
    return f"scan_{timestamp}_{unique}"


# =============================================================================
# API Endpoints
# =============================================================================


@router.post(
    "/scan",
    response_model=FoodScanResponse,
    responses={
        400: {"model": FoodScanError, "description": "Invalid request or scan failed"},
        422: {"description": "Validation error"},
        500: {"model": FoodScanError, "description": "Processing error"},
    },
    summary="Scan food image for nutritional analysis",
    description="""
    Upload RGB image with depth data for food identification and nutritional estimation.
    
    **Request format:** multipart/form-data with:
    - `rgb`: JPEG image file (required, max 10 MB)
    - `depth_u16`: 16-bit PNG depth map in millimeters (required, max 20 MB)
    - `confidence_u8`: 8-bit PNG confidence map, 0-255 (optional, max 5 MB)
    - `metadata`: JSON string with FoodScanRequest fields (required)
    
    **MVP Constraints:**
    - Food must be on a plate/table
    - Table surface must be visible
    - Plate must be fully in frame
    
    **Error Codes:**
    - `NEEDS_TABLE_VISIBLE`: Table surface not detected
    - `DEPTH_TOO_SPARSE`: Insufficient depth data
    - `NO_FOOD_DETECTED`: No food found in image
    - `INVALID_INTRINSICS`: Camera intrinsics invalid
    - `PLATE_NOT_IN_FRAME`: Plate not fully visible
    """,
)
async def scan_food(
    rgb: Annotated[UploadFile, File(description="RGB image (JPEG)")],
    depth_u16: Annotated[UploadFile, File(description="16-bit depth map (PNG)")],
    metadata: Annotated[str, Form(description="JSON metadata (FoodScanRequest)")],
    confidence_u8: Annotated[
        UploadFile | None, File(description="Confidence map (PNG)")
    ] = None,
) -> FoodScanResponse:
    """
    Process food scan with RGB + depth data.
    
    MVP-2.1: Validates input and returns structured response.
    ML pipeline will be implemented in MVP-2.3 through MVP-2.8.
    """
    # -------------------------------------------------------------------------
    # Step 1: Parse and validate metadata JSON
    # -------------------------------------------------------------------------
    try:
        request_data = json.loads(metadata)
    except json.JSONDecodeError as e:
        logger.warning(f"Invalid JSON in metadata: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=FoodScanError(
                error_code=ScanErrorCode.PROCESSING_ERROR,
                message=f"Invalid JSON in metadata: {e}",
            ).model_dump(),
        )
    
    try:
        scan_request = FoodScanRequest(**request_data)
    except Exception as e:
        logger.warning(f"Metadata validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )
    
    # -------------------------------------------------------------------------
    # Step 2: Validate camera intrinsics
    # -------------------------------------------------------------------------
    intrinsics_dict = scan_request.intrinsics.model_dump()
    intrinsics_errors = validate_intrinsics(intrinsics_dict)
    
    if intrinsics_errors:
        logger.warning(f"Invalid intrinsics: {intrinsics_errors}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=FoodScanError(
                error_code=ScanErrorCode.INVALID_INTRINSICS,
                message="Camera intrinsics validation failed",
                details={"errors": intrinsics_errors},
            ).model_dump(),
        )
    
    # -------------------------------------------------------------------------
    # Step 3: Validate file types and sizes
    # -------------------------------------------------------------------------
    
    # Validate RGB image
    if rgb.content_type not in ["image/jpeg", "image/jpg"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=FoodScanError(
                error_code=ScanErrorCode.PROCESSING_ERROR,
                message="RGB image must be JPEG format",
                details={"received_type": rgb.content_type},
            ).model_dump(),
        )
    await validate_file_size(rgb, MAX_RGB_SIZE, "RGB image")
    
    # Validate depth map
    if depth_u16.content_type != "image/png":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=FoodScanError(
                error_code=ScanErrorCode.PROCESSING_ERROR,
                message="Depth map must be PNG format",
                details={"received_type": depth_u16.content_type},
            ).model_dump(),
        )
    await validate_file_size(depth_u16, MAX_DEPTH_SIZE, "Depth map")
    
    # Validate depth dimensions match intrinsics
    depth_info = await validate_depth_dimensions(
        depth_u16,
        scan_request.intrinsics.width,
        scan_request.intrinsics.height,
    )
    
    # Validate confidence map if provided
    if confidence_u8 is not None:
        if confidence_u8.content_type != "image/png":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=FoodScanError(
                    error_code=ScanErrorCode.PROCESSING_ERROR,
                    message="Confidence map must be PNG format",
                    details={"received_type": confidence_u8.content_type},
                ).model_dump(),
            )
        await validate_file_size(confidence_u8, MAX_CONFIDENCE_SIZE, "Confidence map")
    
    # -------------------------------------------------------------------------
    # Step 4: Generate scan ID
    # -------------------------------------------------------------------------
    scan_id = generate_scan_id(scan_request.user_id)
    
    # -------------------------------------------------------------------------
    # Step 5: Log request (structured logging)
    # -------------------------------------------------------------------------
    logger.info(
        "Food scan request",
        extra={
            "scan_id": scan_id,
            "user_id": scan_request.user_id,
            "device_model": scan_request.device.model,
            "device_platform": scan_request.device.platform,
            "image_dimensions": f"{scan_request.intrinsics.width}x"
                               f"{scan_request.intrinsics.height}",
            "depth_bit_depth": depth_info.get("bit_depth"),
            "has_confidence": confidence_u8 is not None,
            "opt_in_artifacts": scan_request.opt_in_store_artifacts,
        },
    )
    
    # -------------------------------------------------------------------------
    # Step 6: Food Recognition via LLaVA/Ollama (MVP-2.4)
    # -------------------------------------------------------------------------
    # Read RGB image for recognition
    rgb_data = await rgb.read()
    await rgb.seek(0)
    
    # Check if food recognition is enabled (default: True)
    use_food_recognition = os.getenv("FOOD_RECOGNITION_ENABLED", "true").lower() == "true"
    
    if use_food_recognition:
        try:
            food_service = get_food_recognition_service()
            
            # Check service health first
            is_healthy = await food_service.health_check()
            if not is_healthy:
                logger.warning("Food recognition service unhealthy, using fallback")
                raise FoodRecognitionError(
                    message="Service not available",
                    error_code="SERVICE_UNAVAILABLE",
                    provider=food_service.provider_name,
                )
            
            # Recognize foods in the image
            recognition_result = await food_service.recognize(
                image_data=rgb_data,
                max_foods=5,
                include_macros=True,
                include_portions=True,
            )
            
            logger.info(
                f"Food recognition complete",
                extra={
                    "scan_id": scan_id,
                    "provider": recognition_result.provider,
                    "foods_found": len(recognition_result.foods),
                    "confidence": recognition_result.overall_confidence,
                    "processing_time_ms": recognition_result.processing_time_ms,
                },
            )
            
            # Convert recognition results to API response format
            food_candidates = []
            for food in recognition_result.foods:
                candidate = FoodCandidate(
                    canonical_food_id=f"llava_{food.label.lower().replace(' ', '_')}",
                    label=food.label,
                    probability=food.confidence,
                    is_mixed_dish=food.is_mixed_dish,
                )
                food_candidates.append(candidate)
            
            # Get macros from primary food or total
            primary_food = recognition_result.primary_food
            total_macros = recognition_result.total_estimated_macros
            
            if total_macros:
                macros = Macros(
                    carbs=total_macros.carbs,
                    protein=total_macros.protein,
                    fat=total_macros.fat,
                    fiber=total_macros.fiber,
                )
            else:
                # Fallback macros if recognition didn't provide them
                macros = Macros(carbs=0.0, protein=0.0, fat=0.0, fiber=0.0)
            
            # Estimate grams from primary food
            grams_est = primary_food.estimated_grams if primary_food else 100.0
            
            # Build macro ranges (±20% for MVP)
            variance = 0.20
            macro_ranges = MacroRanges(
                carbs_p10=max(0, macros.carbs * (1 - variance)),
                carbs_p90=macros.carbs * (1 + variance),
                protein_p10=max(0, macros.protein * (1 - variance)),
                protein_p90=macros.protein * (1 + variance),
                fat_p10=max(0, macros.fat * (1 - variance)),
                fat_p90=macros.fat * (1 + variance),
                fiber_p10=max(0, macros.fiber * (1 - variance)),
                fiber_p90=macros.fiber * (1 + variance),
            )
            
            # Select quality based on confidence
            if recognition_result.overall_confidence >= 0.8:
                scan_quality = ScanQuality.GOOD
            elif recognition_result.overall_confidence >= 0.5:
                scan_quality = ScanQuality.OK
            else:
                scan_quality = ScanQuality.POOR
            
            # Add uncertainty reasons
            uncertainty_reasons = []
            if recognition_result.overall_confidence < 0.7:
                uncertainty_reasons.append("low_recognition_confidence")
            if not food_candidates:
                uncertainty_reasons.append("no_food_detected")
            
            selected_food = food_candidates[0] if food_candidates else None
            
            return FoodScanResponse(
                scan_id=scan_id,
                food_candidates=food_candidates,
                selected_food=selected_food,
                volume_ml=grams_est * 0.9 if grams_est else 100.0,  # Rough estimate
                grams_est=grams_est or 100.0,
                macros=macros,
                macro_ranges=macro_ranges,
                confidence_score=recognition_result.overall_confidence,
                scan_quality=scan_quality,
                uncertainty_reasons=uncertainty_reasons,
                debug={
                    "validation": "passed",
                    "depth_info": depth_info,
                    "provider": recognition_result.provider,
                    "processing_time_ms": recognition_result.processing_time_ms,
                    "raw_response": recognition_result.raw_response[:500] if scan_request.opt_in_store_artifacts else None,
                } if scan_request.opt_in_store_artifacts else None,
                processed_at=datetime.now(UTC),
            )
            
        except FoodRecognitionError as e:
            logger.warning(f"Food recognition failed: {e.message}, using fallback")
            # Fall through to fallback response below
    
    # -------------------------------------------------------------------------
    # Fallback: Return mock response when recognition unavailable
    # -------------------------------------------------------------------------
    logger.info(f"Using fallback mock response for scan {scan_id}")
    
    mock_candidate = FoodCandidate(
        canonical_food_id="fallback_001",
        label="[Recognition unavailable] Please try again",
        probability=0.5,
        is_mixed_dish=False,
    )
    
    mock_macros = Macros(carbs=30.0, protein=10.0, fat=5.0, fiber=2.0)
    
    mock_ranges = MacroRanges(
        carbs_p10=20.0,
        carbs_p90=40.0,
        protein_p10=5.0,
        protein_p90=15.0,
        fat_p10=2.0,
        fat_p90=10.0,
        fiber_p10=1.0,
        fiber_p90=4.0,
    )
    
    return FoodScanResponse(
        scan_id=scan_id,
        food_candidates=[mock_candidate],
        selected_food=mock_candidate,
        volume_ml=200.0,
        grams_est=150.0,
        macros=mock_macros,
        macro_ranges=mock_ranges,
        confidence_score=0.5,
        scan_quality=ScanQuality.OK,
        uncertainty_reasons=["recognition_service_unavailable"],
        debug={
            "validation": "passed",
            "depth_info": depth_info,
            "note": "Food recognition service unavailable, using fallback",
        } if scan_request.opt_in_store_artifacts else None,
        processed_at=datetime.now(UTC),
    )


@router.get(
    "/recognition/health",
    summary="Check food recognition service health",
    description="Check if the food recognition service (Ollama/LLaVA) is available.",
)
async def check_recognition_health() -> dict:
    """
    Check if the food recognition service is healthy and available.
    
    Returns:
        dict with status, provider, and model information
    """
    try:
        service = get_food_recognition_service()
        is_healthy = await service.health_check()
        
        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "provider": service.provider_name,
            "available": is_healthy,
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "error",
            "provider": "unknown",
            "available": False,
            "error": str(e),
        }


@router.get(
    "/scan/{scan_id}",
    response_model=FoodScanResponse,
    summary="Get scan result by ID",
    description="Retrieve a previously processed scan result.",
)
async def get_scan(scan_id: str) -> FoodScanResponse:
    """
    Retrieve scan result by ID.
    
    TODO: Implement storage lookup (MVP-2.2)
    """
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=FoodScanError(
            error_code=ScanErrorCode.PROCESSING_ERROR,
            message=f"Scan {scan_id} not found",
            details={"note": "Storage not yet implemented (MVP-2.2)"},
        ).model_dump(),
    )


# =============================================================================
# Meal Estimate Endpoints (MVP-3.3)
# =============================================================================


class SaveMealRequest(BaseModel):
    """Request to save a confirmed meal estimate."""
    scan_id: str = Field(..., description="ID of the scan this meal came from")
    user_id: str = Field(..., description="User identifier")
    source: str = Field(default="vision", description="Source of estimate")
    canonical_food_id: str = Field(..., description="Food database ID")
    food_label: str = Field(..., description="Human-readable food name")
    macros: Macros = Field(..., description="Macronutrient values")
    macro_ranges: MacroRanges | None = Field(None, description="Confidence ranges")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")
    uncertainty_reasons: list[str] = Field(
        default_factory=list, description="Reasons for uncertainty"
    )
    user_overrides: dict | None = Field(None, description="User corrections")


class MealEstimateResponse(BaseModel):
    """Response after saving a meal estimate."""
    id: str = Field(..., description="Unique meal estimate ID")
    scan_id: str = Field(..., description="Source scan ID")
    user_id: str = Field(..., description="User ID")
    food_label: str = Field(..., description="Food name")
    macros: Macros = Field(..., description="Macros")
    confidence: float = Field(..., description="Confidence score")
    created_at: datetime = Field(..., description="Creation timestamp")


@router.post(
    "/meals",
    response_model=MealEstimateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Save confirmed meal estimate",
    description="""
    Save a user-confirmed meal estimate to the meal_estimates collection.
    
    This is called after the user reviews scan results and confirms their selection.
    The data is stored separately from CareLink pump data.
    """,
)
async def save_meal_estimate(request: SaveMealRequest) -> MealEstimateResponse:
    """
    Save a confirmed meal estimate.
    
    MVP-3.3: Stores to meal_estimates collection (separate from pump_data).
    """
    # Generate meal ID
    meal_id = f"meal_{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}_{uuid4().hex[:8]}"
    created_at = datetime.now(UTC)
    
    logger.info(
        "Saving meal estimate",
        extra={
            "meal_id": meal_id,
            "scan_id": request.scan_id,
            "user_id": request.user_id,
            "food_label": request.food_label,
            "confidence": request.confidence,
        },
    )
    
    # TODO: Actually persist to MongoDB via MealEstimateRepository
    # For now, return success response for Flutter integration testing
    
    return MealEstimateResponse(
        id=meal_id,
        scan_id=request.scan_id,
        user_id=request.user_id,
        food_label=request.food_label,
        macros=request.macros,
        confidence=request.confidence,
        created_at=created_at,
    )


class MealListResponse(BaseModel):
    """Response for meal list query."""
    meals: list[MealEstimateResponse] = Field(default_factory=list)
    total: int = Field(default=0)


@router.get(
    "/meals",
    response_model=MealListResponse,
    summary="Get meal estimates for user",
    description="Retrieve meal estimates from the meal_estimates collection.",
)
async def get_meal_estimates(
    user_id: str,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    limit: int = 50,
) -> MealListResponse:
    """
    Get meal estimates for a user within a date range.
    
    MVP-3.3: Queries meal_estimates collection (separate from pump_data).
    """
    logger.info(
        "Fetching meal estimates",
        extra={
            "user_id": user_id,
            "start_date": start_date,
            "end_date": end_date,
            "limit": limit,
        },
    )
    
    # TODO: Actually query MongoDB via MealEstimateRepository
    # For now, return empty list for integration testing
    
    return MealListResponse(meals=[], total=0)
