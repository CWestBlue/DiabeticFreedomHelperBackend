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

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import Response
from pydantic import BaseModel, Field

from diabetic_api.api.dependencies import UoWDep, get_uow
from diabetic_api.db.unit_of_work import UnitOfWork
from diabetic_api.models.food_scan import (
    DetectedFood,
    FoodCandidate,
    FoodScanError,
    FoodScanRequest,
    FoodScanResponse,
    MacroRanges,
    MacroSource,
    Macros,
    ScanErrorCode,
    ScanQuality,
    SegmentationResult,
    UncertaintyReason,
)
from diabetic_api.core.config import get_settings
from diabetic_api.services.food_recognition import (
    FoodRecognitionError,
    get_food_recognition_service,
    create_per_mask_service,
    RecognizedFoodWithMask,
)
from diabetic_api.services.nutrition_lookup import (
    get_nutrition_lookup_service,
    NutritionLookupError,
)
from diabetic_api.services.confidence_engine import get_confidence_engine
from diabetic_api.services.segmentation import get_segmentation_client
from diabetic_api.services.volume import (
    VolumeComputationService,
    VolumeError,
    get_volume_service,
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
    
    Accepts both PNG format and raw 16-bit depth bytes.
    For raw bytes: expects width * height * 2 bytes (16-bit per pixel).
    """
    content = await depth_file.read()
    await depth_file.seek(0)
    
    # Check if it's PNG format
    is_png = content[:8] == b"\x89PNG\r\n\x1a\n"
    
    if is_png:
        # Parse PNG header for dimensions
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
            "format": "png",
        }
    else:
        # Assume raw 16-bit depth bytes
        # Expected size: width * height * 2 bytes (16-bit per pixel)
        expected_size = expected_width * expected_height * 2
        actual_size = len(content)
        
        # Allow some tolerance for row padding (up to 10% larger)
        if actual_size < expected_size or actual_size > expected_size * 1.1:
            logger.warning(
                f"Depth size mismatch: expected ~{expected_size} bytes for "
                f"{expected_width}x{expected_height}, got {actual_size}"
            )
            # Don't fail - just log warning and use what we have
        
        validation_info = {
            "width": expected_width,
            "height": expected_height,
            "bit_depth": 16,
            "size_bytes": actual_size,
            "format": "raw_u16",
        }
        
        logger.info(
            f"Received raw depth data: {actual_size} bytes "
            f"(expected {expected_size} for {expected_width}x{expected_height})"
        )
        
        # For raw format, dimensions come from intrinsics, so always match
        return validation_info
    
    # Check dimensions match (only for PNG where we parse them)
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
# Helper Functions
# =============================================================================


async def _store_scan_artifacts(
    uow: UnitOfWork,
    scan_id: str,
    rgb_data: bytes,
    depth_file: UploadFile | None,
    depth_info: dict | None,
    confidence_file: UploadFile | None,
    intrinsics,
) -> None:
    """
    Store scan artifacts to GridFS when user opts in.
    
    Args:
        uow: Unit of Work for database access
        scan_id: The scan identifier
        rgb_data: RGB image bytes (already read)
        depth_file: Depth map upload file (optional)
        depth_info: Depth validation info (optional)
        confidence_file: Confidence map upload file (optional)
        intrinsics: Camera intrinsics for dimensions
    """
    try:
        # Store RGB image
        await uow.scan_artifacts.store_artifact_with_data(
            gridfs=uow.gridfs,
            scan_id=scan_id,
            artifact_type="rgb",
            data=rgb_data,
            content_type="image/jpeg",
            width=intrinsics.width,
            height=intrinsics.height,
        )
        logger.info(f"Stored RGB artifact for scan {scan_id}: {len(rgb_data)} bytes")
        
        # Store depth map if provided
        if depth_file is not None and depth_info is not None:
            depth_data = await depth_file.read()
            await depth_file.seek(0)  # Reset for any further processing
            
            content_type = (
                "image/png" if depth_info.get("format") == "png" 
                else "application/octet-stream"
            )
            
            await uow.scan_artifacts.store_artifact_with_data(
                gridfs=uow.gridfs,
                scan_id=scan_id,
                artifact_type="depth_u16",
                data=depth_data,
                content_type=content_type,
                width=depth_info.get("width", intrinsics.width),
                height=depth_info.get("height", intrinsics.height),
                bit_depth=depth_info.get("bit_depth", 16),
            )
            logger.info(f"Stored depth artifact for scan {scan_id}: {len(depth_data)} bytes")
        
        # Store confidence map if provided
        if confidence_file is not None:
            confidence_data = await confidence_file.read()
            await confidence_file.seek(0)  # Reset for any further processing
            
            await uow.scan_artifacts.store_artifact_with_data(
                gridfs=uow.gridfs,
                scan_id=scan_id,
                artifact_type="confidence_u8",
                data=confidence_data,
                content_type="application/octet-stream",
                width=intrinsics.width,
                height=intrinsics.height,
                bit_depth=8,
            )
            logger.info(f"Stored confidence artifact for scan {scan_id}: {len(confidence_data)} bytes")
            
    except Exception as e:
        # Log error but don't fail the scan - artifact storage is optional
        logger.error(f"Failed to store artifacts for scan {scan_id}: {e}")


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
    metadata: Annotated[str, Form(description="JSON metadata (FoodScanRequest)")],
    depth_u16: Annotated[
        UploadFile | None, File(description="16-bit depth map (PNG, optional)")
    ] = None,
    confidence_u8: Annotated[
        UploadFile | None, File(description="Confidence map (PNG)")
    ] = None,
    uow: UnitOfWork = Depends(get_uow),
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
    
    # Validate depth map (optional for MVP - many devices don't support depth)
    # Accept both PNG format and raw 16-bit depth bytes
    depth_info = None
    if depth_u16 is not None:
        # Accept PNG or raw bytes (application/octet-stream)
        allowed_depth_types = ["image/png", "application/octet-stream", None]
        if depth_u16.content_type not in allowed_depth_types:
            logger.warning(f"Unexpected depth content type: {depth_u16.content_type}, allowing anyway")
        await validate_file_size(depth_u16, MAX_DEPTH_SIZE, "Depth map")
        
        # Validate depth dimensions match intrinsics
        depth_info = await validate_depth_dimensions(
            depth_u16,
            scan_request.intrinsics.width,
            scan_request.intrinsics.height,
        )
    else:
        logger.info("No depth image provided - using RGB-only food recognition")
    
    # Validate confidence map if provided (accepts PNG or raw bytes)
    if confidence_u8 is not None:
        # Accept PNG or raw bytes (application/octet-stream)
        allowed_conf_types = ["image/png", "application/octet-stream", None]
        if confidence_u8.content_type not in allowed_conf_types:
            logger.warning(f"Unexpected confidence content type: {confidence_u8.content_type}, allowing anyway")
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
            "has_depth": depth_info is not None,
            "depth_bit_depth": depth_info.get("bit_depth") if depth_info else None,
            "has_confidence": confidence_u8 is not None,
            "opt_in_artifacts": scan_request.opt_in_store_artifacts,
        },
    )
    
    # -------------------------------------------------------------------------
    # Step 6: Read RGB image
    # -------------------------------------------------------------------------
    rgb_data = await rgb.read()
    await rgb.seek(0)
    
    # -------------------------------------------------------------------------
    # Step 6a: Segmentation via FastSAM (MVP-2.3) - Feature flagged
    # -------------------------------------------------------------------------
    settings = get_settings()
    segmentation_result: SegmentationResult | None = None
    segmentation_time_ms: int | None = None
    
    if settings.segmentation_enabled:
        try:
            segmentation_client = get_segmentation_client()
            
            # Check health first
            is_seg_healthy = await segmentation_client.is_healthy()
            if not is_seg_healthy:
                logger.warning(
                    "Segmentation service unhealthy, skipping segmentation",
                    extra={"scan_id": scan_id},
                )
            else:
                # Run segmentation
                segmentation_result = await segmentation_client.segment(
                    image_data=rgb_data,
                    prompt="food on plate",  # Guide segmentation toward food
                    return_visualization=scan_request.opt_in_store_artifacts,
                )
                segmentation_time_ms = segmentation_result.processing_time_ms
                
                logger.info(
                    "Segmentation complete",
                    extra={
                        "scan_id": scan_id,
                        "masks_found": len(segmentation_result.masks),
                        "processing_time_ms": segmentation_time_ms,
                        "model_version": segmentation_result.model_version,
                    },
                )
                
                # Store segmentation mask artifact if opted in
                if scan_request.opt_in_store_artifacts and segmentation_result.visualization_base64:
                    import base64
                    vis_data = base64.b64decode(segmentation_result.visualization_base64)
                    await uow.scan_artifacts.store_artifact_with_data(
                        gridfs=uow.gridfs,
                        scan_id=scan_id,
                        artifact_type="segmentation_mask",
                        data=vis_data,
                        content_type="image/png",
                        width=segmentation_result.image_width,
                        height=segmentation_result.image_height,
                    )
                    logger.info(f"Stored segmentation visualization for scan {scan_id}")
                    
        except Exception as e:
            logger.warning(
                f"Segmentation failed, continuing without: {e}",
                extra={"scan_id": scan_id},
            )
    else:
        logger.debug("Segmentation disabled via feature flag")
    
    # -------------------------------------------------------------------------
    # Step 6b: Volume Computation (MVP-2.5) - Requires segmentation + depth
    # -------------------------------------------------------------------------
    volume_result = None
    volume_time_ms: int | None = None
    volume_ml_computed: float | None = None
    volume_quality: float | None = None
    plane_fit_inliers: int | None = None
    plane_fit_rmse: float | None = None
    
    if segmentation_result and segmentation_result.masks and depth_u16 is not None:
        try:
            # Read depth data for volume computation
            depth_data_for_volume = await depth_u16.read()
            await depth_u16.seek(0)  # Reset for further processing
            
            volume_service = get_volume_service()
            
            logger.info(
                "Computing volume",
                extra={
                    "scan_id": scan_id,
                    "masks_count": len(segmentation_result.masks),
                },
            )
            
            volume_result = volume_service.compute_volume(
                depth_data=depth_data_for_volume,
                segmentation_result=segmentation_result,
                intrinsics_dict=intrinsics_dict,
            )
            
            volume_ml_computed = volume_result.total_volume_ml
            volume_quality = volume_result.overall_quality_score
            volume_time_ms = volume_result.total_processing_time_ms
            plane_fit_inliers = volume_result.plane_fit_inliers
            plane_fit_rmse = volume_result.plane_fit_rmse
            
            logger.info(
                "Volume computation complete",
                extra={
                    "scan_id": scan_id,
                    "volume_ml": volume_ml_computed,
                    "quality_score": volume_quality,
                    "plane_inliers": plane_fit_inliers,
                    "plane_rmse": plane_fit_rmse,
                    "processing_time_ms": volume_time_ms,
                },
            )
            
        except VolumeError as e:
            logger.warning(
                f"Volume computation failed: {e.message}",
                extra={
                    "scan_id": scan_id,
                    "error_code": e.code.value,
                    "details": e.details,
                },
            )
            # Continue without volume - it's not critical
        except Exception as e:
            logger.warning(
                f"Unexpected error in volume computation: {e}",
                extra={"scan_id": scan_id},
            )
    else:
        if not segmentation_result or not segmentation_result.masks:
            logger.debug("Volume computation skipped: no segmentation masks")
        elif depth_u16 is None:
            logger.debug("Volume computation skipped: no depth data")
    
    # -------------------------------------------------------------------------
    # Step 7: Food Recognition via LLaVA/Ollama (MVP-2.4 + MVP-2.9)
    # -------------------------------------------------------------------------
    # Check if food recognition is enabled (default: True)
    use_food_recognition = os.getenv("FOOD_RECOGNITION_ENABLED", "true").lower() == "true"
    # Check if per-mask recognition is enabled (MVP-2.9, default: True)
    use_per_mask_recognition = os.getenv("PER_MASK_RECOGNITION_ENABLED", "true").lower() == "true"
    
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
            
            # MVP-2.9: Use per-mask recognition if segmentation available
            recognition_result = None
            used_per_mask = False
            
            if (
                use_per_mask_recognition
                and segmentation_result
                and segmentation_result.masks
                and len(segmentation_result.masks) > 0
            ):
                try:
                    logger.info(
                        f"Using per-mask recognition with {len(segmentation_result.masks)} masks",
                        extra={"scan_id": scan_id},
                    )
                    
                    per_mask_service = create_per_mask_service(food_service)
                    recognition_result = await per_mask_service.recognize_with_masks(
                        rgb_data=rgb_data,
                        segmentation_result=segmentation_result,
                        max_masks=5,
                        parallel=True,
                    )
                    
                    # Check if per-mask recognition was successful
                    if recognition_result.foods:
                        used_per_mask = True
                        logger.info(
                            f"Per-mask recognition successful: {len(recognition_result.foods)} foods",
                            extra={"scan_id": scan_id},
                        )
                    else:
                        logger.warning(
                            "Per-mask recognition returned no foods, falling back to whole-image",
                            extra={"scan_id": scan_id},
                        )
                        recognition_result = None
                        
                except Exception as e:
                    logger.warning(
                        f"Per-mask recognition failed, falling back to whole-image: {e}",
                        extra={"scan_id": scan_id},
                    )
                    recognition_result = None
            
            # Fallback to whole-image recognition if per-mask not used or failed
            if recognition_result is None:
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
                    "is_multi_food_plate": recognition_result.is_multi_food_plate,
                    "confidence": recognition_result.overall_confidence,
                    "processing_time_ms": recognition_result.processing_time_ms,
                    "used_per_mask": used_per_mask,  # MVP-2.9
                },
            )
            
            # MVP-3.4: Build detected_foods array (one entry per distinct food)
            detected_foods: list[DetectedFood] = []
            food_candidates: list[FoodCandidate] = []  # Legacy support
            
            for idx, food in enumerate(recognition_result.foods):
                # Get macros for this specific food
                food_macros = None
                if food.estimated_macros:
                    food_macros = Macros(
                        carbs=food.estimated_macros.carbs,
                        protein=food.estimated_macros.protein,
                        fat=food.estimated_macros.fat,
                        fiber=food.estimated_macros.fiber,
                    )
                
                # Create primary candidate for this detected food
                primary_candidate = FoodCandidate(
                    canonical_food_id=f"llava_{food.label.lower().replace(' ', '_')}_{idx}",
                    label=food.label,
                    probability=food.confidence,
                    is_mixed_dish=food.is_mixed_dish,
                    visible_components=food.visible_components,
                    estimated_grams=food.estimated_grams,
                    macros=food_macros,
                )
                
                # Add to legacy food_candidates for backward compatibility
                food_candidates.append(primary_candidate)
                
                # Generate portion-based alternatives for this specific food
                alternatives: list[FoodCandidate] = []
                variation_templates = [
                    {"suffix": "(smaller portion)", "gram_mult": 0.7, "macro_mult": 0.7},
                    {"suffix": "(larger portion)", "gram_mult": 1.4, "macro_mult": 1.4},
                ]
                
                for var_idx, template in enumerate(variation_templates):
                    var_macros = None
                    if food_macros:
                        mult = template['macro_mult']
                        var_macros = Macros(
                            carbs=round(food_macros.carbs * mult, 1),
                            protein=round(food_macros.protein * mult, 1),
                            fat=round(food_macros.fat * mult, 1),
                            fiber=round(food_macros.fiber * mult, 1),
                        )
                    
                    var_grams = None
                    if food.estimated_grams:
                        var_grams = round(food.estimated_grams * template['gram_mult'], 0)
                    
                    alt = FoodCandidate(
                        canonical_food_id=f"{primary_candidate.canonical_food_id}_var{var_idx + 1}",
                        label=f"{food.label} {template['suffix']}",
                        probability=max(0.1, food.confidence - (0.15 * (var_idx + 1))),
                        is_mixed_dish=food.is_mixed_dish,
                        visible_components=food.visible_components,
                        estimated_grams=var_grams,
                        macros=var_macros,
                    )
                    alternatives.append(alt)
                
                # Create DetectedFood entry
                detected_food = DetectedFood(
                    id=f"food_{idx}",
                    primary=primary_candidate,
                    alternatives=alternatives,
                    selected=True,  # All foods selected by default
                )
                detected_foods.append(detected_food)
            
            # Calculate totals from all detected foods
            primary_food = recognition_result.primary_food
            total_macros = recognition_result.total_estimated_macros
            is_multi_food_plate = recognition_result.is_multi_food_plate
            
            # Store LLaVA macros
            llava_macros: Macros | None = None
            if total_macros:
                llava_macros = Macros(
                    carbs=total_macros.carbs,
                    protein=total_macros.protein,
                    fat=total_macros.fat,
                    fiber=total_macros.fiber,
                )
            
            # Estimate grams from primary food
            grams_est = primary_food.estimated_grams if primary_food else 100.0
            
            # ---------------------------------------------------------------
            # USDA Nutrition Lookup (MVP-2.7)
            # ---------------------------------------------------------------
            usda_macros: Macros | None = None
            usda_food_id: str | None = None
            usda_food_name: str | None = None
            usda_match_score: float | None = None
            macro_source = MacroSource.LLAVA
            
            nutrition_service = get_nutrition_lookup_service()
            if nutrition_service and primary_food:
                try:
                    logger.info(f"Looking up USDA nutrition for: {primary_food.label}")
                    usda_result = await nutrition_service.search_food(
                        primary_food.label,
                        max_results=3,
                    )
                    
                    # Always capture match score for confidence calculation
                    usda_match_score = usda_result.match_score
                    
                    if usda_result.found and usda_result.nutrients:
                        # Scale USDA nutrients to estimated portion size
                        scaled_nutrients = usda_result.nutrients.scale_to_grams(
                            grams_est or 100.0
                        )
                        
                        usda_macros = Macros(
                            carbs=round(scaled_nutrients.carbs, 1),
                            protein=round(scaled_nutrients.protein, 1),
                            fat=round(scaled_nutrients.fat, 1),
                            fiber=round(scaled_nutrients.fiber, 1),
                        )
                        usda_food_id = usda_result.food_id
                        usda_food_name = usda_result.food_name
                        macro_source = MacroSource.USDA
                        
                        logger.info(
                            f"USDA match found: {usda_food_name} (ID: {usda_food_id}), "
                            f"match_score={usda_match_score:.2f}"
                        )
                except NutritionLookupError as e:
                    logger.warning(f"USDA lookup failed: {e.message}")
                except Exception as e:
                    logger.warning(f"Unexpected error in USDA lookup: {e}")
            
            # Use USDA macros as primary if available, else LLaVA
            if usda_macros:
                macros = usda_macros
            elif llava_macros:
                macros = llava_macros
            else:
                macros = Macros(carbs=0.0, protein=0.0, fat=0.0, fiber=0.0)
                macro_source = MacroSource.UNKNOWN
            
            # ---------------------------------------------------------------
            # Confidence Engine (MVP-2.8)
            # ---------------------------------------------------------------
            confidence_engine = get_confidence_engine()
            
            # Check if any food is a mixed dish
            is_mixed_dish = any(f.is_mixed_dish for f in recognition_result.foods)
            
            confidence_result = confidence_engine.calculate_confidence(
                recognition_confidence=recognition_result.overall_confidence,
                usda_match_score=usda_match_score,
                has_usda_match=usda_macros is not None,
                macro_source=macro_source,
                has_depth_data=depth_info is not None,
                is_mixed_dish=is_mixed_dish,
                estimated_grams=grams_est,
            )
            
            # Use confidence engine results
            scan_quality = confidence_result.scan_quality
            uncertainty_reasons = confidence_result.uncertainty_reasons
            
            # Calculate macro ranges using confidence-based variance
            macro_ranges = confidence_engine.calculate_macro_ranges(
                macros, confidence_result.macro_variance
            )
            
            # Add unknown food if no candidates
            if not food_candidates:
                uncertainty_reasons.append(UncertaintyReason.UNKNOWN_FOOD)
            
            # For legacy support: selected_food is the first candidate
            selected_food = food_candidates[0] if food_candidates else None
            
            # Calculate total grams from all detected foods
            total_grams = sum(
                df.primary.estimated_grams or 0 
                for df in detected_foods
            ) or grams_est or 100.0
            
            # -----------------------------------------------------------------
            # Store artifacts to GridFS (if opted in)
            # -----------------------------------------------------------------
            if scan_request.opt_in_store_artifacts:
                await _store_scan_artifacts(
                    uow=uow,
                    scan_id=scan_id,
                    rgb_data=rgb_data,
                    depth_file=depth_u16,
                    depth_info=depth_info,
                    confidence_file=confidence_u8,
                    intrinsics=scan_request.intrinsics,
                )
            
            return FoodScanResponse(
                scan_id=scan_id,
                # MVP-3.4: Multi-food support
                is_multi_food_plate=is_multi_food_plate,
                detected_foods=detected_foods,
                # Legacy support
                food_candidates=food_candidates,
                selected_food=selected_food,
                # Totals (MVP-2.5: Use computed volume if available, else estimate)
                volume_ml=volume_ml_computed if volume_ml_computed is not None else total_grams * 0.9,
                grams_est=total_grams,
                macros=macros,
                macro_ranges=macro_ranges,
                macro_source=macro_source,
                llava_macros=llava_macros,
                usda_macros=usda_macros,
                usda_food_id=usda_food_id,
                usda_food_name=usda_food_name,
                confidence_score=confidence_result.overall_score,
                scan_quality=scan_quality,
                uncertainty_reasons=uncertainty_reasons,
                debug={
                    "validation": "passed",
                    "depth_info": depth_info,
                    "provider": recognition_result.provider,
                    "processing_time_ms": recognition_result.processing_time_ms,
                    "raw_response": recognition_result.raw_response[:500] if scan_request.opt_in_store_artifacts else None,
                    "usda_lookup": usda_food_id is not None,
                    "is_multi_food_plate": is_multi_food_plate,
                    "detected_foods_count": len(detected_foods),
                    "confidence_factors": confidence_result.factors.to_dict(),
                    "macro_variance": f"±{confidence_result.macro_variance * 100:.0f}%",
                    # Segmentation info (MVP-2.3)
                    "segmentation_enabled": settings.segmentation_enabled,
                    "segmentation_time_ms": segmentation_time_ms,
                    "segmentation_masks_count": len(segmentation_result.masks) if segmentation_result else None,
                    "segmentation_provider": segmentation_result.model_version if segmentation_result else None,
                    # Per-mask recognition info (MVP-2.9)
                    "per_mask_recognition_used": used_per_mask,
                    # Volume computation info (MVP-2.5)
                    "volume_computed": volume_ml_computed is not None,
                    "volume_ml": volume_ml_computed,
                    "volume_quality_score": volume_quality,
                    "volume_time_ms": volume_time_ms,
                    "plane_fit_inliers": plane_fit_inliers,
                    "plane_fit_rmse": plane_fit_rmse,
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
    
    # Store artifacts even in fallback case (if opted in)
    if scan_request.opt_in_store_artifacts:
        await _store_scan_artifacts(
            uow=uow,
            scan_id=scan_id,
            rgb_data=rgb_data,
            depth_file=depth_u16,
            depth_info=depth_info,
            confidence_file=confidence_u8,
            intrinsics=scan_request.intrinsics,
        )
    
    mock_candidate = FoodCandidate(
        canonical_food_id="fallback_001",
        label="[Recognition unavailable] Please try again",
        probability=0.5,
        is_mixed_dish=False,
    )
    
    mock_detected_food = DetectedFood(
        id="food_0",
        primary=mock_candidate,
        alternatives=[],
        selected=True,
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
        # MVP-3.4: Multi-food support
        is_multi_food_plate=False,
        detected_foods=[mock_detected_food],
        # Legacy support
        food_candidates=[mock_candidate],
        selected_food=mock_candidate,
        volume_ml=volume_ml_computed if volume_ml_computed is not None else 200.0,
        grams_est=150.0,
        macros=mock_macros,
        macro_ranges=mock_ranges,
        macro_source=MacroSource.UNKNOWN,
        llava_macros=None,
        usda_macros=None,
        usda_food_id=None,
        usda_food_name=None,
        confidence_score=0.5,
        scan_quality=ScanQuality.OK,
        uncertainty_reasons=[UncertaintyReason.RECOGNITION_SERVICE_UNAVAILABLE],
        debug={
            "validation": "passed",
            "depth_info": depth_info,
            "note": "Food recognition service unavailable, using fallback",
            "segmentation_enabled": settings.segmentation_enabled,
            "segmentation_time_ms": segmentation_time_ms,
            "segmentation_masks_count": len(segmentation_result.masks) if segmentation_result else None,
            # Volume computation info (MVP-2.5)
            "volume_computed": volume_ml_computed is not None,
            "volume_ml": volume_ml_computed,
            "volume_quality_score": volume_quality,
            "volume_time_ms": volume_time_ms,
            "plane_fit_inliers": plane_fit_inliers,
            "plane_fit_rmse": plane_fit_rmse,
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
    "/segmentation/health",
    summary="Check segmentation service health",
    description="Check if the segmentation service (FastSAM) is available.",
)
async def check_segmentation_health() -> dict:
    """
    Check if the segmentation service is healthy and available.
    
    Returns:
        dict with status, model_loaded, gpu_available, and model information
    """
    settings = get_settings()
    
    if not settings.segmentation_enabled:
        return {
            "status": "disabled",
            "enabled": False,
            "message": "Segmentation is disabled via feature flag (SEGMENTATION_ENABLED=false)",
        }
    
    try:
        client = get_segmentation_client()
        health = await client.health_check()
        
        return {
            "status": health.get("status", "unknown"),
            "enabled": True,
            "model_loaded": health.get("model_loaded", False),
            "gpu_available": health.get("gpu_available", False),
            "gpu_name": health.get("gpu_name"),
            "gpu_memory_used_mb": health.get("gpu_memory_used_mb"),
            "model_version": health.get("model_version", "unknown"),
        }
    except Exception as e:
        logger.error(f"Segmentation health check failed: {e}")
        return {
            "status": "error",
            "enabled": True,
            "model_loaded": False,
            "gpu_available": False,
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
# Artifact Retrieval Endpoints (GridFS Storage)
# =============================================================================


@router.get(
    "/scan/{scan_id}/artifacts/{artifact_type}",
    summary="Get scan artifact",
    description="""
    Retrieve a scan artifact (RGB image, depth map, or confidence map).
    
    **Artifact Types:**
    - `rgb`: Original RGB image (JPEG)
    - `depth_u16`: 16-bit depth map (PNG or raw bytes)
    - `confidence_u8`: 8-bit confidence map
    
    **Note:** Artifacts are only available if the user opted in via
    `opt_in_store_artifacts=true` during the scan.
    """,
    responses={
        200: {
            "description": "Artifact binary data",
            "content": {
                "image/jpeg": {},
                "image/png": {},
                "application/octet-stream": {},
            },
        },
        404: {"description": "Artifact not found"},
    },
)
async def get_scan_artifact(
    scan_id: str,
    artifact_type: str,
    uow: UnitOfWork = Depends(get_uow),
) -> Response:
    """
    Retrieve artifact binary data for a scan.
    
    Returns the raw binary data with appropriate Content-Type header.
    """
    # Validate artifact type
    valid_types = {"rgb", "depth_u16", "confidence_u8"}
    if artifact_type not in valid_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=FoodScanError(
                error_code=ScanErrorCode.PROCESSING_ERROR,
                message=f"Invalid artifact type: {artifact_type}",
                details={"valid_types": list(valid_types)},
            ).model_dump(),
        )
    
    # Try to retrieve artifact
    result = await uow.scan_artifacts.get_artifact_data(
        gridfs=uow.gridfs,
        scan_id=scan_id,
        artifact_type=artifact_type,
    )
    
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=FoodScanError(
                error_code=ScanErrorCode.PROCESSING_ERROR,
                message=f"Artifact not found: {scan_id}/{artifact_type}",
                details={
                    "scan_id": scan_id,
                    "artifact_type": artifact_type,
                    "note": "Artifact may not exist or scan was not stored",
                },
            ).model_dump(),
        )
    
    data, artifact = result
    
    logger.info(
        f"Serving artifact {artifact_type} for scan {scan_id}: "
        f"{len(data)} bytes"
    )
    
    return Response(
        content=data,
        media_type=artifact.content_type,
        headers={
            "Content-Disposition": f'inline; filename="{scan_id}_{artifact_type}"',
            "X-Artifact-Type": artifact_type,
            "X-Scan-Id": scan_id,
        },
    )


@router.get(
    "/scan/{scan_id}/artifacts",
    summary="List scan artifacts",
    description="List all available artifacts for a scan.",
)
async def list_scan_artifacts(
    scan_id: str,
    uow: UnitOfWork = Depends(get_uow),
) -> dict:
    """
    List all artifacts available for a scan.
    
    Returns metadata about stored artifacts without the binary data.
    """
    artifacts = await uow.scan_artifacts.get_artifacts_for_scan(scan_id)
    
    return {
        "scan_id": scan_id,
        "artifacts": [
            {
                "artifact_type": a.artifact_type,
                "content_type": a.content_type,
                "size_bytes": a.size_bytes,
                "width": a.width,
                "height": a.height,
                "created_at": a.created_at.isoformat() if a.created_at else None,
                "ttl_expires_at": a.ttl_expires_at.isoformat() if a.ttl_expires_at else None,
            }
            for a in artifacts
        ],
        "count": len(artifacts),
    }


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
async def save_meal_estimate(
    request: SaveMealRequest,
    uow: UnitOfWork = Depends(get_uow),
) -> MealEstimateResponse:
    """
    Save a confirmed meal estimate.
    
    MVP-3.3: Stores to meal_estimates collection (separate from pump_data).
    """
    created_at = datetime.now(UTC)
    
    logger.info(
        "Saving meal estimate",
        extra={
            "scan_id": request.scan_id,
            "user_id": request.user_id,
            "food_label": request.food_label,
            "confidence": request.confidence,
        },
    )
    
    # Persist to MongoDB via MealEstimateRepository
    meal_id = await uow.meal_estimates.create_from_scan(
        scan_id=request.scan_id,
        user_id=request.user_id,
        canonical_food_id=request.canonical_food_id,
        food_label=request.food_label,
        macros=request.macros.model_dump(),
        confidence=request.confidence,
        macro_ranges=request.macro_ranges.model_dump() if request.macro_ranges else None,
        uncertainty_reasons=request.uncertainty_reasons,
    )
    
    logger.info(f"Meal estimate saved with ID: {meal_id}")
    
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
    uow: UnitOfWork = Depends(get_uow),
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
    
    # Query MongoDB via MealEstimateRepository
    meals = await uow.meal_estimates.get_user_meals(
        user_id=user_id,
        start=start_date,
        end=end_date,
        limit=limit,
    )
    
    # Convert to response format
    meal_responses = [
        MealEstimateResponse(
            id=str(meal.id) if meal.id else "",
            scan_id=meal.scan_id,
            user_id=meal.user_id,
            food_label=meal.food_label,
            macros=meal.macros if meal.macros else Macros(carbs=0, protein=0, fat=0, fiber=0),
            confidence=meal.confidence,
            created_at=meal.created_at,
        )
        for meal in meals
    ]
    
    return MealListResponse(meals=meal_responses, total=len(meal_responses))
