"""Food Scan API routes.

Meal Vision Feature - MVP-2.1
Endpoint for processing food images with depth data.
"""

from datetime import UTC, datetime
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

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

router = APIRouter()


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
    - `rgb`: JPEG image file (required)
    - `depth_u16`: 16-bit PNG depth map in millimeters (required)
    - `confidence_u8`: 8-bit PNG confidence map, 0-255 (optional)
    - `metadata`: JSON string with FoodScanRequest fields (required)
    
    **MVP Constraints:**
    - Food must be on a plate/table
    - Table surface must be visible
    - Plate must be fully in frame
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
    
    This is a stub implementation that returns mock data.
    Real ML pipeline will be implemented in MVP-2.3 through MVP-2.8.
    """
    import json

    # Parse and validate metadata
    try:
        request_data = json.loads(metadata)
        scan_request = FoodScanRequest(**request_data)
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=FoodScanError(
                error_code=ScanErrorCode.PROCESSING_ERROR,
                message=f"Invalid JSON in metadata: {e}",
            ).model_dump(),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )

    # Validate file types
    if rgb.content_type not in ["image/jpeg", "image/jpg"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=FoodScanError(
                error_code=ScanErrorCode.PROCESSING_ERROR,
                message="RGB image must be JPEG format",
            ).model_dump(),
        )

    if depth_u16.content_type != "image/png":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=FoodScanError(
                error_code=ScanErrorCode.PROCESSING_ERROR,
                message="Depth map must be PNG format",
            ).model_dump(),
        )

    # Generate scan ID with user context
    scan_id = f"scan_{uuid4().hex[:12]}"

    # Log scan request (for debugging)
    # TODO: Replace print with proper logging (MVP-4.3)
    print(f"Food scan request: user={scan_request.user_id}, device={scan_request.device.model}")

    # ==========================================================================
    # STUB: Return mock response
    # TODO: Replace with actual ML pipeline (MVP-2.3 through MVP-2.8)
    # ==========================================================================

    mock_candidate = FoodCandidate(
        canonical_food_id="stub_food_001",
        label="[STUB] Unknown Food",
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
        uncertainty_reasons=[],
        debug=None,
        processed_at=datetime.now(UTC),
    )


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
        detail=f"Scan {scan_id} not found (storage not yet implemented)",
    )
