"""Request and response models for the segmentation API."""

from pydantic import BaseModel, Field


class SegmentationMask(BaseModel):
    """A single segmentation mask for a detected region."""

    mask_base64: str = Field(description="Base64-encoded PNG mask image")
    bbox: tuple[int, int, int, int] = Field(
        description="Bounding box as (x, y, width, height)"
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")
    area_pixels: int = Field(ge=0, description="Number of pixels in the mask")
    centroid: tuple[int, int] = Field(description="Center point of mask (x, y)")


class SegmentRequest(BaseModel):
    """Request body for segmentation endpoint."""

    image_base64: str = Field(description="Base64-encoded image (PNG or JPEG)")
    prompt: str | None = Field(
        default=None,
        description="Optional text prompt to guide segmentation (e.g., 'food on plate')",
    )
    return_visualization: bool = Field(
        default=False,
        description="If true, return a visualization image with masks overlaid",
    )


class SegmentResponse(BaseModel):
    """Response from segmentation endpoint."""

    masks: list[SegmentationMask] = Field(description="List of detected masks")
    processing_time_ms: int = Field(description="Time taken for inference in milliseconds")
    image_width: int = Field(description="Width of input image")
    image_height: int = Field(description="Height of input image")
    model_version: str = Field(description="Version/name of the segmentation model")
    visualization_base64: str | None = Field(
        default=None,
        description="Base64-encoded visualization image (if requested)",
    )


class HealthResponse(BaseModel):
    """Response from health check endpoint."""

    status: str = Field(description="Service status")
    model_loaded: bool = Field(description="Whether the model is loaded and ready")
    gpu_available: bool = Field(description="Whether GPU is available")
    gpu_name: str | None = Field(default=None, description="Name of the GPU if available")
    gpu_memory_used_mb: float | None = Field(
        default=None, description="GPU memory currently used in MB"
    )
    model_version: str = Field(description="Version/name of the loaded model")
