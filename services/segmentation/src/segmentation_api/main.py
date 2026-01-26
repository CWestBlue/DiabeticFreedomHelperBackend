"""FastAPI application for the segmentation service."""

import base64
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .models import HealthResponse, SegmentRequest, SegmentResponse
from .segmentation import segmenter

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model on startup, cleanup on shutdown."""
    logger.info("Starting segmentation service...")
    try:
        segmenter.load_model()
        logger.info("Segmentation service ready")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # Continue running - health check will report model not loaded
    yield
    logger.info("Shutting down segmentation service...")


app = FastAPI(
    title="Food Segmentation Service",
    description="FastSAM-based image segmentation for food detection",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check service health and model status."""
    gpu_info = segmenter.get_gpu_info()

    return HealthResponse(
        status="ok" if segmenter.is_loaded else "degraded",
        model_loaded=segmenter.is_loaded,
        gpu_available=gpu_info["gpu_available"],
        gpu_name=gpu_info["gpu_name"],
        gpu_memory_used_mb=gpu_info["gpu_memory_used_mb"],
        model_version=segmenter.model_version,
    )


@app.post("/segment", response_model=SegmentResponse)
async def segment_image(request: SegmentRequest) -> SegmentResponse:
    """
    Segment an image to identify distinct regions.

    The image should be base64-encoded PNG or JPEG.
    Returns a list of segmentation masks with bounding boxes and confidence scores.
    """
    if not segmenter.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Service is starting up.",
        )

    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image_base64)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid base64 image data: {e}",
        ) from e

    try:
        # Run segmentation
        result = segmenter.segment(
            image_data=image_data,
            prompt=request.prompt,
            return_visualization=request.return_visualization,
        )

        return SegmentResponse(
            masks=[
                {
                    "mask_base64": m["mask_base64"],
                    "bbox": m["bbox"],
                    "confidence": m["confidence"],
                    "area_pixels": m["area_pixels"],
                    "centroid": m["centroid"],
                }
                for m in result["masks"]
            ],
            processing_time_ms=result["processing_time_ms"],
            image_width=result["image_width"],
            image_height=result["image_height"],
            model_version=result["model_version"],
            visualization_base64=result.get("visualization_base64"),
        )

    except Exception as e:
        logger.exception("Segmentation failed")
        raise HTTPException(
            status_code=500,
            detail=f"Segmentation failed: {e}",
        ) from e


@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "Food Segmentation Service",
        "version": "1.0.0",
        "model": segmenter.model_version,
        "endpoints": {
            "/health": "Service health check",
            "/segment": "POST - Segment an image",
            "/docs": "OpenAPI documentation",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "segmentation_api.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )
