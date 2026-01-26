"""HTTP client for the segmentation microservice."""

import base64
import logging
from functools import lru_cache

import httpx

from diabetic_api.core.config import get_settings
from diabetic_api.models.food_scan import SegmentationMask, SegmentationResult

logger = logging.getLogger(__name__)


class SegmentationClient:
    """Client for communicating with the segmentation microservice."""

    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        """
        Initialize the segmentation client.

        Args:
            base_url: Base URL of the segmentation service (e.g., "http://localhost:8001")
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout),
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def health_check(self) -> dict:
        """
        Check the health of the segmentation service.

        Returns:
            Health status dictionary with model_loaded, gpu_available, etc.

        Raises:
            httpx.HTTPError: If the request fails
        """
        client = await self._get_client()
        response = await client.get("/health")
        response.raise_for_status()
        return response.json()

    async def is_healthy(self) -> bool:
        """
        Check if the segmentation service is healthy and ready.

        Returns:
            True if service is healthy and model is loaded
        """
        try:
            health = await self.health_check()
            return health.get("status") == "ok" and health.get("model_loaded", False)
        except Exception as e:
            logger.warning(f"Segmentation service health check failed: {e}")
            return False

    async def segment(
        self,
        image_data: bytes,
        prompt: str | None = None,
        return_visualization: bool = False,
    ) -> SegmentationResult:
        """
        Segment an image to identify distinct regions.

        Args:
            image_data: Raw image bytes (PNG or JPEG)
            prompt: Optional text prompt to guide segmentation (e.g., "food on plate")
            return_visualization: Whether to return a visualization image

        Returns:
            SegmentationResult with masks and metadata

        Raises:
            httpx.HTTPError: If the request fails
            ValueError: If the response is invalid
        """
        client = await self._get_client()

        # Encode image as base64
        image_base64 = base64.b64encode(image_data).decode("utf-8")

        # Build request
        request_body = {
            "image_base64": image_base64,
            "prompt": prompt,
            "return_visualization": return_visualization,
        }

        logger.debug(f"Sending segmentation request (image size: {len(image_data)} bytes)")

        response = await client.post("/segment", json=request_body)
        response.raise_for_status()

        data = response.json()

        # Parse response into our model
        masks = [
            SegmentationMask(
                mask_base64=m["mask_base64"],
                bbox=tuple(m["bbox"]),
                confidence=m["confidence"],
                area_pixels=m["area_pixels"],
                centroid=tuple(m["centroid"]),
            )
            for m in data.get("masks", [])
        ]

        result = SegmentationResult(
            masks=masks,
            processing_time_ms=data["processing_time_ms"],
            image_width=data["image_width"],
            image_height=data["image_height"],
            model_version=data["model_version"],
            visualization_base64=data.get("visualization_base64"),
        )

        logger.info(
            f"Segmentation complete: {len(masks)} masks in {result.processing_time_ms}ms"
        )

        return result


@lru_cache
def get_segmentation_client() -> SegmentationClient:
    """
    Get a cached segmentation client instance.

    Returns:
        SegmentationClient configured from settings
    """
    settings = get_settings()
    return SegmentationClient(
        base_url=settings.segmentation_service_url,
        timeout=settings.segmentation_timeout,
    )
