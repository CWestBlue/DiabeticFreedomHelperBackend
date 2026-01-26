"""Unit tests for the segmentation service client.

MVP-2.3: Segmentation Module Interface + Initial GPU Model
"""

import base64
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from diabetic_api.services.segmentation.client import SegmentationClient
from diabetic_api.models.food_scan import SegmentationResult, SegmentationMask


# Sample test image (1x1 red pixel PNG)
TINY_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
)
TINY_PNG_BYTES = base64.b64decode(TINY_PNG_BASE64)


class TestSegmentationClient:
    """Tests for SegmentationClient."""

    @pytest.fixture
    def client(self):
        """Create a test client."""
        return SegmentationClient(
            base_url="http://localhost:8001",
            timeout=10.0,
        )

    @pytest.mark.asyncio
    async def test_health_check_success(self, client):
        """Test successful health check."""
        mock_response = {
            "status": "ok",
            "model_loaded": True,
            "gpu_available": True,
            "gpu_name": "NVIDIA GeForce RTX 3070",
            "gpu_memory_used_mb": 1500.0,
            "model_version": "FastSAM-s",
        }

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http_client = AsyncMock()
            mock_http_client.get = AsyncMock(
                return_value=MagicMock(
                    json=lambda: mock_response,
                    raise_for_status=lambda: None,
                )
            )
            mock_get_client.return_value = mock_http_client

            result = await client.health_check()

            assert result["status"] == "ok"
            assert result["model_loaded"] is True
            assert result["gpu_available"] is True
            mock_http_client.get.assert_called_once_with("/health")

    @pytest.mark.asyncio
    async def test_is_healthy_returns_true_when_ok(self, client):
        """Test is_healthy returns True when service is ready."""
        with patch.object(client, "health_check") as mock_health:
            mock_health.return_value = {
                "status": "ok",
                "model_loaded": True,
            }

            result = await client.is_healthy()

            assert result is True

    @pytest.mark.asyncio
    async def test_is_healthy_returns_false_when_model_not_loaded(self, client):
        """Test is_healthy returns False when model isn't loaded."""
        with patch.object(client, "health_check") as mock_health:
            mock_health.return_value = {
                "status": "degraded",
                "model_loaded": False,
            }

            result = await client.is_healthy()

            assert result is False

    @pytest.mark.asyncio
    async def test_is_healthy_returns_false_on_error(self, client):
        """Test is_healthy returns False when health check fails."""
        with patch.object(client, "health_check") as mock_health:
            mock_health.side_effect = Exception("Connection refused")

            result = await client.is_healthy()

            assert result is False

    @pytest.mark.asyncio
    async def test_segment_success(self, client):
        """Test successful segmentation request."""
        mock_mask_data = base64.b64encode(TINY_PNG_BYTES).decode()
        mock_response = {
            "masks": [
                {
                    "mask_base64": mock_mask_data,
                    "bbox": [10, 20, 100, 150],
                    "confidence": 0.95,
                    "area_pixels": 5000,
                    "centroid": [60, 95],
                }
            ],
            "processing_time_ms": 150,
            "image_width": 1920,
            "image_height": 1080,
            "model_version": "FastSAM-s",
            "visualization_base64": None,
        }

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http_client = AsyncMock()
            mock_http_client.post = AsyncMock(
                return_value=MagicMock(
                    json=lambda: mock_response,
                    raise_for_status=lambda: None,
                )
            )
            mock_get_client.return_value = mock_http_client

            result = await client.segment(
                image_data=TINY_PNG_BYTES,
                prompt="food on plate",
            )

            assert isinstance(result, SegmentationResult)
            assert len(result.masks) == 1
            assert result.masks[0].confidence == 0.95
            assert result.masks[0].bbox == (10, 20, 100, 150)
            assert result.masks[0].area_pixels == 5000
            assert result.processing_time_ms == 150
            assert result.model_version == "FastSAM-s"

            # Verify request format
            mock_http_client.post.assert_called_once()
            call_args = mock_http_client.post.call_args
            assert call_args[0][0] == "/segment"
            request_body = call_args[1]["json"]
            assert "image_base64" in request_body
            assert request_body["prompt"] == "food on plate"

    @pytest.mark.asyncio
    async def test_segment_empty_masks(self, client):
        """Test segmentation with no masks detected."""
        mock_response = {
            "masks": [],
            "processing_time_ms": 100,
            "image_width": 1920,
            "image_height": 1080,
            "model_version": "FastSAM-s",
            "visualization_base64": None,
        }

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http_client = AsyncMock()
            mock_http_client.post = AsyncMock(
                return_value=MagicMock(
                    json=lambda: mock_response,
                    raise_for_status=lambda: None,
                )
            )
            mock_get_client.return_value = mock_http_client

            result = await client.segment(image_data=TINY_PNG_BYTES)

            assert isinstance(result, SegmentationResult)
            assert len(result.masks) == 0
            assert result.processing_time_ms == 100


class TestSegmentationMaskModel:
    """Tests for SegmentationMask Pydantic model."""

    def test_valid_mask(self):
        """Test creating a valid mask."""
        mask = SegmentationMask(
            mask_base64="base64data",
            bbox=(10, 20, 100, 150),
            confidence=0.95,
            area_pixels=5000,
            centroid=(60, 95),
        )

        assert mask.mask_base64 == "base64data"
        assert mask.bbox == (10, 20, 100, 150)
        assert mask.confidence == 0.95
        assert mask.area_pixels == 5000
        assert mask.centroid == (60, 95)

    def test_confidence_bounds(self):
        """Test confidence must be between 0 and 1."""
        # Valid
        mask = SegmentationMask(
            mask_base64="data",
            bbox=(0, 0, 10, 10),
            confidence=0.0,
            area_pixels=100,
            centroid=(5, 5),
        )
        assert mask.confidence == 0.0

        mask = SegmentationMask(
            mask_base64="data",
            bbox=(0, 0, 10, 10),
            confidence=1.0,
            area_pixels=100,
            centroid=(5, 5),
        )
        assert mask.confidence == 1.0


class TestSegmentationResultModel:
    """Tests for SegmentationResult Pydantic model."""

    def test_valid_result(self):
        """Test creating a valid result."""
        result = SegmentationResult(
            masks=[],
            processing_time_ms=150,
            image_width=1920,
            image_height=1080,
            model_version="FastSAM-s",
        )

        assert result.masks == []
        assert result.processing_time_ms == 150
        assert result.image_width == 1920
        assert result.image_height == 1080
        assert result.model_version == "FastSAM-s"
        assert result.visualization_base64 is None

    def test_result_with_visualization(self):
        """Test result with visualization image."""
        result = SegmentationResult(
            masks=[],
            processing_time_ms=200,
            image_width=1920,
            image_height=1080,
            model_version="FastSAM-s",
            visualization_base64="base64visualizationdata",
        )

        assert result.visualization_base64 == "base64visualizationdata"
