"""API tests for the segmentation service.

These tests verify the API contract without requiring GPU/model.
"""

import base64
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


# Sample test image (1x1 red pixel PNG)
TINY_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
)


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_when_model_loaded(self):
        """Test health returns ok when model is loaded."""
        # Mock the segmenter to simulate loaded state
        with patch("segmentation_api.main.segmenter") as mock_segmenter:
            mock_segmenter.is_loaded = True
            mock_segmenter.model_version = "FastSAM-s"
            mock_segmenter.get_gpu_info.return_value = {
                "gpu_available": True,
                "gpu_name": "NVIDIA GeForce RTX 3070",
                "gpu_memory_used_mb": 1500.0,
            }

            # Import after mocking to get mocked version
            from segmentation_api.main import app
            client = TestClient(app)

            response = client.get("/health")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"
            assert data["model_loaded"] is True
            assert data["gpu_available"] is True

    def test_health_when_model_not_loaded(self):
        """Test health returns degraded when model isn't loaded."""
        with patch("segmentation_api.main.segmenter") as mock_segmenter:
            mock_segmenter.is_loaded = False
            mock_segmenter.model_version = "FastSAM-s"
            mock_segmenter.get_gpu_info.return_value = {
                "gpu_available": False,
                "gpu_name": None,
                "gpu_memory_used_mb": None,
            }

            from segmentation_api.main import app
            client = TestClient(app)

            response = client.get("/health")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "degraded"
            assert data["model_loaded"] is False


class TestSegmentEndpoint:
    """Tests for /segment endpoint."""

    def test_segment_returns_503_when_model_not_loaded(self):
        """Test segment returns 503 when model isn't ready."""
        with patch("segmentation_api.main.segmenter") as mock_segmenter:
            mock_segmenter.is_loaded = False

            from segmentation_api.main import app
            client = TestClient(app)

            response = client.post(
                "/segment",
                json={"image_base64": TINY_PNG_BASE64},
            )

            assert response.status_code == 503
            assert "not loaded" in response.json()["detail"].lower()

    def test_segment_returns_400_for_invalid_base64(self):
        """Test segment returns 400 for invalid base64."""
        with patch("segmentation_api.main.segmenter") as mock_segmenter:
            mock_segmenter.is_loaded = True

            from segmentation_api.main import app
            client = TestClient(app)

            response = client.post(
                "/segment",
                json={"image_base64": "not-valid-base64!!!"},
            )

            assert response.status_code == 400
            assert "invalid" in response.json()["detail"].lower()

    def test_segment_success(self):
        """Test successful segmentation."""
        mock_result = {
            "masks": [
                {
                    "mask_base64": TINY_PNG_BASE64,
                    "bbox": (10, 20, 100, 150),
                    "confidence": 0.95,
                    "area_pixels": 5000,
                    "centroid": (60, 95),
                }
            ],
            "processing_time_ms": 150,
            "image_width": 100,
            "image_height": 100,
            "model_version": "FastSAM-s",
            "visualization_base64": None,
        }

        with patch("segmentation_api.main.segmenter") as mock_segmenter:
            mock_segmenter.is_loaded = True
            mock_segmenter.segment.return_value = mock_result

            from segmentation_api.main import app
            client = TestClient(app)

            response = client.post(
                "/segment",
                json={
                    "image_base64": TINY_PNG_BASE64,
                    "prompt": "food on plate",
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert len(data["masks"]) == 1
            assert data["masks"][0]["confidence"] == 0.95
            assert data["processing_time_ms"] == 150


class TestRootEndpoint:
    """Tests for root / endpoint."""

    def test_root_returns_service_info(self):
        """Test root returns service information."""
        with patch("segmentation_api.main.segmenter") as mock_segmenter:
            mock_segmenter.model_version = "FastSAM-s"

            from segmentation_api.main import app
            client = TestClient(app)

            response = client.get("/")

            assert response.status_code == 200
            data = response.json()
            assert data["service"] == "Food Segmentation Service"
            assert "endpoints" in data
