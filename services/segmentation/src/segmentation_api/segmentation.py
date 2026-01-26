"""FastSAM model wrapper for food image segmentation using ultralytics."""

import base64
import io
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from .config import settings

logger = logging.getLogger(__name__)


class FastSAMSegmenter:
    """Wrapper for FastSAM segmentation model using ultralytics."""

    def __init__(self) -> None:
        """Initialize the segmenter (model loaded lazily)."""
        self._model = None
        self._device = None
        self._model_version = settings.model_name

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None

    @property
    def model_version(self) -> str:
        """Get the model version string."""
        return self._model_version

    def load_model(self) -> None:
        """Load the FastSAM model into memory."""
        if self._model is not None:
            logger.info("Model already loaded")
            return

        logger.info(f"Loading FastSAM model: {settings.model_name}")
        start_time = time.time()

        # Determine device
        if settings.device == "cuda" and torch.cuda.is_available():
            self._device = "cuda"
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")

            # Set memory limit if specified
            if settings.gpu_memory_limit_gb > 0:
                try:
                    total_mem = torch.cuda.get_device_properties(0).total_memory
                    fraction = min(1.0, (settings.gpu_memory_limit_gb * 1e9) / total_mem)
                    torch.cuda.set_per_process_memory_fraction(fraction)
                except Exception as e:
                    logger.warning(f"Could not set GPU memory limit: {e}")
        else:
            self._device = "cpu"
            logger.info("Using CPU")

        # Import and load model using ultralytics directly
        try:
            from ultralytics import YOLO

            # Determine model file - FastSAM models are YOLO-compatible
            model_file = "FastSAM-s.pt" if "s" in settings.model_name.lower() else "FastSAM-x.pt"
            model_path = Path(settings.model_path) / model_file

            # If model doesn't exist locally, ultralytics will download it
            if not model_path.exists():
                logger.info(f"Model not found at {model_path}, will download")
                model_path.parent.mkdir(parents=True, exist_ok=True)
                # Use just the model name - ultralytics will download it
                self._model = YOLO(model_file)
            else:
                self._model = YOLO(str(model_path))

            load_time = time.time() - start_time
            logger.info(f"Model loaded in {load_time:.2f}s")

        except Exception as e:
            logger.error(f"Failed to load FastSAM model: {e}")
            raise RuntimeError(f"Failed to load FastSAM: {e}") from e

    def segment(
        self,
        image_data: bytes,
        prompt: str | None = None,
        return_visualization: bool = False,
    ) -> dict:
        """
        Segment an image and return masks.

        Args:
            image_data: Raw image bytes (PNG or JPEG)
            prompt: Optional text prompt (not used in basic FastSAM, kept for API compatibility)
            return_visualization: Whether to return a visualization image

        Returns:
            Dictionary with masks, timing, and optional visualization
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        start_time = time.time()

        # Decode image
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image_np = np.array(image)
        height, width = image_np.shape[:2]

        # Run FastSAM inference using ultralytics
        results = self._model(
            image_np,
            device=self._device,
            retina_masks=True,
            conf=settings.confidence_threshold,
            iou=settings.iou_threshold,
            verbose=False,
        )

        # Process results
        mask_results = []
        
        if results and len(results) > 0 and results[0].masks is not None:
            masks_data = results[0].masks.data  # Tensor of shape (N, H, W)
            
            # Convert to numpy if tensor
            if torch.is_tensor(masks_data):
                masks_np = masks_data.cpu().numpy()
            else:
                masks_np = masks_data

            for i, mask in enumerate(masks_np[: settings.max_masks]):
                # Ensure mask is 2D
                if mask.ndim == 3:
                    mask = mask.squeeze()

                # Resize mask to original image size if needed
                if mask.shape != (height, width):
                    mask = cv2.resize(mask.astype(np.float32), (width, height))

                # Convert to binary mask
                binary_mask = (mask > 0.5).astype(np.uint8) * 255

                # Calculate bounding box
                contours, _ = cv2.findContours(
                    binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                if not contours:
                    continue

                x, y, w, h = cv2.boundingRect(np.vstack(contours))

                # Calculate centroid
                moments = cv2.moments(binary_mask)
                if moments["m00"] > 0:
                    cx = int(moments["m10"] / moments["m00"])
                    cy = int(moments["m01"] / moments["m00"])
                else:
                    cx, cy = x + w // 2, y + h // 2

                # Area in pixels
                area = int(np.sum(binary_mask > 0))

                # Skip very small masks (noise)
                if area < 100:
                    continue

                # Encode mask as PNG
                mask_png = cv2.imencode(".png", binary_mask)[1].tobytes()
                mask_base64 = base64.b64encode(mask_png).decode("utf-8")

                # Get confidence from results if available
                if results[0].boxes is not None and i < len(results[0].boxes.conf):
                    confidence = float(results[0].boxes.conf[i])
                else:
                    # Use area ratio as proxy confidence
                    confidence = min(1.0, area / (width * height * 0.3))

                mask_results.append({
                    "mask_base64": mask_base64,
                    "bbox": (x, y, w, h),
                    "confidence": confidence,
                    "area_pixels": area,
                    "centroid": (cx, cy),
                })

        # Sort by area (largest first)
        mask_results.sort(key=lambda m: m["area_pixels"], reverse=True)

        processing_time_ms = int((time.time() - start_time) * 1000)

        result = {
            "masks": mask_results,
            "processing_time_ms": processing_time_ms,
            "image_width": width,
            "image_height": height,
            "model_version": self._model_version,
            "visualization_base64": None,
        }

        # Generate visualization if requested
        if return_visualization and mask_results:
            result["visualization_base64"] = self._create_visualization(
                image_np, mask_results
            )

        logger.info(
            f"Segmentation complete: {len(mask_results)} masks in {processing_time_ms}ms"
        )

        return result

    def _create_visualization(
        self, image: np.ndarray, masks: list[dict]
    ) -> str:
        """Create a visualization image with masks overlaid."""
        vis_image = image.copy()

        # Color palette for masks
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (255, 128, 0),  # Orange
            (128, 0, 255),  # Purple
        ]

        for i, mask_data in enumerate(masks):
            color = colors[i % len(colors)]

            # Decode mask
            mask_bytes = base64.b64decode(mask_data["mask_base64"])
            mask = cv2.imdecode(
                np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_GRAYSCALE
            )

            # Create colored overlay
            overlay = np.zeros_like(vis_image)
            overlay[mask > 0] = color

            # Blend with original
            vis_image = cv2.addWeighted(vis_image, 1.0, overlay, 0.4, 0)

            # Draw bounding box
            x, y, w, h = mask_data["bbox"]
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)

            # Draw centroid
            cx, cy = mask_data["centroid"]
            cv2.circle(vis_image, (cx, cy), 5, color, -1)

        # Encode as PNG
        _, buffer = cv2.imencode(".png", cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        return base64.b64encode(buffer.tobytes()).decode("utf-8")

    def get_gpu_info(self) -> dict:
        """Get GPU information."""
        if not torch.cuda.is_available():
            return {
                "gpu_available": False,
                "gpu_name": None,
                "gpu_memory_used_mb": 0.0,
            }

        return {
            "gpu_available": True,
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_used_mb": torch.cuda.memory_allocated(0) / 1024 / 1024,
        }


# Global segmenter instance (loaded once at startup)
segmenter = FastSAMSegmenter()
