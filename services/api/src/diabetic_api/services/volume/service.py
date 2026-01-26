"""Main volume computation service.

MVP-2.5: Orchestrates plane fitting and depth integration
to compute food volumes from scan data.
"""

import base64
import io
import logging
import time
from functools import lru_cache
from typing import Tuple

import cv2
import numpy as np

from diabetic_api.models.food_scan import SegmentationMask, SegmentationResult

from .models import (
    AggregatedVolumeResult,
    CameraIntrinsics,
    PlaneFitResult,
    VolumeComputationResult,
    VolumeError,
    VolumeErrorCode,
)
from .plane_fitting import fit_plane_ransac, get_table_plane_from_masks
from .depth_integration import compute_volume_from_mask, compute_quality_score

logger = logging.getLogger(__name__)


class VolumeComputationService:
    """Service for computing food volume from depth and segmentation data."""

    # Configuration
    RING_WIDTH_PIXELS = 30  # Width of table region around food mask
    MIN_TABLE_POINTS = 100  # Minimum points for plane fitting
    MIN_DEPTH_MM = 100.0  # Minimum valid depth (10cm)
    MAX_DEPTH_MM = 2000.0  # Maximum valid depth (2m)
    MIN_HEIGHT_MM = 2.0  # Minimum food height above table
    MAX_HEIGHT_MM = 150.0  # Maximum food height (15cm)

    def __init__(self) -> None:
        """Initialize the volume computation service."""
        logger.info("VolumeComputationService initialized")

    def compute_volume(
        self,
        depth_data: bytes,
        segmentation_result: SegmentationResult,
        intrinsics_dict: dict,
    ) -> AggregatedVolumeResult:
        """
        Compute volume for all detected food regions.

        Args:
            depth_data: Raw 16-bit PNG depth map bytes (in mm)
            segmentation_result: Result from segmentation service with masks
            intrinsics_dict: Camera intrinsics as dict (fx, fy, cx, cy, width, height)

        Returns:
            AggregatedVolumeResult with per-mask and total volumes

        Raises:
            VolumeError: If volume computation fails
        """
        start_time = time.time()

        # Parse intrinsics
        intrinsics = CameraIntrinsics(
            fx=intrinsics_dict["fx"],
            fy=intrinsics_dict["fy"],
            cx=intrinsics_dict["cx"],
            cy=intrinsics_dict["cy"],
            width=intrinsics_dict["width"],
            height=intrinsics_dict["height"],
        )

        # Decode depth map
        depth_map = self._decode_depth_map(depth_data)
        logger.debug(f"Depth map shape: {depth_map.shape}, dtype: {depth_map.dtype}")

        # Decode segmentation masks
        masks = self._decode_masks(segmentation_result.masks)

        if not masks:
            raise VolumeError(
                code=VolumeErrorCode.NO_MASK_PROVIDED,
                message="No segmentation masks provided",
            )

        # Fit table plane using ring regions around all masks
        try:
            plane = get_table_plane_from_masks(
                depth_map=depth_map,
                masks=masks,
                intrinsics=intrinsics,
                ring_width_pixels=self.RING_WIDTH_PIXELS,
                min_table_points=self.MIN_TABLE_POINTS,
            )
        except VolumeError:
            raise
        except Exception as e:
            logger.error(f"Plane fitting failed: {e}")
            raise VolumeError(
                code=VolumeErrorCode.PLANE_FIT_FAILED,
                message=f"Failed to fit table plane: {str(e)}",
            ) from e

        logger.info(
            f"Table plane fitted: {plane.inlier_count} inliers, "
            f"RMSE={plane.rmse:.2f}mm, ratio={plane.inlier_ratio:.1%}"
        )

        # Compute volume for each mask
        mask_results = []
        total_volume_ml = 0.0
        quality_scores = []

        for i, (mask, seg_mask) in enumerate(zip(masks, segmentation_result.masks)):
            mask_start = time.time()

            try:
                volume_ml, stats = compute_volume_from_mask(
                    depth_map=depth_map,
                    mask=mask,
                    plane=plane,
                    intrinsics=intrinsics,
                    min_depth_mm=self.MIN_DEPTH_MM,
                    max_depth_mm=self.MAX_DEPTH_MM,
                    min_height_mm=self.MIN_HEIGHT_MM,
                    max_height_mm=self.MAX_HEIGHT_MM,
                )

                quality = compute_quality_score(
                    plane=plane,
                    depth_valid_ratio=stats["valid_depth_ratio"],
                    points_above_plane=stats["points_above_plane"],
                    total_mask_pixels=stats["mask_area_pixels"],
                )

                mask_time_ms = int((time.time() - mask_start) * 1000)

                result = VolumeComputationResult(
                    volume_ml=volume_ml,
                    quality_score=quality,
                    method="depth_integration",
                    plane_fit_inliers=plane.inlier_count,
                    plane_fit_rmse=plane.rmse,
                    plane_inlier_ratio=plane.inlier_ratio,
                    mask_valid_depth_ratio=stats["valid_depth_ratio"],
                    mask_area_pixels=stats["mask_area_pixels"],
                    mean_height_mm=stats["mean_height_mm"],
                    max_height_mm=stats["max_height_mm"],
                    processing_time_ms=mask_time_ms,
                )

                mask_results.append(result)
                total_volume_ml += volume_ml
                quality_scores.append(quality)

                logger.info(
                    f"Mask {i}: {volume_ml:.1f}mL, quality={quality:.2f}, "
                    f"height={stats['mean_height_mm']:.1f}mm"
                )

            except VolumeError as e:
                logger.warning(f"Volume computation failed for mask {i}: {e.message}")
                # Add a zero-volume result for failed masks
                mask_results.append(
                    VolumeComputationResult(
                        volume_ml=0.0,
                        quality_score=0.0,
                        method="depth_integration",
                        plane_fit_inliers=plane.inlier_count,
                        plane_fit_rmse=plane.rmse,
                        plane_inlier_ratio=plane.inlier_ratio,
                        mask_valid_depth_ratio=0.0,
                        mask_area_pixels=seg_mask.area_pixels,
                        mean_height_mm=0.0,
                        max_height_mm=0.0,
                        processing_time_ms=0,
                    )
                )
                quality_scores.append(0.0)

        # Overall quality is weighted average by volume
        if total_volume_ml > 0:
            weights = [r.volume_ml / total_volume_ml for r in mask_results]
            overall_quality = sum(q * w for q, w in zip(quality_scores, weights))
        else:
            overall_quality = 0.0 if not quality_scores else sum(quality_scores) / len(quality_scores)

        total_time_ms = int((time.time() - start_time) * 1000)

        logger.info(
            f"Volume computation complete: {total_volume_ml:.1f}mL total, "
            f"quality={overall_quality:.2f}, time={total_time_ms}ms"
        )

        return AggregatedVolumeResult(
            total_volume_ml=total_volume_ml,
            overall_quality_score=overall_quality,
            mask_results=mask_results,
            plane_fit_inliers=plane.inlier_count,
            plane_fit_rmse=plane.rmse,
            total_processing_time_ms=total_time_ms,
        )

    def _decode_depth_map(self, depth_data: bytes) -> np.ndarray:
        """
        Decode 16-bit PNG depth map from bytes.

        Args:
            depth_data: Raw PNG bytes

        Returns:
            (H, W) uint16 numpy array with depth in mm

        Raises:
            VolumeError: If depth data is invalid
        """
        try:
            # Decode PNG
            nparr = np.frombuffer(depth_data, np.uint8)
            depth_map = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

            if depth_map is None:
                raise VolumeError(
                    code=VolumeErrorCode.INVALID_DEPTH_DATA,
                    message="Failed to decode depth map PNG",
                )

            # Ensure it's uint16
            if depth_map.dtype != np.uint16:
                logger.warning(f"Depth map dtype is {depth_map.dtype}, converting to uint16")
                depth_map = depth_map.astype(np.uint16)

            return depth_map

        except VolumeError:
            raise
        except Exception as e:
            raise VolumeError(
                code=VolumeErrorCode.INVALID_DEPTH_DATA,
                message=f"Failed to decode depth map: {str(e)}",
            ) from e

    def _decode_masks(self, seg_masks: list[SegmentationMask]) -> list[np.ndarray]:
        """
        Decode base64 mask images to numpy arrays.

        Args:
            seg_masks: List of SegmentationMask objects

        Returns:
            List of (H, W) binary mask arrays
        """
        masks = []

        for seg_mask in seg_masks:
            try:
                # Decode base64
                mask_bytes = base64.b64decode(seg_mask.mask_base64)
                nparr = np.frombuffer(mask_bytes, np.uint8)
                mask = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

                if mask is not None:
                    masks.append(mask)
                else:
                    logger.warning("Failed to decode mask, skipping")

            except Exception as e:
                logger.warning(f"Failed to decode mask: {e}")

        return masks

    def compute_single_mask_volume(
        self,
        depth_data: bytes,
        mask_data: bytes,
        intrinsics_dict: dict,
        plane: PlaneFitResult | None = None,
    ) -> Tuple[float, float, PlaneFitResult]:
        """
        Compute volume for a single mask (convenience method).

        If no plane is provided, fits one using the ring around the mask.

        Args:
            depth_data: Raw 16-bit PNG depth map bytes
            mask_data: Raw PNG mask bytes
            intrinsics_dict: Camera intrinsics dict
            plane: Optional pre-fitted plane

        Returns:
            Tuple of (volume_ml, quality_score, plane)
        """
        intrinsics = CameraIntrinsics(
            fx=intrinsics_dict["fx"],
            fy=intrinsics_dict["fy"],
            cx=intrinsics_dict["cx"],
            cy=intrinsics_dict["cy"],
            width=intrinsics_dict["width"],
            height=intrinsics_dict["height"],
        )

        depth_map = self._decode_depth_map(depth_data)

        # Decode mask
        nparr = np.frombuffer(mask_data, np.uint8)
        mask = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise VolumeError(
                code=VolumeErrorCode.NO_MASK_PROVIDED,
                message="Failed to decode mask",
            )

        # Fit plane if not provided
        if plane is None:
            plane = get_table_plane_from_masks(
                depth_map=depth_map,
                masks=[mask],
                intrinsics=intrinsics,
                ring_width_pixels=self.RING_WIDTH_PIXELS,
                min_table_points=self.MIN_TABLE_POINTS,
            )

        volume_ml, stats = compute_volume_from_mask(
            depth_map=depth_map,
            mask=mask,
            plane=plane,
            intrinsics=intrinsics,
        )

        quality = compute_quality_score(
            plane=plane,
            depth_valid_ratio=stats["valid_depth_ratio"],
            points_above_plane=stats["points_above_plane"],
            total_mask_pixels=stats["mask_area_pixels"],
        )

        return volume_ml, quality, plane


@lru_cache
def get_volume_service() -> VolumeComputationService:
    """Get a cached volume computation service instance."""
    return VolumeComputationService()
