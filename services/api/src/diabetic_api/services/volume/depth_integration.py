"""Depth integration for volume computation.

MVP-2.5: Integrates depth values above the fitted table plane
to compute food volume in milliliters.
"""

import logging
import numpy as np
from typing import Tuple

from .models import (
    CameraIntrinsics,
    PlaneFitResult,
    VolumeError,
    VolumeErrorCode,
)

logger = logging.getLogger(__name__)


def compute_volume_from_mask(
    depth_map: np.ndarray,
    mask: np.ndarray,
    plane: PlaneFitResult,
    intrinsics: CameraIntrinsics,
    min_depth_mm: float = 100.0,
    max_depth_mm: float = 2000.0,
    min_height_mm: float = 1.0,  # Ignore heights below this (noise)
    max_height_mm: float = 200.0,  # Ignore unrealistic heights
) -> Tuple[float, dict]:
    """
    Compute volume of food region above the table plane.

    Uses depth integration: for each pixel in the mask, compute its height
    above the fitted plane and integrate to get total volume.

    Args:
        depth_map: (H, W) uint16 depth map in mm
        mask: (H, W) binary mask of food region
        plane: Fitted table plane
        intrinsics: Camera intrinsic parameters
        min_depth_mm: Minimum valid depth value
        max_depth_mm: Maximum valid depth value
        min_height_mm: Minimum height above plane to include
        max_height_mm: Maximum height above plane to include

    Returns:
        Tuple of (volume_ml, stats_dict)

    Raises:
        VolumeError: If depth data is insufficient
    """
    # Get mask pixel coordinates
    binary_mask = mask > 0
    v_coords, u_coords = np.where(binary_mask)

    total_mask_pixels = len(v_coords)
    if total_mask_pixels == 0:
        raise VolumeError(
            code=VolumeErrorCode.NO_MASK_PROVIDED,
            message="Empty mask provided",
        )

    # Get depth values for mask pixels
    depths = depth_map[v_coords, u_coords].astype(np.float64)

    # Filter valid depths
    valid_depth_mask = (depths > min_depth_mm) & (depths < max_depth_mm)
    valid_count = np.sum(valid_depth_mask)
    valid_ratio = valid_count / total_mask_pixels

    if valid_ratio < 0.1:  # Less than 10% valid depths
        raise VolumeError(
            code=VolumeErrorCode.DEPTH_TOO_SPARSE,
            message=f"Insufficient valid depth pixels: {valid_ratio:.1%}",
            details={
                "valid_pixels": int(valid_count),
                "total_pixels": total_mask_pixels,
                "valid_ratio": float(valid_ratio),
            },
        )

    # Use only valid pixels
    u_valid = u_coords[valid_depth_mask].astype(np.float64)
    v_valid = v_coords[valid_depth_mask].astype(np.float64)
    depths_valid = depths[valid_depth_mask]

    # Project to 3D points
    points_3d = intrinsics.pixel_to_3d(u_valid, v_valid, depths_valid)

    # Calculate height above plane (signed distance)
    # Positive distance means point is above the plane (towards camera)
    heights = -plane.distance_to_plane(points_3d)

    # Filter heights to reasonable range
    height_mask = (heights > min_height_mm) & (heights < max_height_mm)
    valid_heights = heights[height_mask]
    valid_depths_for_area = depths_valid[height_mask]

    if len(valid_heights) == 0:
        # No points above plane - food might be flat or plane fit is off
        logger.warning("No points above plane detected")
        return 0.0, {
            "valid_depth_ratio": float(valid_ratio),
            "mask_area_pixels": total_mask_pixels,
            "mean_height_mm": 0.0,
            "max_height_mm": 0.0,
            "points_above_plane": 0,
        }

    # Compute volume by integrating height × pixel area
    # Each pixel covers a different area depending on its depth
    total_volume_mm3 = 0.0

    for height, depth in zip(valid_heights, valid_depths_for_area):
        pixel_area_mm2 = intrinsics.get_pixel_area_mm2(depth)
        total_volume_mm3 += height * pixel_area_mm2

    # Convert mm³ to mL (1 mL = 1000 mm³)
    volume_ml = total_volume_mm3 / 1000.0

    # Compute statistics
    mean_height = float(np.mean(valid_heights))
    max_height = float(np.max(valid_heights))

    stats = {
        "valid_depth_ratio": float(valid_ratio),
        "mask_area_pixels": total_mask_pixels,
        "mean_height_mm": mean_height,
        "max_height_mm": max_height,
        "points_above_plane": len(valid_heights),
        "points_below_plane": int(np.sum(heights <= min_height_mm)),
    }

    logger.debug(
        f"Volume computed: {volume_ml:.1f}mL from {len(valid_heights)} points, "
        f"mean height={mean_height:.1f}mm, max height={max_height:.1f}mm"
    )

    return volume_ml, stats


def compute_quality_score(
    plane: PlaneFitResult,
    depth_valid_ratio: float,
    points_above_plane: int,
    total_mask_pixels: int,
) -> float:
    """
    Compute a quality score for the volume estimation.

    Factors:
    - Plane fit quality (inlier ratio, RMSE)
    - Depth data coverage within mask
    - Number of valid height measurements

    Args:
        plane: Fitted plane result
        depth_valid_ratio: Ratio of valid depth pixels in mask
        points_above_plane: Number of points with valid height above plane
        total_mask_pixels: Total pixels in the mask

    Returns:
        Quality score between 0 and 1
    """
    # Plane fit quality (0-1)
    # Good: inlier ratio > 0.7, RMSE < 5mm
    plane_score = min(1.0, plane.inlier_ratio / 0.7) * min(1.0, 5.0 / max(plane.rmse, 0.1))
    plane_score = max(0.0, min(1.0, plane_score))

    # Depth coverage score (0-1)
    # Good: > 50% valid depth pixels
    depth_score = min(1.0, depth_valid_ratio / 0.5)

    # Height measurement coverage (0-1)
    # Good: > 30% of mask pixels have valid height above plane
    height_ratio = points_above_plane / max(total_mask_pixels, 1)
    height_score = min(1.0, height_ratio / 0.3)

    # Weighted combination
    quality = 0.4 * plane_score + 0.3 * depth_score + 0.3 * height_score

    return max(0.0, min(1.0, quality))


def estimate_grams_from_volume(
    volume_ml: float,
    food_density: float = 1.0,  # Default: assume water density (1 g/mL)
) -> float:
    """
    Estimate weight in grams from volume.

    Args:
        volume_ml: Volume in milliliters
        food_density: Food density in g/mL (default 1.0 for water-like foods)

    Returns:
        Estimated weight in grams
    """
    return volume_ml * food_density
