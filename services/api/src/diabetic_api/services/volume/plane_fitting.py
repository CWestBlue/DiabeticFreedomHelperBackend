"""RANSAC plane fitting for table surface detection.

MVP-2.5: Uses ring region around food mask to fit table plane.
"""

import logging
import numpy as np
from typing import Tuple

from .models import PlaneFitResult, VolumeError, VolumeErrorCode

logger = logging.getLogger(__name__)


def fit_plane_ransac(
    points: np.ndarray,
    max_iterations: int = 1000,
    distance_threshold: float = 5.0,  # mm
    min_inliers_ratio: float = 0.3,
) -> PlaneFitResult:
    """
    Fit a plane to 3D points using RANSAC.

    Args:
        points: (N, 3) array of 3D points in mm
        max_iterations: Maximum RANSAC iterations
        distance_threshold: Inlier distance threshold in mm
        min_inliers_ratio: Minimum ratio of inliers required

    Returns:
        PlaneFitResult with plane coefficients and quality metrics

    Raises:
        VolumeError: If plane fitting fails
    """
    if len(points) < 3:
        raise VolumeError(
            code=VolumeErrorCode.INSUFFICIENT_TABLE_PIXELS,
            message="Need at least 3 points to fit a plane",
            details={"point_count": len(points)},
        )

    n_points = len(points)
    best_inliers = None
    best_inlier_count = 0
    best_plane = None

    # RANSAC iterations
    for _ in range(max_iterations):
        # Randomly sample 3 points
        indices = np.random.choice(n_points, 3, replace=False)
        p1, p2, p3 = points[indices]

        # Compute plane normal via cross product
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)

        # Skip degenerate cases (collinear points)
        norm_length = np.linalg.norm(normal)
        if norm_length < 1e-8:
            continue

        # Normalize
        normal = normal / norm_length
        a, b, c = normal

        # Plane equation: ax + by + cz + d = 0
        # d = -(a*x0 + b*y0 + c*z0) for point on plane
        d = -np.dot(normal, p1)

        # Calculate distances to plane for all points
        distances = np.abs(points @ normal + d)

        # Count inliers
        inliers = distances < distance_threshold
        inlier_count = np.sum(inliers)

        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_inliers = inliers
            best_plane = (a, b, c, d)

    if best_plane is None or best_inlier_count < 3:
        raise VolumeError(
            code=VolumeErrorCode.PLANE_FIT_FAILED,
            message="Could not fit plane to points",
            details={"point_count": n_points, "best_inlier_count": best_inlier_count},
        )

    # Refine plane using all inliers (least squares)
    inlier_points = points[best_inliers]
    a, b, c, d = _refine_plane_least_squares(inlier_points)

    # Ensure normal points "up" (positive Y or Z depending on camera orientation)
    # For typical food scanning, we want normal pointing towards camera (positive Z)
    if c < 0:
        a, b, c, d = -a, -b, -c, -d

    # Calculate RMSE
    final_distances = np.abs(inlier_points @ np.array([a, b, c]) + d)
    rmse = np.sqrt(np.mean(final_distances ** 2))

    inlier_ratio = best_inlier_count / n_points

    logger.debug(
        f"Plane fit: {best_inlier_count}/{n_points} inliers ({inlier_ratio:.1%}), RMSE={rmse:.2f}mm"
    )

    return PlaneFitResult(
        a=float(a),
        b=float(b),
        c=float(c),
        d=float(d),
        inlier_count=int(best_inlier_count),
        total_points=n_points,
        rmse=float(rmse),
        inlier_ratio=float(inlier_ratio),
    )


def _refine_plane_least_squares(points: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Refine plane fit using least squares on inlier points.

    Uses SVD to find the best-fit plane.

    Args:
        points: (N, 3) array of inlier points

    Returns:
        Tuple of (a, b, c, d) plane coefficients
    """
    # Center the points
    centroid = np.mean(points, axis=0)
    centered = points - centroid

    # SVD to find the plane normal (smallest singular value)
    _, _, vh = np.linalg.svd(centered)
    normal = vh[-1]  # Normal is the last row of V^T

    # Normalize
    normal = normal / np.linalg.norm(normal)
    a, b, c = normal

    # d from centroid
    d = -np.dot(normal, centroid)

    return a, b, c, d


def extract_ring_region_points(
    depth_map: np.ndarray,
    mask: np.ndarray,
    intrinsics: "CameraIntrinsics",
    ring_width_pixels: int = 30,
    min_depth_mm: float = 100.0,
    max_depth_mm: float = 2000.0,
) -> np.ndarray:
    """
    Extract 3D points from the ring region around a food mask (table surface).

    Args:
        depth_map: (H, W) uint16 depth map in mm
        mask: (H, W) binary mask of food region
        intrinsics: Camera intrinsic parameters
        ring_width_pixels: Width of ring region around mask
        min_depth_mm: Minimum valid depth
        max_depth_mm: Maximum valid depth

    Returns:
        (N, 3) array of 3D points from the ring region
    """
    import cv2
    from .models import CameraIntrinsics as CI

    # Ensure mask is binary
    binary_mask = (mask > 0).astype(np.uint8)

    # Dilate mask to create outer boundary
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (ring_width_pixels * 2 + 1, ring_width_pixels * 2 + 1)
    )
    dilated = cv2.dilate(binary_mask, kernel, iterations=1)

    # Ring region = dilated - original mask
    ring_mask = dilated.astype(bool) & ~binary_mask.astype(bool)

    # Get coordinates where ring is true
    v_coords, u_coords = np.where(ring_mask)

    if len(v_coords) == 0:
        return np.array([]).reshape(0, 3)

    # Get depth values
    depths = depth_map[v_coords, u_coords].astype(np.float64)

    # Filter valid depths
    valid = (depths > min_depth_mm) & (depths < max_depth_mm)
    u_valid = u_coords[valid]
    v_valid = v_coords[valid]
    depths_valid = depths[valid]

    if len(depths_valid) == 0:
        return np.array([]).reshape(0, 3)

    # Project to 3D
    points_3d = intrinsics.pixel_to_3d(
        u_valid.astype(np.float64),
        v_valid.astype(np.float64),
        depths_valid,
    )

    return points_3d


def get_table_plane_from_masks(
    depth_map: np.ndarray,
    masks: list[np.ndarray],
    intrinsics: "CameraIntrinsics",
    ring_width_pixels: int = 30,
    min_table_points: int = 100,
) -> PlaneFitResult:
    """
    Fit a table plane using ring regions from all food masks.

    Args:
        depth_map: (H, W) uint16 depth map in mm
        masks: List of binary masks for food regions
        intrinsics: Camera intrinsic parameters
        ring_width_pixels: Width of ring region around each mask
        min_table_points: Minimum points required for plane fitting

    Returns:
        PlaneFitResult for the table surface

    Raises:
        VolumeError: If insufficient table points or plane fit fails
    """
    from .models import CameraIntrinsics

    all_points = []

    for mask in masks:
        points = extract_ring_region_points(
            depth_map=depth_map,
            mask=mask,
            intrinsics=intrinsics,
            ring_width_pixels=ring_width_pixels,
        )
        if len(points) > 0:
            all_points.append(points)

    if not all_points:
        raise VolumeError(
            code=VolumeErrorCode.INSUFFICIENT_TABLE_PIXELS,
            message="No valid table surface points found around food masks",
            details={"mask_count": len(masks)},
        )

    combined_points = np.vstack(all_points)

    if len(combined_points) < min_table_points:
        raise VolumeError(
            code=VolumeErrorCode.INSUFFICIENT_TABLE_PIXELS,
            message=f"Insufficient table pixels: {len(combined_points)} < {min_table_points}",
            details={
                "point_count": len(combined_points),
                "min_required": min_table_points,
            },
        )

    logger.info(f"Fitting plane to {len(combined_points)} table surface points")
    return fit_plane_ransac(combined_points)
