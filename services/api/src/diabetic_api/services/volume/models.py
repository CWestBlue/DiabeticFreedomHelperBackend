"""Data models for volume computation service.

MVP-2.5: Volume Computation
"""

from dataclasses import dataclass
from enum import Enum
from typing import Annotated

import numpy as np
from pydantic import BaseModel, Field


class VolumeErrorCode(str, Enum):
    """Error codes for volume computation failures."""

    INSUFFICIENT_TABLE_PIXELS = "INSUFFICIENT_TABLE_PIXELS"
    PLANE_FIT_FAILED = "PLANE_FIT_FAILED"
    DEPTH_TOO_SPARSE = "DEPTH_TOO_SPARSE"
    INVALID_DEPTH_DATA = "INVALID_DEPTH_DATA"
    NO_MASK_PROVIDED = "NO_MASK_PROVIDED"
    COMPUTATION_ERROR = "COMPUTATION_ERROR"


class VolumeError(Exception):
    """Exception raised when volume computation fails."""

    def __init__(self, code: VolumeErrorCode, message: str, details: dict | None = None) -> None:
        self.code = code
        self.message = message
        self.details = details or {}
        super().__init__(message)


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters for 3D projection."""

    fx: float  # Focal length X (pixels)
    fy: float  # Focal length Y (pixels)
    cx: float  # Principal point X (pixels)
    cy: float  # Principal point Y (pixels)
    width: int  # Image width (pixels)
    height: int  # Image height (pixels)

    def pixel_to_3d(self, u: np.ndarray, v: np.ndarray, depth_mm: np.ndarray) -> np.ndarray:
        """
        Project 2D pixel coordinates to 3D points.

        Args:
            u: X pixel coordinates (N,)
            v: Y pixel coordinates (N,)
            depth_mm: Depth values in millimeters (N,)

        Returns:
            3D points array of shape (N, 3) in mm [X, Y, Z]
        """
        # Standard pinhole camera projection
        x = (u - self.cx) * depth_mm / self.fx
        y = (v - self.cy) * depth_mm / self.fy
        z = depth_mm
        return np.stack([x, y, z], axis=-1)

    def get_pixel_area_mm2(self, depth_mm: float) -> float:
        """
        Calculate the area of a pixel in mm² at a given depth.

        Args:
            depth_mm: Depth in millimeters

        Returns:
            Pixel area in mm²
        """
        # At depth d, one pixel covers (d/fx) × (d/fy) mm²
        return (depth_mm / self.fx) * (depth_mm / self.fy)


@dataclass
class PlaneFitResult:
    """Result of RANSAC plane fitting."""

    # Plane equation: ax + by + cz + d = 0, where (a,b,c) is unit normal
    a: float
    b: float
    c: float
    d: float

    # Quality metrics
    inlier_count: int
    total_points: int
    rmse: float  # Root mean square error of inliers
    inlier_ratio: float

    @property
    def normal(self) -> np.ndarray:
        """Get plane normal vector."""
        return np.array([self.a, self.b, self.c])

    def distance_to_plane(self, points: np.ndarray) -> np.ndarray:
        """
        Calculate signed distance from points to plane.

        Args:
            points: (N, 3) array of 3D points

        Returns:
            (N,) array of signed distances (positive = above plane)
        """
        # Distance = (ax + by + cz + d) / ||(a,b,c)||
        # Since (a,b,c) is unit normal, ||(a,b,c)|| = 1
        return points[:, 0] * self.a + points[:, 1] * self.b + points[:, 2] * self.c + self.d

    def is_valid(self, min_inlier_ratio: float = 0.5, max_rmse: float = 10.0) -> bool:
        """Check if plane fit meets quality thresholds."""
        return self.inlier_ratio >= min_inlier_ratio and self.rmse <= max_rmse


class VolumeComputationResult(BaseModel):
    """Result of volume computation for a single food region."""

    # Volume estimate
    volume_ml: float = Field(..., ge=0, description="Estimated volume in milliliters")
    quality_score: Annotated[float, Field(ge=0, le=1)] = Field(
        ..., description="Volume estimation quality (0-1)"
    )
    method: str = Field(
        default="depth_integration", description="Computation method used"
    )

    # Plane fit info
    plane_fit_inliers: int = Field(..., description="Number of RANSAC inliers")
    plane_fit_rmse: float = Field(..., description="Plane fit RMSE in mm")
    plane_inlier_ratio: float = Field(..., description="Ratio of inliers to total points")

    # Mask statistics
    mask_valid_depth_ratio: float = Field(
        ..., description="Ratio of valid depth pixels within mask"
    )
    mask_area_pixels: int = Field(..., description="Total pixels in mask")
    mean_height_mm: float = Field(..., description="Mean height above table in mm")
    max_height_mm: float = Field(..., description="Maximum height above table in mm")

    # Processing
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")


class AggregatedVolumeResult(BaseModel):
    """Aggregated volume computation result for all detected foods."""

    # Total volume across all masks
    total_volume_ml: float = Field(..., ge=0, description="Total volume in mL")
    overall_quality_score: Annotated[float, Field(ge=0, le=1)] = Field(
        ..., description="Overall quality score"
    )

    # Per-mask results
    mask_results: list[VolumeComputationResult] = Field(
        default_factory=list, description="Per-mask volume results"
    )

    # Plane fit (shared across all masks)
    plane_fit_inliers: int = Field(..., description="RANSAC plane fit inliers")
    plane_fit_rmse: float = Field(..., description="Plane fit RMSE in mm")

    # Processing
    total_processing_time_ms: int = Field(..., description="Total processing time")
