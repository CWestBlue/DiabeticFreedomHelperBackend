"""Volume computation service for food scan.

MVP-2.5: Computes food volume in mL from depth maps and segmentation masks
using RANSAC plane fitting on table surface pixels.
"""

from .service import VolumeComputationService, get_volume_service
from .models import VolumeComputationResult, PlaneFitResult, VolumeError

__all__ = [
    "VolumeComputationService",
    "get_volume_service",
    "VolumeComputationResult",
    "PlaneFitResult",
    "VolumeError",
]
