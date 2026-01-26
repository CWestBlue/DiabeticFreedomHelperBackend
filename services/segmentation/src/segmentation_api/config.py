"""Configuration settings for the segmentation service."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8001

    # Model settings
    model_name: str = "FastSAM-s"  # Options: FastSAM-s (small), FastSAM-x (large)
    model_path: str = "/app/models"  # Directory to store downloaded models
    device: str = "cuda"  # "cuda" or "cpu"

    # GPU memory settings
    gpu_memory_limit_gb: float = 2.0  # Max GPU memory to use

    # Inference settings
    confidence_threshold: float = 0.5  # Minimum confidence for masks
    iou_threshold: float = 0.7  # IoU threshold for NMS
    max_masks: int = 10  # Maximum number of masks to return

    # Logging
    log_level: str = "INFO"

    class Config:
        env_prefix = "SEGMENTATION_"
        env_file = ".env"


settings = Settings()
