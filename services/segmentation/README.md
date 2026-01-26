# Food Segmentation Service

FastSAM-based image segmentation service for identifying food regions in images.

## Overview

This microservice provides food image segmentation using [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM) (Fast Segment Anything Model). It exposes a simple REST API for segmenting images and returning pixel-level masks.

## Features

- **FastSAM Integration**: Uses FastSAM-s (small) or FastSAM-x (large) models
- **GPU Acceleration**: CUDA support for NVIDIA GPUs (tested on RTX 3070)
- **Memory Management**: Configurable GPU memory limits
- **REST API**: Simple `/segment` endpoint for image segmentation
- **Health Checks**: `/health` endpoint for monitoring

## API Endpoints

### `POST /segment`

Segment an image to identify distinct regions.

**Request:**

```json
{
  "image_base64": "<base64-encoded PNG or JPEG>",
  "prompt": "food on plate",
  "return_visualization": false
}
```

**Response:**

```json
{
  "masks": [
    {
      "mask_base64": "<base64-encoded PNG mask>",
      "bbox": [x, y, width, height],
      "confidence": 0.95,
      "area_pixels": 12345,
      "centroid": [cx, cy]
    }
  ],
  "processing_time_ms": 150,
  "image_width": 1920,
  "image_height": 1080,
  "model_version": "FastSAM-s"
}
```

### `GET /health`

Check service health and model status.

**Response:**

```json
{
  "status": "ok",
  "model_loaded": true,
  "gpu_available": true,
  "gpu_name": "NVIDIA GeForce RTX 3070",
  "gpu_memory_used_mb": 1234.5,
  "model_version": "FastSAM-s"
}
```

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `SEGMENTATION_HOST` | `0.0.0.0` | Server host |
| `SEGMENTATION_PORT` | `8001` | Server port |
| `SEGMENTATION_DEVICE` | `cuda` | Device (`cuda` or `cpu`) |
| `SEGMENTATION_MODEL_NAME` | `FastSAM-s` | Model variant |
| `SEGMENTATION_MODEL_PATH` | `/app/models` | Model storage directory |
| `SEGMENTATION_GPU_MEMORY_LIMIT_GB` | `2.0` | Max GPU memory (GB) |
| `SEGMENTATION_CONFIDENCE_THRESHOLD` | `0.5` | Min mask confidence |
| `SEGMENTATION_IOU_THRESHOLD` | `0.7` | IoU threshold for NMS |
| `SEGMENTATION_MAX_MASKS` | `10` | Max masks to return |
| `SEGMENTATION_LOG_LEVEL` | `INFO` | Logging level |

## Deployment

### Docker (Recommended)

```bash
# Build the image
docker build -t segmentation-service .

# Run with GPU support
docker run -d \
  --gpus all \
  -p 8001:8001 \
  -v ./models:/app/models \
  --name segmentation-service \
  segmentation-service
```

### Docker Compose

See the `docker-compose.truenas.yml` file in the `backend/` root directory.

### Local Development (CPU)

```bash
# Install dependencies
pip install -e ".[cpu,dev]"

# Run the service
SEGMENTATION_DEVICE=cpu python -m uvicorn segmentation_api.main:app --reload
```

## GPU Requirements

- NVIDIA GPU with CUDA support
- NVIDIA Container Toolkit (for Docker)
- Minimum 2GB GPU memory
- Tested on RTX 3070 (8GB)

## Integration

This service is called by the main Diabetic AI Chat API during food scanning:

```
User uploads food image
        │
        ▼
┌───────────────────┐
│  Main API         │
│  /food/scan       │
└────────┬──────────┘
         │ HTTP POST /segment
         ▼
┌───────────────────┐
│  Segmentation Svc │ ◄── This service
│  FastSAM          │
└────────┬──────────┘
         │ Masks
         ▼
┌───────────────────┐
│  Volume Compute   │
│  (MVP-2.5)        │
└───────────────────┘
```

## License

Internal use only.
