# Segmentation Service Dockerfile with CUDA support for FastSAM
# Requires NVIDIA Container Toolkit on the host

# Use NVIDIA CUDA base image with Python
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS base

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.10 and dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Set working directory
WORKDIR /app

# Create models directory
RUN mkdir -p /app/models

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir \
    torch==2.1.0+cu121 \
    torchvision==0.16.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Install FastSAM from GitHub
RUN pip install --no-cache-dir git+https://github.com/CASIA-IVA-Lab/FastSAM.git

# Install ultralytics (required by FastSAM)
RUN pip install --no-cache-dir ultralytics

# Copy project files
COPY pyproject.toml README.md ./
COPY src/ src/

# Install the application
RUN pip install --no-cache-dir -e .

# Download FastSAM model weights on build (optional - can also download at runtime)
# Uncomment to pre-download the model:
# RUN python -c "from ultralytics import YOLO; YOLO('FastSAM-s.pt')"

# Environment variables
ENV SEGMENTATION_HOST=0.0.0.0
ENV SEGMENTATION_PORT=8001
ENV SEGMENTATION_DEVICE=cuda
ENV SEGMENTATION_MODEL_NAME=FastSAM-s
ENV SEGMENTATION_MODEL_PATH=/app/models
ENV SEGMENTATION_GPU_MEMORY_LIMIT_GB=2.0
ENV SEGMENTATION_LOG_LEVEL=INFO

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/health')" || exit 1

# Run the service
CMD ["python", "-m", "uvicorn", "segmentation_api.main:app", "--host", "0.0.0.0", "--port", "8001"]
