# Multi-stage build for optimized container size

# 1. Base image with Python and dependencies
FROM python:3.9-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Args for build-time configuration
ARG SERVICE=api
ENV SERVICE_TYPE=${SERVICE}

# 2. API Service
FROM base as api
COPY . /app/
RUN mkdir -p /app/logs/api

# Set API-specific environment variables
ENV PYTHONPATH=/app
ENV PORT=8000
ENV MODEL_PATH=/app/models/final/model.onnx

# Expose API port
EXPOSE 8000

# Set health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Command to run the API
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

# 3. Training Service
FROM nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04 as cuda-base
ARG DEBIAN_FRONTEND=noninteractive

# Install Python and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.9 \
    python3-pip \
    python3.9-dev \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for Python
RUN ln -sf /usr/bin/python3.9 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Create working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . /app/

# Create required directories
RUN mkdir -p /app/logs/training /app/models/checkpoints /app/models/final

# Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

# Command to run the training script
CMD ["python", "-m", "src.train"]

# 4. Final image based on service type
FROM ${SERVICE_TYPE} as final