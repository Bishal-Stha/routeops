# RouteOps - Multilingual Semantic Router & MLOps Gateway
# Containerized deployment Dockerfile supporting CPU/GPU inference,
# local Docker, Render, Railway, and Hugging Face Spaces (Port 7860/8000)

FROM python:3.11-slim AS builder

WORKDIR /app

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Final production image
FROM python:3.11-slim

WORKDIR /app

# Install runtime utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed site-packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create non-root user for Hugging Face Spaces compatibility (uid 1000)
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Copy project files
COPY --chown=user:user . /app

USER user

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

EXPOSE 7860
EXPOSE 8000

# Health check probe using dynamic $PORT fallback
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT:-7860}/health || exit 1

# Start FastAPI gateway via uvicorn supporting dynamic PORT
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-7860}"]
