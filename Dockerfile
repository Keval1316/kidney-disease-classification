# =============================================================================
# Production Dockerfile for CT Kidney Disease Classification FastAPI Service
# =============================================================================
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TF_ENABLE_ONEDNN_OPTS=0 \
    PORT=8000

# Install minimal OS dependencies for healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source, frontend assets, configuration, and trained model artifact
COPY api/ ./api/
COPY app/ ./app/
COPY src/ ./src/
COPY config/ ./config/
COPY artifacts/model/best_model.keras ./artifacts/model/best_model.keras

# Expose default port
EXPOSE 8000

# Health check to ensure API is responsive
HEALTHCHECK --interval=30s --timeout=10s --start-period=35s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Launch uvicorn dynamically honoring the $PORT environment variable (required by Render)
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
