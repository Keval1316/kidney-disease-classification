# =============================================================================
# Production Dockerfile for Hugging Face Spaces, Render & Cloud Deployment
# =============================================================================
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TF_ENABLE_ONEDNN_OPTS=0 \
    PORT=7860

# Install minimal OS dependencies for healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user with UID 1000 (standard for Hugging Face Spaces)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set working directory
WORKDIR $HOME/app

# Install Python dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source, frontend assets, configuration, and trained model artifact
COPY --chown=user api/ ./api/
COPY --chown=user app/ ./app/
COPY --chown=user src/ ./src/
COPY --chown=user config/ ./config/
COPY --chown=user artifacts/model/best_model.keras ./artifacts/model/best_model.keras

# Expose default port (7860 for Hugging Face Spaces)
EXPOSE 7860

# Health check to ensure API is responsive
HEALTHCHECK --interval=30s --timeout=10s --start-period=35s --retries=3 \
    CMD curl -f http://localhost:${PORT:-7860}/health || exit 1

# Launch uvicorn dynamically honoring the $PORT environment variable (Render / HF)
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
