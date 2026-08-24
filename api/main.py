"""
api/main.py
-----------
FastAPI production inference service for CT Kidney Disease Classification.

Endpoints:
    - GET  /         : Service metadata and medical disclaimer
    - GET  /health   : Health status check
    - POST /predict  : Classify an uploaded kidney CT image

Classes:
    - Cyst
    - Normal
    - Stone
    - Tumor
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.inference.predict import CLASS_NAMES, load_model_cached, predict_image
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Directory paths
BASE_DIR = Path(__file__).resolve().parent.parent
APP_DIR = BASE_DIR / "app"

# ---------------------------------------------------------------------------
# Constants & Configuration
# ---------------------------------------------------------------------------
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/bmp",
    "image/tiff",
    "application/octet-stream",  # some clients send raw binary
}
MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

DISCLAIMER = (
    "This model is for educational and research demonstration only. "
    "It is not a medical diagnostic device and must not be used for clinical decision-making."
)


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str = Field(..., examples=["healthy"])
    model_loaded: bool = Field(..., examples=[True])
    supported_classes: list[str] = Field(..., examples=[CLASS_NAMES])


class PredictionResponse(BaseModel):
    prediction: str = Field(..., examples=["Tumor"], description="Predicted class name")
    confidence: float = Field(
        ..., examples=[0.9421], description="Confidence score between 0.0 and 1.0"
    )
    probabilities: Dict[str, float] = Field(
        ...,
        examples=[{"Cyst": 0.02, "Normal": 0.01, "Stone": 0.02, "Tumor": 0.94}],
        description="Softmax probability distribution over all classes",
    )
    disclaimer: str = Field(default=DISCLAIMER)


class RootResponse(BaseModel):
    message: str
    version: str
    docs_url: str
    health_url: str
    ui_url: str = "/ui"
    disclaimer: str


# ---------------------------------------------------------------------------
# Application Lifespan (Startup / Shutdown)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up and load the model on service startup."""
    logger.info("Initializing Kidney Disease Classification API service...")
    try:
        load_model_cached()
        logger.info("Deep learning model loaded into memory and ready for inference.")
    except FileNotFoundError as exc:
        logger.warning(
            "Model file not found during startup: %s. Predictions will fail until model is provided.",
            exc,
        )
    except Exception as exc:
        logger.error("Unexpected error while loading model during startup: %s", exc)
    yield
    logger.info("Shutting down Kidney Disease Classification API service.")


# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="CT Kidney Disease Classification API",
    description=(
        "Production-grade Deep Learning Inference Service using EfficientNetB0 "
        "to classify Kidney CT scans into Normal, Cyst, Tumor, or Stone.\n\n"
        f"**Medical Disclaimer:** {DISCLAIMER}"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for frontend clients (HTML/JS, React, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and UI if app directory exists
if APP_DIR.exists():
    if (APP_DIR / "css").exists():
        app.mount("/css", StaticFiles(directory=str(APP_DIR / "css")), name="css")
    if (APP_DIR / "js").exists():
        app.mount("/js", StaticFiles(directory=str(APP_DIR / "js")), name="js")
    if (APP_DIR / "samples").exists():
        app.mount("/samples", StaticFiles(directory=str(APP_DIR / "samples")), name="samples")
    app.mount("/app", StaticFiles(directory=str(APP_DIR), html=True), name="app")


# ---------------------------------------------------------------------------
# Route: Web UI
# ---------------------------------------------------------------------------
@app.get("/ui", response_class=FileResponse, tags=["Frontend"], summary="Interactive Diagnostic Web UI")
async def serve_ui():
    """Serve the modern HTML5/CSS3/JavaScript web interface."""
    index_file = APP_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Web UI assets not found. Please ensure app/index.html exists.",
        )
    return FileResponse(index_file)


# ---------------------------------------------------------------------------
# Route: Root
# ---------------------------------------------------------------------------
@app.get("/", response_model=RootResponse, tags=["General"])
async def root():
    """Service metadata, documentation links, and medical disclaimer."""
    return RootResponse(
        message="CT Kidney Disease Classification API is running.",
        version="1.0.0",
        docs_url="/docs",
        health_url="/health",
        ui_url="/ui",
        disclaimer=DISCLAIMER,
    )


# ---------------------------------------------------------------------------
# Route: Health
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """
    Check the health status of the API and verify whether the model weights are loaded.
    """
    model_loaded = False
    try:
        load_model_cached()
        model_loaded = True
    except Exception:
        model_loaded = False

    return HealthResponse(
        status="healthy",
        model_loaded=model_loaded,
        supported_classes=CLASS_NAMES,
    )


# ---------------------------------------------------------------------------
# Route: Predict
# ---------------------------------------------------------------------------
@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"],
    summary="Classify a Kidney CT Image",
)
async def predict(
    file: UploadFile = File(..., description="Uploaded kidney CT image (JPG, PNG, etc.)")
):
    """
    Accepts an uploaded image file and returns:
      - Predicted disease class (Normal, Cyst, Tumor, Stone)
      - Confidence score (0.0 to 1.0)
      - Softmax probability distribution across all 4 classes
    """
    # 1. Check filename and extension
    if not file.filename:
        logger.warning("Predict request rejected: empty filename.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file provided or invalid filename.",
        )

    _, ext = os.path.splitext(file.filename.lower())
    if ext not in ALLOWED_EXTENSIONS:
        logger.warning("Predict request rejected: unsupported extension '%s'.", ext)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Allowed formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    # 2. Read image content into bytes
    try:
        contents = await file.read()
    except Exception as exc:
        logger.error("Failed to read uploaded file: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not read upload data: {str(exc)}",
        )

    if not contents or len(contents) == 0:
        logger.warning("Predict request rejected: empty file payload.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded image file is empty.",
        )

    if len(contents) > MAX_FILE_SIZE_BYTES:
        logger.warning(
            "Predict request rejected: file size %d exceeds limit %d bytes.",
            len(contents),
            MAX_FILE_SIZE_BYTES,
        )
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of {MAX_FILE_SIZE_MB}MB.",
        )

    # 3. Perform prediction
    try:
        result = predict_image(image_bytes=contents)
        return PredictionResponse(
            prediction=result["prediction"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
            disclaimer=DISCLAIMER,
        )
    except FileNotFoundError as exc:
        logger.error("Model unavailable: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is currently unavailable on the server. Please contact support.",
        )
    except ValueError as exc:
        logger.warning("Image processing error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Corrupted or unreadable image file: {str(exc)}",
        )
    except Exception as exc:
        logger.exception("Unexpected error during inference: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal server error occurred while processing the image.",
        )


# ---------------------------------------------------------------------------
# CLI / Local Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    logger.info("Starting uvicorn server on %s:%d ...", host, port)
    uvicorn.run("api.main:app", host=host, port=port, reload=True)
