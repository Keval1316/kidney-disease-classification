"""
tests/test_api.py
-----------------
Integration tests for the FastAPI inference endpoints using TestClient.
"""

import io
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.main import app


@pytest.fixture(scope="module")
def client():
    """FastAPI test client fixture with lifespan context."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def sample_jpeg_bytes():
    """Create a temporary synthetic JPEG file in memory."""
    img_data = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img_data).save(buf, format="JPEG")
    return buf.getvalue()


def test_root_endpoint(client):
    """GET / should return 200 and medical disclaimer."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "disclaimer" in data
    assert "docs_url" in data


def test_root_browser_request_serves_html(client):
    """GET / with Accept: text/html should return 200 with HTML UI."""
    response = client.get("/", headers={"accept": "text/html,application/xhtml+xml"})
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "NephroScan AI" in response.text


def test_health_endpoint(client):
    """GET /health should return 200 and healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data
    assert "supported_classes" in data
    assert len(data["supported_classes"]) == 4


def test_predict_valid_image(client, sample_jpeg_bytes):
    """POST /predict with valid JPEG should return 200 and valid schema."""
    response = client.post(
        "/predict",
        files={"file": ("test_ct.jpg", sample_jpeg_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "confidence" in data
    assert "probabilities" in data
    assert "disclaimer" in data
    assert 0.0 <= data["confidence"] <= 1.0
    assert len(data["probabilities"]) == 4


def test_predict_unsupported_extension(client):
    """POST /predict with .txt file should return 400 Bad Request."""
    response = client.post(
        "/predict",
        files={"file": ("notes.txt", b"some text content", "text/plain")},
    )
    assert response.status_code == 400
    data = response.json()
    assert "Unsupported file format" in data["detail"]


def test_predict_empty_file(client):
    """POST /predict with empty payload should return 400 Bad Request."""
    response = client.post(
        "/predict",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"]


def test_ui_endpoint(client):
    """GET /ui should return 200 with HTML content."""
    response = client.get("/ui")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "NephroScan AI" in response.text


def test_static_css_and_js_served(client):
    """GET /css/style.css and /js/app.js should return 200."""
    css_res = client.get("/css/style.css")
    assert css_res.status_code == 200
    assert "#EDEFF0" in css_res.text
    assert "#B3D9E5" in css_res.text
    assert "#30474E" in css_res.text

    js_res = client.get("/js/app.js")
    assert js_res.status_code == 200
    assert "NephroScan AI" in js_res.text
