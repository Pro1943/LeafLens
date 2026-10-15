import cv2
from fastapi.testclient import TestClient
import numpy as np
import pytest

from app.main import app, MAX_UPLOAD_MB
from app.schemas import AnalyzeResponse
from leaflens.config import APP_NAME


@pytest.fixture
def client():
    return TestClient(app)


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "app": APP_NAME}


def test_valid_synthetic_upload(client):
    img = np.full((200, 200, 3), 20, dtype=np.uint8)
    cv2.circle(img, (100, 100), 70, (35, 180, 45), -1)
    success, buf = cv2.imencode(".png", img)
    assert success

    response = client.post(
        "/analyze",
        files={"image": ("test_leaf.png", buf.tobytes(), "image/png")},
    )

    assert response.status_code == 200
    data = response.json()
    parsed = AnalyzeResponse.model_validate(data)
    assert parsed.verdict in ("healthy", "possible_issue", "uncertain", "retake_needed")
    assert parsed.images.original.startswith("data:image/jpeg;base64,")
    assert parsed.images.fixed.startswith("data:image/jpeg;base64,")
    assert parsed.images.overlay.startswith("data:image/jpeg;base64,")


def test_non_image_returns_400(client):
    response = client.post(
        "/analyze",
        files={"image": ("doc.txt", b"plain text data", "text/plain")},
    )
    assert response.status_code == 400
    assert "detail" in response.json()

    response_corrupt = client.post(
        "/analyze",
        files={"image": ("corrupted.png", b"not an image buffer", "image/png")},
    )
    assert response_corrupt.status_code == 400
    assert "detail" in response_corrupt.json()


def test_oversize_returns_413(client, monkeypatch):
    monkeypatch.setattr("app.main.MAX_UPLOAD_MB", 1)
    large_payload = b"\x00" * (1 * 1024 * 1024 + 1024)

    response = client.post(
        "/analyze",
        files={"image": ("huge.png", large_payload, "image/png")},
    )
    assert response.status_code == 413
    assert "detail" in response.json()


def test_cors_header_present(client):
    response = client.get(
        "/health",
        headers={"Origin": "http://localhost:3000"},
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"
