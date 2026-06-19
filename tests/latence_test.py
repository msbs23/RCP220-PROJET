import io
from unittest.mock import MagicMock, patch

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image


def _make_fake_image_bytes() -> bytes:
    """Crée une image PNG synthétique 224x224 en mémoire."""
    img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _fake_model():
    """Retourne un (feature_extractor, head) factice qui produit un tenseur fixe."""
    feature_extractor = MagicMock()
    feature_extractor.return_value = torch.zeros(1, 2048)

    head = MagicMock()
    head.return_value = torch.tensor([[0.8]])  # sigmoid(0.8) ≈ 0.69 → label "malade"

    return feature_extractor, head


@pytest.fixture
def client():
    with patch("api_radio.singleton_model.ModelSingleton.get_instance", return_value=_fake_model()):
        from api_radio.main import app
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


# ── Test 1 : vérification générale de l'API ────────────────────────────────

def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "OK"}


# ── Test 2 : envoi d'une image sur /predict ────────────────────────────────

def test_predict_image(client):
    image_bytes = _make_fake_image_bytes()

    response = client.post(
        "/predict",
        files={"image": ("radio_test.png", image_bytes, "image/png")},
    )

    assert response.status_code == 200
    data = response.json()
    assert "label" in data
    assert "probabilite" in data
    assert data["label"] in ("malade", "sain")
    assert 0.0 <= data["probabilite"] <= 1.0