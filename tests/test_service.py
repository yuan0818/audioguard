"""Self-contained HTTP tests: fixtures generate weights and audio in temporary storage."""
import io
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pytest
import soundfile as sf
import torch
from fastapi.testclient import TestClient
from src.inference.service import build_app
from src.models.crnn import build_model
from src.utils.config import load_config


@pytest.fixture()
def client(tmp_path):
    cfg = load_config("configs/default.yaml")
    cfg["model"]["name"] = "small"
    checkpoint = tmp_path / "model.pt"
    torch.save({"model": build_model(cfg).state_dict(), "config": cfg,
                "thresholds": [0.5] * 7, "clip_thresholds": [0.5] * 7}, checkpoint)
    with TestClient(build_app(str(checkpoint), workers=4, num_threads=1)) as c:
        yield c


def audio_bytes():
    buf = io.BytesIO()
    sf.write(buf, np.random.default_rng(0).normal(0, 0.1, 11025).astype("float32"), 44100, format="WAV")
    return buf.getvalue()


def test_health(client):
    assert client.get("/health").json()["model_loaded"] is True


def test_detect_structure_and_determinism(client):
    def detect():
        r = client.post("/detect", files={"file": ("sample.wav", audio_bytes(), "audio/wav")})
        assert r.status_code == 200, r.text
        return r.json()
    a, b = detect(), detect()
    assert a == b
    assert a["duration_s"] == 0.25
    assert a["n_frames"] > 0
    assert len(a["clip_level"]["probs"]) == 7


def test_concurrency(client):
    buf = audio_bytes()
    def one(_):
        return client.post("/detect", files={"file": ("s.wav", buf, "audio/wav")}).status_code
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert list(pool.map(one, range(16))) == [200] * 16
    assert client.get("/metrics").json()["requests_total"] == 16


def test_empty_upload(client):
    assert client.post("/detect", files={"file": ("s.wav", b"", "audio/wav")}).status_code == 400
