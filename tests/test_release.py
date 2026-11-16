import copy
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf
import torch
from src.features.mel import MelConfig, MelStream, compute_mel
from src.inference.streaming import StreamingDetector, smooth_probs
from src.models.crnn import build_model
from src.utils.config import load_config
from src.training.train import partition_sources, build_loaders
from src.training.losses import build_loss
from src.data.dataset import LabeledDataset, collate_variable


@pytest.mark.parametrize("samples", [1, 2047, 2048, 2489, 44100])
@pytest.mark.parametrize("normalize", [False, True])
def test_exact_stream_frames_and_probabilities(samples, normalize):
    cfg = load_config("configs/default.yaml")
    cfg["features"]["normalize"] = normalize
    cfg["model"]["name"] = "small"
    torch.manual_seed(0)
    model = build_model(cfg).eval()
    audio = np.random.default_rng(0).normal(0, 0.1, samples).astype("float32")
    mc = MelConfig.from_cfg(cfg)
    with torch.no_grad():
        expected = torch.sigmoid(model(torch.from_numpy(compute_mel(audio, mc))[None, None])[0])[0].numpy()
    det = StreamingDetector(model, mc, hop_frames=7, smooth_window=5)
    chunks = [det.push_samples(c) for c in np.array_split(audio, 17)]
    chunks.append(det.flush())
    actual = np.concatenate(chunks, axis=1)
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, smooth_probs(expected, 5), atol=2e-6)
    np.testing.assert_allclose(actual, det.all_probs(), atol=0)
    assert det.flush().shape[1] == 0
    with pytest.raises(RuntimeError):
        det.push_samples(audio)


def test_reject_legacy_clip_normalization_in_stream():
    cfg = load_config("configs/default.yaml")
    cfg["features"].pop("normalization")
    with pytest.raises(ValueError, match="whole recording"):
        StreamingDetector.from_cfg(build_model(cfg), cfg)


def test_first_stream_output_under_300ms_and_bounded_state():
    cfg = load_config("configs/default.yaml")
    model = build_model(cfg).eval()
    det = StreamingDetector.from_cfg(model, cfg, retain_history=False)
    first = None
    for i in range(100):
        out = det.push_samples(np.zeros(441, dtype="float32"))
        if out.shape[1] and first is None:
            first = (i + 1) * 10
    assert first <= 300
    assert det._pending.shape[1] < det.hop_frames
    assert det._state[0].shape[-1] == sum(b._causal_time for b in model.conv)
    assert not det._history


def test_partition_group_and_duplicate_content(tmp_path):
    records = []
    for i in range(6):
        f = tmp_path / f"{i}.wav"
        f.write_bytes(bytes([i]))
        records.append({"path": str(f), "source_id": f"source_{i // 2}"})
    duplicate = tmp_path / "copy.wav"
    duplicate.write_bytes(bytes([0]))
    records.append({"path": str(duplicate), "source_id": "copy"})
    tr, va = partition_sources(records, 42)
    assert {r["path"] for r in tr}.isdisjoint({r["path"] for r in va})
    assert {r["source_id"] for r in tr}.isdisjoint({r["source_id"] for r in va})
    for subset in (tr, va):
        contents = [Path(r["path"]).read_bytes() for r in subset]
        assert contents.count(bytes([0])) in (0, 2)


@pytest.mark.parametrize("kind", ["bce", "focal", "frame_seg"])
def test_loss_ignores_padding(kind):
    loss = build_loss({"train": {"loss": kind}}, torch.ones(7))
    x = torch.randn(1, 7, 4)
    y = torch.randint(0, 2, x.shape).float()
    padded_x = torch.cat([x, torch.full((1, 7, 6), 100.0)], dim=-1)
    padded_y = torch.cat([y, torch.zeros(1, 7, 6)], dim=-1)
    mask = torch.arange(10)[None] < 4
    torch.testing.assert_close(loss(x, y), loss(padded_x, padded_y, mask))


def test_labeled_overlaps_and_clean_regions(tmp_path):
    cfg = load_config("configs/default.yaml")
    f = tmp_path / "recording.wav"
    sf.write(f, np.ones(44100) * 0.1, 44100)
    ds = LabeledDataset(cfg, records=[{"file": str(f), "segments": [
        {"class": "hum", "start_s": 0.2, "end_s": 0.5},
        {"class": "noise", "start_s": 0.3, "end_s": 0.6}]}])
    mel, labels = ds[0]
    assert labels[4, 35] == labels[0, 35] == 1
    assert labels[:, 0].sum() == 0
    assert labels[:, 80].sum() == 0
    batch = collate_variable([ds[0], (mel[:, :10], labels[:, :10])])
    assert batch[2].tolist() == [mel.shape[1], 10]


def test_safe_loader_accepts_legacy_numpy_thresholds(tmp_path):
    from src.inference.offline import load_checkpoint
    cfg = load_config("configs/default.yaml")
    cfg["features"].pop("normalization")
    cfg["model"]["name"] = "small"
    checkpoint = tmp_path / "legacy.pt"
    torch.save({"model": build_model(cfg).state_dict(), "config": cfg,
                "thresholds": np.full(7, 0.5, dtype=np.float32)}, checkpoint)
    loaded, _, thresholds, _ = load_checkpoint(checkpoint, torch.device("cpu"))
    assert MelConfig.from_cfg(loaded).normalization == "clip"
    np.testing.assert_array_equal(thresholds, np.full(7, 0.5, dtype=np.float32))


def test_offline_api_matches_evaluation_scores():
    from src.inference.offline import infer_audio
    cfg = load_config("configs/default.yaml")
    cfg["model"]["name"] = "small"
    audio = np.random.default_rng(1).normal(0, 0.1, 44100).astype("float32")
    model = build_model(cfg).eval()
    with torch.no_grad():
        expected = torch.sigmoid(model(torch.from_numpy(compute_mel(audio, MelConfig.from_cfg(cfg)))[None, None])[0])[0].numpy()
    result = infer_audio(audio, 44100, cfg, model, np.full(7, 0.5), torch.device("cpu"))
    assert result["n_frames"] == expected.shape[1]
    for c, name in enumerate(cfg["classes"]):
        assert abs(result["clip_level"]["probs"][name] - float(expected[c].mean())) <= 0.000501
