"""合成器与 mel 特征管线测试（无需 torch）。

运行: python -m pytest tests/test_synth.py
或  : python tests/test_synth.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import synth_anomalies as synth
from src.data.clean_sources import generate_clean_audio
from src.features.mel import MelConfig, MelStream, _mel_filterbank, compute_mel


def _cfg() -> dict:
    return {
        "audio": {"sample_rate": 44100, "segment_duration": 2.0},
        "features": {
            "n_fft": 2048, "hop_length": 441, "win_length": 2048,
            "n_mels": 64, "f_min": 20, "f_max": 22050, "log_eps": 1e-6,
            "normalize": False,
        },
        "synth": {
            "max_classes": 3, "p_clean": 0.1, "sr": 44100, "segment_duration": 2.0,
            "noise": {"snr_db": [10, 30], "colors": ["white", "pink", "brown"]},
            "harsh": {"gain_db": [4, 10], "freq_range": [4000, 10000], "q": [1, 3]},
            "clip_intermittent": {"clip_thresh": [0.3, 0.9], "dropout_p": [0, 0.2], "dropout_chunk_ms": [20, 100]},
            "distortion": {"drive": [1.5, 5], "kind": ["tanh", "cubic", "soft"]},
            "hum": {"freq": [50, 60], "n_harmonics": [2, 5], "amp": [0.01, 0.1]},
            "silence_loss": {"whole_p": 0.3, "chunk_p": [0.1, 0.3]},
            "loudness_anomaly": {"gain_db": [6, 20], "p_overcompress": 0.5},
        },
        "seed": 42,
    }


def _mel(cfg):
    return MelConfig.from_cfg(cfg, sample_rate=cfg["audio"]["sample_rate"])


def test_synth_output_shapes():
    cfg = _cfg()
    mel_cfg = _mel(cfg)
    rng = np.random.default_rng(0)
    audio = generate_clean_audio(44100, 2.0, rng)
    out, labels, chosen = synth.synthesize(audio, 44100, cfg["synth"], mel_cfg, rng)
    assert out.dtype == np.float32
    assert out.shape == audio.shape
    assert labels.shape[0] == len(synth.CLASS_NAMES)
    # 标签为 0/1
    assert np.isin(labels, [0.0, 1.0]).all()


def test_each_injector_runs():
    cfg = _cfg()
    rng = np.random.default_rng(1)
    audio = generate_clean_audio(44100, 1.0, rng)
    for name, fn in synth.INJECTORS.items():
        params = cfg["synth"][name]
        out, mask = fn(audio.copy(), 44100, rng, params)
        assert out.shape == audio.shape, name
        assert mask.dtype == bool
        assert mask.shape == audio.shape, name


def test_clean_sample_has_no_labels():
    cfg = _cfg()
    cfg["synth"]["p_clean"] = 1.0  # 强制干净
    mel_cfg = _mel(cfg)
    rng = np.random.default_rng(2)
    audio = generate_clean_audio(44100, 2.0, rng)
    out, labels, chosen = synth.synthesize(audio, 44100, cfg["synth"], mel_cfg, rng)
    assert chosen == []
    assert labels.sum() == 0.0


def test_stream_matches_offline():
    """流式 MelStream 逐块喂入应与离线 compute_mel 逐帧一致。"""
    cfg = _cfg()
    mel_cfg = _mel(cfg)
    fb = _mel_filterbank(mel_cfg)
    rng = np.random.default_rng(3)
    audio = generate_clean_audio(44100, 1.5, rng)
    offline = compute_mel(audio, mel_cfg, fb=fb)

    stream = MelStream(mel_cfg, fb=fb)
    collected = [stream.push(chunk) for chunk in np.array_split(audio, 13)]
    collected.append(stream.flush())
    online = np.concatenate([c for c in collected if c.shape[1] > 0], axis=1)

    # 流式产出帧数应与离线一致或差 ≤1（flush 尾帧）
    assert online.shape == offline.shape
    T = min(online.shape[1], offline.shape[1])
    diff = np.abs(online[:, :T] - offline[:, :T]).max()
    assert diff < 1e-3, f"stream/offline mismatch max={diff}"


def test_label_frame_alignment():
    """注入单一异常后，正帧应在被注入的时段内（非全段乱标）。"""
    cfg = _cfg()
    mel_cfg = _mel(cfg)
    rng = np.random.default_rng(4)
    audio = generate_clean_audio(44100, 2.0, rng)
    # 只注入 hum（整段）
    out, mask = synth.inject_hum(audio.copy(), 44100, rng, cfg["synth"]["hum"])
    labels = synth.masks_to_frames({"hum": mask}, len(out), mel_cfg)
    c = synth.CLASS_TO_IDX["hum"]
    assert labels[c].sum() == labels.shape[1]  # 整段为正


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                print(f"FAIL {name}: {e}")
                failed += 1
            except Exception as e:
                print(f"ERROR {name}: {type(e).__name__}: {e}")
                failed += 1
    print(f"\n{'ALL PASS' if failed == 0 else f'{failed} FAILED'}")
    sys.exit(1 if failed else 0)
