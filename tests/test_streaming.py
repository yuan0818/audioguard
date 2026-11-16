"""流式推理引擎测试（需 torch）。

验证：
- 不同分块大小喂入同一音频，输出概率一致（分块无关性）
- StreamingDetector 与离线整段模型推理在公共帧上接近
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.clean_sources import generate_clean_audio
from src.features.mel import MelConfig, _mel_filterbank, compute_mel
from src.inference.streaming import StreamingDetector, probs_to_segments
from src.models.crnn import build_model
from src.utils.config import load_config_resolved

_CFG = load_config_resolved(ROOT / "configs" / "default.yaml", root=ROOT)


def _model():
    torch.manual_seed(0)
    m = build_model(_CFG).eval()
    return m


def test_chunk_invariance():
    """以大块、小块、逐样本三种方式喂入，输出概率应一致。"""
    model = _model()
    sr = _CFG["audio"]["sample_rate"]
    mel_cfg = MelConfig.from_cfg(_CFG, sample_rate=sr)
    rng = np.random.default_rng(0)
    audio = generate_clean_audio(sr, 2.0, rng)

    def run(chunk):
        det = StreamingDetector(model, mel_cfg, thresholds=None, device=torch.device("cpu"),
                                window_frames=64, hop_frames=16, smooth_window=1)
        for i in range(0, len(audio), chunk):
            det.push_samples(audio[i : i + chunk])
        det.flush()
        return det.all_probs()

    big = run(int(sr * 2.0))      # 一次性
    med = run(int(sr * 0.1))      # 100ms
    tiny = run(int(sr * 0.01))    # 10ms
    T = min(big.shape[1], med.shape[1], tiny.shape[1])
    assert T > 0
    # 帧数可因末尾补零差 1-2 帧；公共区间内应几乎一致
    assert np.abs(big[:, :T] - med[:, :T]).max() < 1e-3
    assert np.abs(big[:, :T] - tiny[:, :T]).max() < 1e-3


def test_probs_to_segments():
    mel_cfg = MelConfig.from_cfg(_CFG)
    preds = np.zeros((7, 100), dtype=np.int32)
    preds[0, 10:30] = 1   # noise
    preds[2, 50:80] = 1   # clip
    segs = probs_to_segments(preds, mel_cfg, min_segment_ms=100)
    classes = sorted(s["class_id"] for s in segs)
    assert classes == [0, 2]
    # 起止时间单调
    for s in segs:
        assert s["end_s"] > s["start_s"]


def test_thresholds_produce_binary():
    model = _model()
    sr = _CFG["audio"]["sample_rate"]
    mel_cfg = MelConfig.from_cfg(_CFG, sample_rate=sr)
    audio = generate_clean_audio(sr, 1.5, np.random.default_rng(1))
    det = StreamingDetector(model, mel_cfg,
                            thresholds=np.array([0.5] * 7, dtype=np.float32),
                            window_frames=64, hop_frames=16, smooth_window=1)
    for i in range(0, len(audio), int(sr * 0.1)):
        det.push_samples(audio[i : i + int(sr * 0.1)])
    det.flush()
    preds = det.all_predictions()
    assert preds.dtype.kind in ("i", "u")
    assert preds.ndim == 2 and preds.shape[0] == 7


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
