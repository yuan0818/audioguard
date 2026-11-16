"""Causal streaming with exact frame order, shared features and per-stream state."""
from __future__ import annotations

import numpy as np
import torch

from src.features.mel import MelConfig, MelStream, normalize_mel


def smooth_probs(probs, window=1):
    """Causal moving average. No future frames or padding affect past outputs."""
    if window <= 1 or probs.shape[-1] == 0:
        return probs
    out = np.empty_like(probs)
    for i in range(probs.shape[-1]):
        out[..., i] = probs[..., max(0, i - window + 1):i + 1].mean(axis=-1)
    return out


class StreamingDetector:
    def __init__(self, model, mel_cfg: MelConfig, thresholds=None, device=None,
                 window_frames=64, hop_frames=16, smooth_window=1, retain_history=True):
        if mel_cfg.normalize and mel_cfg.normalization == "clip":
            raise ValueError("Per-clip normalization requires the whole recording. "
                             "Use offline inference for legacy weights; train with fixed normalization for streaming.")
        if hop_frames < 1 or smooth_window < 1:
            raise ValueError("hop_frames and smooth_window must be positive")
        self.model = model.eval()
        self.mel_cfg = mel_cfg
        self.stream = MelStream(mel_cfg)
        self.thresholds = thresholds
        self.device = device or torch.device("cpu")
        self.hop_frames = hop_frames
        self.smooth_window = smooth_window
        self.retain_history = retain_history
        self._pending = np.zeros((mel_cfg.n_features, 0), dtype=np.float32)
        self._state = None
        self._history = []
        self._smooth = np.zeros((model.n_classes, 0), dtype=np.float32)
        self._finished = False

    @classmethod
    def from_cfg(cls, model, cfg, thresholds=None, device=None, retain_history=True):
        mel_cfg = MelConfig.from_cfg(cfg)
        s = cfg.get("inference", {}).get("streaming", {})
        hop = max(1, round(s.get("hop_ms", 100) / mel_cfg.frame_ms()))
        return cls(model, mel_cfg, thresholds, device, hop_frames=hop,
                   smooth_window=int(s.get("smooth_window", 1)), retain_history=retain_history)

    def _infer(self, mel):
        x = torch.from_numpy(normalize_mel(mel, self.mel_cfg))[None, None].to(self.device)
        with torch.no_grad():
            logits, self._state = self.model.forward_stream(x, self._state)
        raw = torch.sigmoid(logits)[0].cpu().numpy()
        joined = np.concatenate([self._smooth, raw], axis=1)
        out = smooth_probs(joined, self.smooth_window)[:, -raw.shape[1]:]
        self._smooth = joined[:, -(self.smooth_window - 1):] if self.smooth_window > 1 else joined[:, :0]
        if self.retain_history:
            self._history.append(out)
        return out

    def _drain(self, final=False):
        out = []
        while self._pending.shape[1] >= self.hop_frames or (final and self._pending.shape[1]):
            n = min(self.hop_frames, self._pending.shape[1])
            out.append(self._infer(self._pending[:, :n]))
            self._pending = self._pending[:, n:]
        return np.concatenate(out, axis=1) if out else np.zeros((self.model.n_classes, 0), dtype=np.float32)

    def push_samples(self, samples):
        if self._finished:
            raise RuntimeError("Cannot push after flush")
        mel = self.stream.push(samples)
        self._pending = np.concatenate([self._pending, mel], axis=1)
        return self._drain()

    def flush(self):
        if self._finished:
            return np.zeros((self.model.n_classes, 0), dtype=np.float32)
        self._finished = True
        self._pending = np.concatenate([self._pending, self.stream.flush()], axis=1)
        return self._drain(final=True)

    def all_probs(self):
        if not self.retain_history:
            raise RuntimeError("History disabled; consume push_samples/flush results")
        return np.concatenate(self._history, axis=1) if self._history else np.zeros((self.model.n_classes, 0), dtype=np.float32)

    def all_predictions(self):
        probs = self.all_probs()
        thresholds = 0.5 if self.thresholds is None else np.asarray(self.thresholds).reshape(-1, 1)
        return (probs >= thresholds).astype(np.int32)


def probs_to_segments(
    preds: np.ndarray, mel_cfg: MelConfig, min_segment_ms: int = 200
) -> list[dict]:
    """per-frame 预测 [n_classes, T] -> 时间轴段落 [{class_id,start_s,end_s,duration_s}]。

    合并短于 min_segment_ms 的连续正预测。
    """
    n_classes, T = preds.shape
    frame_ms = mel_cfg.frame_ms()
    min_frames = max(1, int(min_segment_ms / frame_ms))
    segs = []
    for c in range(n_classes):
        run = preds[c]
        i = 0
        while i < T:
            if run[i]:
                j = i
                while j < T and run[j]:
                    j += 1
                if j - i >= min_frames:
                    start = i * frame_ms / 1000.0
                    end = j * frame_ms / 1000.0
                    segs.append({"class_id": int(c), "start_s": round(start, 3),
                                 "end_s": round(end, 3), "duration_s": round(end - start, 3)})
                i = j
            else:
                i += 1
    segs.sort(key=lambda s: (s["start_s"], s["class_id"]))
    return segs
