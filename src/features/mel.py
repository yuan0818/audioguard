"""统一 log-mel 谱图特征提取。

训练与推理共用本模块，保证特征一致（避免偏移）。采用 center=False 分帧，
使离线整段计算与流式增量计算逐帧对齐。

接口：
    compute_mel(audio, sr, cfg) -> np.ndarray [n_mels, T]
    MelStream: 增量喂入音频块，逐帧吐出 mel 帧。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import librosa


@dataclass
class MelConfig:
    """Mel 超参（由 config dict 构造）。"""

    n_fft: int = 2048
    hop_length: int = 441
    win_length: int = 2048
    n_mels: int = 128
    f_min: float = 20.0
    f_max: float = 22050.0
    sample_rate: int = 44100
    log_eps: float = 1e-6
    normalize: bool = True
    normalization: str = "clip"  # legacy checkpoints; new configs use fixed
    norm_mean: float = -40.0
    norm_std: float = 20.0

    @classmethod
    def from_cfg(cls, cfg: Mapping, sample_rate: int | None = None) -> "MelConfig":
        feat = dict(cfg.get("features", {}))
        sr = sample_rate or cfg.get("audio", {}).get("sample_rate", 44100)
        return cls(
            n_fft=feat.get("n_fft", 2048),
            hop_length=feat.get("hop_length", 441),
            win_length=feat.get("win_length", feat.get("n_fft", 2048)),
            n_mels=feat.get("n_mels", 128),
            f_min=feat.get("f_min", 20.0),
            f_max=feat.get("f_max", sr / 2),
            sample_rate=sr,
            log_eps=feat.get("log_eps", 1e-6),
            normalize=feat.get("normalize", True),
            normalization=feat.get("normalization", "clip"),
            norm_mean=feat.get("norm_mean", -40.0),
            norm_std=feat.get("norm_std", 20.0),
        )

    def frame_ms(self) -> float:
        return self.hop_length * 1000.0 / self.sample_rate


def _mel_filterbank(cfg: MelConfig) -> np.ndarray:
    """构造 mel 滤波器组 [n_mels, n_fft//2+1]。"""
    return librosa.filters.mel(
        sr=cfg.sample_rate,
        n_fft=cfg.n_fft,
        n_mels=cfg.n_mels,
        fmin=cfg.f_min,
        fmax=cfg.f_max,
        htk=False,
        norm="slaney",
    ).astype(np.float32)


def _window(cfg: MelConfig) -> np.ndarray:
    return librosa.filters.get_window("hann", cfg.win_length, fftbins=True).astype(np.float32)


def _frames(audio: np.ndarray, cfg: MelConfig) -> np.ndarray:
    """center=False 分帧，返回 [n_frames, n_fft] 的加窗帧。"""
    if len(audio) < cfg.n_fft:
        pad = np.zeros(cfg.n_fft - len(audio), dtype=np.float32)
        audio = np.concatenate([audio, pad])
    # librosa.util.frame: 滑窗切片
    frames = librosa.util.frame(
        audio, frame_length=cfg.win_length, hop_length=cfg.hop_length, axis=0
    )
    win = _window(cfg)
    return frames * win[np.newaxis, :].astype(np.float32)


def _frames_to_mel(frames: np.ndarray, cfg: MelConfig, fb: np.ndarray) -> np.ndarray:
    """加窗帧 -> log-mel [n_mels, n_frames]。

    输出随帧增长，可直接被流式与离线复用。
    """
    # 每帧 rFFT -> 功率谱
    spec = np.fft.rfft(frames, n=cfg.n_fft, axis=1)  # [T, n_fft//2+1]
    power = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)
    # mel 滤波
    mel = fb @ power.T  # [n_mels, T]
    # log 压缩（power -> dB）
    log_mel = 10.0 * np.log10(mel + cfg.log_eps)
    return log_mel.astype(np.float32)


def compute_mel(audio: np.ndarray, cfg: MelConfig, fb: np.ndarray | None = None) -> np.ndarray:
    """整段音频 -> log-mel 谱图 [n_mels, T_frames]。

    Args:
        audio: float32 一维波形。
        cfg: MelConfig。
        fb: 预构造 mel 滤波器组（可选，复用以省时）。
    """
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1 or audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError("Expected nonempty finite mono audio")
    if cfg.win_length != cfg.n_fft:
        raise ValueError("win_length must equal n_fft")
    if fb is None:
        fb = _mel_filterbank(cfg)
    frames = _frames(audio, cfg)
    log_mel = _frames_to_mel(frames, cfg, fb)
    return normalize_mel(log_mel, cfg)


def normalize_mel(mel: np.ndarray, cfg: MelConfig) -> np.ndarray:
    if not cfg.normalize:
        return mel
    if cfg.normalization == "fixed":
        if cfg.norm_std <= 0:
            raise ValueError("norm_std must be positive")
        return ((mel - cfg.norm_mean) / cfg.norm_std).astype(np.float32)
    if cfg.normalization != "clip":
        raise ValueError(f"Unknown normalization: {cfg.normalization}")
    return ((mel - mel.mean()) / (mel.std() + 1e-6)).astype(np.float32)


class MelStream:
    """增量 mel 计算：逐块喂入音频，吐出新产生的 mel 帧。

    用于流式推理。内部维护样本缓冲与帧游标，复用同一滤波器组。
    与 compute_mel(center=False, 无归一化) 逐帧一致；归一化在调用方按需后处理。
    """

    def __init__(self, cfg: MelConfig, fb: np.ndarray | None = None) -> None:
        self.cfg = cfg
        self.fb = fb if fb is not None else _mel_filterbank(cfg)
        self.win = _window(cfg)
        self._buf = np.zeros(0, dtype=np.float32)
        self._next_frame_start = 0  # 下一帧在 buf 中的起始样本索引
        self._finished = False
        self._total = 0
        self._frames_emitted = 0
        self._carried = 0  # buf 中已处理过的样本数（用于游标换算）

    def push(self, chunk: np.ndarray) -> np.ndarray:
        """喂入一块音频，返回本块新产生的 log-mel 帧 [n_mels, T_new]（可能为 0 列）。"""
        chunk = np.asarray(chunk, dtype=np.float32)
        if self._finished:
            raise RuntimeError("Cannot push after flush")
        if chunk.ndim != 1 or not np.isfinite(chunk).all():
            raise ValueError("Expected finite mono audio")
        if self.cfg.win_length != self.cfg.n_fft:
            raise ValueError("win_length must equal n_fft")
        self._total += len(chunk)
        self._buf = np.concatenate([self._buf, chunk])

        cfg = self.cfg
        new_frames = []
        # 当缓冲中从游标起存在完整一帧时产出
        while self._next_frame_start + cfg.n_fft <= len(self._buf):
            start = self._next_frame_start
            frame = self._buf[start : start + cfg.win_length]
            new_frames.append(frame * self.win)
            self._next_frame_start += cfg.hop_length

        if not new_frames:
            return np.zeros((cfg.n_mels, 0), dtype=np.float32)

        self._frames_emitted += len(new_frames)
        frames = np.stack(new_frames, axis=0).astype(np.float32)
        log_mel = _frames_to_mel(frames, cfg, self.fb)

        # 回收已不再需要的旧样本（早于游标 - n_fft 的部分）
        keep_from = max(0, self._next_frame_start - cfg.n_fft + cfg.hop_length)
        if keep_from > 0:
            self._buf = self._buf[keep_from:]
            self._next_frame_start -= keep_from
        return log_mel

    def flush(self) -> np.ndarray:
        """返回尾部不足一帧时按零填充补出的最后一帧（可选，保持与离线尾部一致）。"""
        if self._finished:
            return np.zeros((self.cfg.n_mels, 0), dtype=np.float32)
        self._finished = True
        # Match compute_mel: one padded frame only for nonempty short audio;
        # discard an incomplete trailing frame after full frames were emitted.
        if self._frames_emitted or not self._total:
            return np.zeros((self.cfg.n_mels, 0), dtype=np.float32)
        frame = np.pad(self._buf, (0, self.cfg.n_fft - len(self._buf))) * self.win
        return _frames_to_mel(frame[None, :], self.cfg, self.fb)



def frames_to_time(n_frames: int, cfg: MelConfig) -> np.ndarray:
    """帧索引 -> 时间（秒），center=False 下帧中心 = (i*hop + n_fft//2)/sr。"""
    i = np.arange(n_frames)
    return (i * cfg.hop_length + cfg.n_fft // 2) / cfg.sample_rate
