"""统一 log-mel 谱图特征提取。

训练与推理共用本模块，保证特征一致（避免偏移）。采用 center=False 分帧，
使离线整段计算与流式增量计算逐帧对齐。

接口：
    compute_mel(audio, sr, cfg) -> np.ndarray [n_mels, T]
    MelStream: 增量喂入音频块，逐帧吐出 mel 帧。

可选时域统计（features.time_features: true）：对每帧同一窗口的原始样本计算
N_TD 个统计量（峰值/RMS/峰均比/近满幅占比/瞬态跳变/过零率/零值与平顶占比/
子块包络跨度），作为额外行拼接在 log-mel 之后 -> [n_mels + N_TD, T]。
它们只依赖该帧窗口内的样本，因此天然因果、流式与离线逐帧一致；
用于弥补 46ms STFT 窗口抹平的毫秒级爆音、削波、断续线索。
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
    time_features: bool = False  # 旧 checkpoint 无此键 -> 仅 log-mel

    @property
    def n_td(self) -> int:
        return N_TD if self.time_features else 0

    @property
    def n_features(self) -> int:
        return self.n_mels + self.n_td

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
            time_features=bool(feat.get("time_features", False)),
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


def _raw_frames(audio: np.ndarray, cfg: MelConfig) -> np.ndarray:
    """center=False 分帧，返回 [n_frames, n_fft] 的未加窗帧。"""
    if len(audio) < cfg.n_fft:
        pad = np.zeros(cfg.n_fft - len(audio), dtype=np.float32)
        audio = np.concatenate([audio, pad])
    # librosa.util.frame: 滑窗切片
    return librosa.util.frame(
        audio, frame_length=cfg.win_length, hop_length=cfg.hop_length, axis=0
    )


def _frames(audio: np.ndarray, cfg: MelConfig) -> np.ndarray:
    """center=False 分帧，返回 [n_frames, n_fft] 的加窗帧。"""
    win = _window(cfg)
    return _raw_frames(audio, cfg) * win[np.newaxis, :].astype(np.float32)


# --------------------------------------------------------------------------- #
# 时域统计特征
# --------------------------------------------------------------------------- #
TD_NAMES = (
    "peak_db",     # 帧峰值电平
    "rms_db",      # 帧 RMS 电平
    "crest_db",    # 峰均比（削波/过压缩会降低，爆音会升高）
    "top_frac",    # |x| >= 0.98*peak 的样本占比（硬削波平顶）
    "jump",        # log10(max|dx| / mean|dx|)（点击/爆音瞬态）
    "zcr",         # 过零率
    "zero_frac",   # 近零样本占比（dropout/静音）
    "flat_frac",   # 相邻样本完全相等且非零的占比（削波平顶/卡死）
    "env_range",   # 帧内 16 个子块 RMS(dB) 的极差（断续/卡顿）
)
N_TD = len(TD_NAMES)
TD_LEVEL_ROWS = (0, 1)   # 随整体增益平移的行（peak_db, rms_db）
TD_DB_SCALE = 1.0 / 20.0  # 这两行 1 dB 对应的特征偏移
_TD_SUBBLOCKS = 16


def time_domain_features(raw: np.ndarray) -> np.ndarray:
    """未加窗帧 [T, N] -> 固定仿射归一化后的时域统计 [N_TD, T]。

    只使用每帧自身窗口内的样本，与 log-mel 帧完全对齐（因果、流式可复现）。
    """
    x = np.asarray(raw, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("Expected [n_frames, n_fft] frames")
    T, N = x.shape
    if T == 0:
        return np.zeros((N_TD, 0), dtype=np.float32)
    ax = np.abs(x)
    peak = ax.max(axis=1)
    rms = np.sqrt((x * x).mean(axis=1))
    peak_db = 20.0 * np.log10(peak + 1e-5)
    rms_db = 20.0 * np.log10(rms + 1e-5)
    crest_db = peak_db - rms_db
    top_frac = np.where(peak > 1e-4, (ax >= 0.98 * peak[:, None]).mean(axis=1), 0.0)
    dx = np.abs(np.diff(x, axis=1))
    jump = np.log10((dx.max(axis=1) + 1e-7) / (dx.mean(axis=1) + 1e-7))
    sgn = np.signbit(x)
    zcr = (sgn[:, 1:] != sgn[:, :-1]).mean(axis=1)
    zero_frac = (ax < 1e-5).mean(axis=1)
    flat_frac = ((dx < 1e-7) & (ax[:, 1:] > 1e-3)).mean(axis=1)
    nb = N // _TD_SUBBLOCKS
    sub = x[:, : nb * _TD_SUBBLOCKS].reshape(T, _TD_SUBBLOCKS, nb)
    sub_db = 20.0 * np.log10(np.sqrt((sub * sub).mean(axis=2)) + 1e-5)
    env_range = np.minimum(sub_db.max(axis=1) - sub_db.min(axis=1), 60.0)
    feats = np.stack([
        (peak_db + 20.0) / 20.0,
        (rms_db + 30.0) / 20.0,
        (crest_db - 12.0) / 6.0,
        top_frac * 10.0,
        jump - 1.0,
        zcr * 4.0,
        zero_frac * 2.0,
        flat_frac * 2.0,
        env_range / 20.0,
    ], axis=0)
    return feats.astype(np.float32)


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
    raw = _raw_frames(audio, cfg)
    log_mel = _frames_to_mel(raw * _window(cfg)[np.newaxis, :], cfg, fb)
    return normalize_mel(_stack_td(log_mel, raw, cfg), cfg)


def _stack_td(log_mel: np.ndarray, raw: np.ndarray, cfg: MelConfig) -> np.ndarray:
    if not cfg.time_features:
        return log_mel
    return np.concatenate([log_mel, time_domain_features(raw)], axis=0)


def normalize_mel(mel: np.ndarray, cfg: MelConfig) -> np.ndarray:
    """归一化 log-mel 行；时域统计行（若有）已自带固定归一化，原样保留。"""
    if not cfg.normalize:
        return mel
    m = mel[: cfg.n_mels]
    if cfg.normalization == "fixed":
        if cfg.norm_std <= 0:
            raise ValueError("norm_std must be positive")
        m = (m - cfg.norm_mean) / cfg.norm_std
    elif cfg.normalization == "clip":
        m = (m - m.mean()) / (m.std() + 1e-6)
    else:
        raise ValueError(f"Unknown normalization: {cfg.normalization}")
    if mel.shape[0] == cfg.n_mels:
        return m.astype(np.float32)
    return np.concatenate([m, mel[cfg.n_mels:]], axis=0).astype(np.float32)


def gain_scale(cfg: MelConfig) -> float:
    """1 dB 增益在归一化 log-mel 上的偏移量（逐段标准化下增益被抵消 -> 0）。"""
    if not cfg.normalize:
        return 1.0
    if cfg.normalization == "fixed":
        return 1.0 / cfg.norm_std
    return 0.0


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
        """喂入一块音频，返回本块新产生的特征帧 [n_features, T_new]（可能为 0 列）。"""
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
            new_frames.append(self._buf[start : start + cfg.win_length])
            self._next_frame_start += cfg.hop_length

        if not new_frames:
            return np.zeros((cfg.n_features, 0), dtype=np.float32)

        self._frames_emitted += len(new_frames)
        raw = np.stack(new_frames, axis=0).astype(np.float32)
        log_mel = _stack_td(_frames_to_mel(raw * self.win, cfg, self.fb), raw, cfg)

        # 回收已不再需要的旧样本（早于游标 - n_fft 的部分）
        keep_from = max(0, self._next_frame_start - cfg.n_fft + cfg.hop_length)
        if keep_from > 0:
            self._buf = self._buf[keep_from:]
            self._next_frame_start -= keep_from
        return log_mel

    def flush(self) -> np.ndarray:
        """返回尾部不足一帧时按零填充补出的最后一帧（可选，保持与离线尾部一致）。"""
        if self._finished:
            return np.zeros((self.cfg.n_features, 0), dtype=np.float32)
        self._finished = True
        # Match compute_mel: one padded frame only for nonempty short audio;
        # discard an incomplete trailing frame after full frames were emitted.
        if self._frames_emitted or not self._total:
            return np.zeros((self.cfg.n_features, 0), dtype=np.float32)
        raw = np.pad(self._buf, (0, self.cfg.n_fft - len(self._buf)))[None, :]
        return _stack_td(_frames_to_mel(raw * self.win, self.cfg, self.fb), raw, self.cfg)



def frames_to_time(n_frames: int, cfg: MelConfig) -> np.ndarray:
    """帧索引 -> 时间（秒），center=False 下帧中心 = (i*hop + n_fft//2)/sr。"""
    i = np.arange(n_frames)
    return (i * cfg.hop_length + cfg.n_fft // 2) / cfg.sample_rate
