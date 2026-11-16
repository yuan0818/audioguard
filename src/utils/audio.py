"""音频 IO 工具：加载、重采样、归一化、能量计算。

统一音频进入模型前的预处理，保证训练与推理一致。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def load_audio(path: str | os.PathLike, sr: int, mono: bool = True) -> np.ndarray:
    """加载音频文件并重采样到 sr，返回 float32 一维数组。

    Args:
        path: 音频文件路径（wav/flac/ogg…，依赖 soundfile 后端）。
        sr: 目标采样率。
        mono: 是否下混为单声道。
    """
    path = Path(path)
    with open(path, "rb") as f:
        return decode_audio(f.read(), sr, mono=mono)


def decode_audio(buf: bytes, sr: int, mono: bool = True) -> np.ndarray:
    """从字节缓冲（如 HTTP 上传）解码音频并重采样到 sr，返回 float32 一维数组。

    不落盘，直接对 BytesIO 解码；与 load_audio 共用重采样逻辑。
    """
    import io

    audio, orig_sr = sf.read(io.BytesIO(buf), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1) if mono else audio.T
    else:
        if not mono:
            audio = audio[:, None].T
    audio = np.asarray(audio, dtype=np.float32)
    if orig_sr != sr:
        audio = resample_poly(audio, sr, orig_sr).astype(np.float32)
    return audio


def to_mono(audio: np.ndarray) -> np.ndarray:
    """确保一维单声道。"""
    if audio.ndim == 2:
        return audio.mean(axis=0) if audio.shape[0] > audio.shape[1] else audio.mean(axis=1)
    return audio


def peak_normalize(audio: np.ndarray, target_dbfs: float = -1.0) -> np.ndarray:
    """峰值归一化到目标 dBFS（默认 -1 dBFS，避免削波）。"""
    peak = np.max(np.abs(audio))
    if peak < 1e-8:
        return audio
    target = 10.0 ** (target_dbfs / 20.0)
    return (audio / peak) * target


def rms_db(audio: np.ndarray) -> float:
    """RMS 能量（dBFS）。"""
    rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2) + 1e-12)
    return 20.0 * np.log10(rms + 1e-12)


def db_to_gain(db: float) -> float:
    return 10.0 ** (db / 20.0)


def is_silence(audio: np.ndarray, thresh_db: float = -60.0) -> bool:
    """判断整段是否接近静音。"""
    return rms_db(audio) < thresh_db


def segment_audio(
    audio: np.ndarray, sr: int, segment_duration: float, hop_duration: float | None = None
) -> list[np.ndarray]:
    """将长音频切分为定长片段（末尾不足段丢弃）。

    Args:
        audio: 一维波形。
        sr: 采样率。
        segment_duration: 每段秒数。
        hop_duration: 步进秒数，默认等于 segment_duration（无重叠）。
    """
    seg_len = int(segment_duration * sr)
    hop = int((hop_duration or segment_duration) * sr)
    if len(audio) < seg_len:
        return [audio]
    return [audio[i : i + seg_len] for i in range(0, len(audio) - seg_len + 1, hop)]
