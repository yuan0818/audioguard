"""干净音源生成器（无 torch 依赖，便于数据管线独立验证）。

合成「干净」音频作为无真实数据时的 bootstrap 源。非真实音源，仅用于管线自举与
最小验证；真实训练应替换为公开数据集（见 download_datasets.py）。
"""

from __future__ import annotations

import numpy as np


def generate_clean_audio(
    sr: int, duration: float, rng: np.random.Generator
) -> np.ndarray:
    """生成一段合成「干净」音频：基频 + 谐波 + 缓变包络，模拟语音/音乐载波。"""
    n = int(sr * duration)
    t = np.arange(n) / sr
    f0 = float(rng.uniform(120.0, 300.0))
    signal = np.zeros(n, dtype=np.float32)
    for k in range(1, 6):
        amp = 1.0 / k
        vibr = 1.0 + 0.01 * np.sin(2 * np.pi * 5.0 * t)
        signal += amp * np.sin(2 * np.pi * f0 * k * vibr * t)
    env = np.exp(-t / (duration * 0.6))
    tremolo = 0.75 + 0.25 * np.sin(2 * np.pi * 3.0 * t)
    signal = signal * env * tremolo
    signal = signal / (np.max(np.abs(signal)) + 1e-9) * 0.7
    return signal.astype(np.float32)
