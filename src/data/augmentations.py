"""训练时特征增强（作用于 log-mel + 标签）。

- gain：整体增益扰动（不改标签）
- time_shift：循环平移（标签随之平移，保持对齐）
- freq_mask：频带掩蔽（不改标签）
- time_mask：时间掩蔽（被掩帧标签置 0，避免在人为静音段预测异常）
"""

from __future__ import annotations

import numpy as np


def aug_gain(mel: np.ndarray, rng: np.random.Generator, db: tuple = (-4.0, 4.0)) -> np.ndarray:
    g = 10.0 ** (rng.uniform(*db) / 20.0)
    return mel * g


def aug_time_shift(
    mel: np.ndarray, labels: np.ndarray, rng: np.random.Generator, max_frac: float = 0.1
) -> tuple[np.ndarray, np.ndarray]:
    T = mel.shape[1]
    shift = int(rng.integers(-int(T * max_frac), int(T * max_frac) + 1))
    if shift == 0:
        return mel, labels
    mel2 = np.roll(mel, shift, axis=1)
    labels2 = np.roll(labels, shift, axis=1)
    return mel2, labels2


def aug_freq_mask(
    mel: np.ndarray, rng: np.random.Generator, n_masks: int = 1, max_width: int = 20
) -> np.ndarray:
    mel = mel.copy()
    n_mels = mel.shape[0]
    for _ in range(n_masks):
        w = int(rng.integers(1, min(max_width, n_mels) + 1))
        s = int(rng.integers(0, n_mels - w + 1))
        mel[s : s + w, :] = 0.0
    return mel


def aug_time_mask(
    mel: np.ndarray, labels: np.ndarray, rng: np.random.Generator, n_masks: int = 1, max_width: int = 30
) -> tuple[np.ndarray, np.ndarray]:
    mel = mel.copy()
    labels = labels.copy()
    T = mel.shape[1]
    for _ in range(n_masks):
        w = int(rng.integers(1, min(max_width, T) + 1))
        s = int(rng.integers(0, T - w + 1))
        mel[:, s : s + w] = 0.0
        # Masking observations does not turn positive labels into clean labels.
    return mel, labels


def apply_augmentations(
    mel: np.ndarray, labels: np.ndarray, rng: np.random.Generator, p: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """以概率 p 应用组合增强。"""
    if rng.random() > p:
        return mel, labels
    mel = aug_gain(mel, rng)
    mel, labels = aug_time_shift(mel, labels, rng)
    mel = aug_freq_mask(mel, rng)
    mel, labels = aug_time_mask(mel, labels, rng)
    return mel, labels
