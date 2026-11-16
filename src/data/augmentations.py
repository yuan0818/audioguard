"""训练时特征增强（作用于归一化后的特征矩阵 + 帧标签）。

特征矩阵 [n_feat, T]：前 n_mels 行为 log-mel，其后（可选）为时域统计行。

- gain：整体电平偏移。log 域下增益是加性偏移（dB / norm_std），不是乘法。
        带 loudness_anomaly 标签的样本跳过（电平本身就是该类的判据）。
- start_crop：随机丢弃开头若干帧（替代 np.roll 循环平移，避免首尾拼接
        产生的人为断点——它看起来正是 clip_intermittent）。
- freq_mask：mel 频带掩蔽，用该频带自身的时间均值填充（不制造假静音/假陷波）。
- time_mask：默认关闭。人为时间空洞与 dropout/silence_loss 类过于相似；
        开启时同样用均值填充，标签不变。
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

DEFAULTS = {
    "p": 0.5,
    "gain_db": (-4.0, 4.0),
    "crop_max_frac": 0.1,
    "freq_masks": 1,
    "freq_mask_max": 20,
    "time_mask_p": 0.0,
    "time_mask_max": 10,
}


def aug_gain(feat: np.ndarray, rng: np.random.Generator, db: tuple, mel_scale: float,
             n_mels: int, level_rows: tuple = (), level_scale: float = 0.0) -> np.ndarray:
    """dB 增益 -> 归一化 log-mel 上的常数偏移；时域电平行同步偏移。"""
    g = float(rng.uniform(*db))
    feat = feat.copy()
    feat[:n_mels] += g * mel_scale
    for r in level_rows:
        feat[n_mels + r] += g * level_scale
    return feat


def aug_start_crop(feat: np.ndarray, labels: np.ndarray, rng: np.random.Generator,
                   max_frac: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    T = feat.shape[1]
    k_max = min(int(T * max_frac), T - 1)
    if k_max <= 0:
        return feat, labels
    k = int(rng.integers(0, k_max + 1))
    return feat[:, k:], labels[:, k:]


def aug_freq_mask(feat: np.ndarray, rng: np.random.Generator, n_mels: int,
                  n_masks: int = 1, max_width: int = 20) -> np.ndarray:
    feat = feat.copy()
    for _ in range(n_masks):
        w = int(rng.integers(1, min(max_width, n_mels) + 1))
        s = int(rng.integers(0, n_mels - w + 1))
        feat[s:s + w, :] = feat[s:s + w, :].mean(axis=1, keepdims=True)
    return feat


def aug_time_mask(feat: np.ndarray, rng: np.random.Generator, n_mels: int,
                  max_width: int = 10) -> np.ndarray:
    feat = feat.copy()
    T = feat.shape[1]
    w = int(rng.integers(1, min(max_width, T) + 1))
    s = int(rng.integers(0, T - w + 1))
    feat[:n_mels, s:s + w] = feat[:n_mels].mean(axis=1, keepdims=True)
    return feat


def apply_augmentations(
    feat: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
    p: float | None = None,
    *,
    n_mels: int | None = None,
    params: Mapping | None = None,
    mel_gain_scale: float = 1.0 / 20.0,
    level_rows: tuple = (),
    level_scale: float = 0.0,
    level_class: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """以概率 p 应用组合增强。

    mel_gain_scale: 1 dB 对应的特征偏移量（fixed 归一化为 1/norm_std；未归一化为 1；
                    逐段标准化时为 0，即增益无意义而跳过）。
    level_class: 电平敏感类别索引（loudness_anomaly），该类为正时不做增益。
    """
    cfg = {**DEFAULTS, **(params or {})}
    if p is None:
        p = cfg["p"]
    if rng.random() > p:
        return feat, labels
    n_mels = feat.shape[0] if n_mels is None else n_mels
    level_hit = level_class is not None and level_class < labels.shape[0] and labels[level_class].any()
    if mel_gain_scale and not level_hit:
        feat = aug_gain(feat, rng, tuple(cfg["gain_db"]), mel_gain_scale, n_mels, level_rows, level_scale)
    feat, labels = aug_start_crop(feat, labels, rng, cfg["crop_max_frac"])
    if cfg["freq_masks"]:
        feat = aug_freq_mask(feat, rng, n_mels, int(cfg["freq_masks"]), int(cfg["freq_mask_max"]))
    if cfg["time_mask_p"] and rng.random() < cfg["time_mask_p"]:
        feat = aug_time_mask(feat, rng, n_mels, int(cfg["time_mask_max"]))
    return feat, labels
