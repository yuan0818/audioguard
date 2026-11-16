"""片段级（整段）概率聚合：训练损失、验证评估与离线推理共用同一实现。

- mean           : 帧概率时间均值。短瞬态（一次爆音只占几帧）会被大量干净帧稀释，
                   旧 checkpoint 使用此方式（配置中无 clip_pooling 键时的默认值）。
- linear_softmax : sum(p^2) / sum(p)。由最自信的帧主导，又不像 max 那样只给单帧梯度；
                   声音事件检测（SED）弱标签聚合的常用选择。
- max            : 帧概率最大值。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping

import numpy as np

if TYPE_CHECKING:  # torch 仅在训练/推理路径需要；NumPy 聚合可独立导入
    import torch

POOLING_METHODS = ("mean", "linear_softmax", "max")


def clip_pooling_method(cfg: Mapping) -> str:
    method = cfg.get("train", {}).get("clip_pooling", "mean")
    if method not in POOLING_METHODS:
        raise ValueError(f"Unknown clip_pooling: {method}")
    return method


def clip_pool_np(probs: np.ndarray, method: str = "mean", axis: int = -1) -> np.ndarray:
    """帧概率 [..., T] -> 片段概率 [...]（沿 axis 聚合）。"""
    probs = np.asarray(probs, dtype=np.float64)
    if method == "mean":
        out = probs.mean(axis=axis)
    elif method == "linear_softmax":
        out = (probs * probs).sum(axis=axis) / np.maximum(probs.sum(axis=axis), 1e-7)
    elif method == "max":
        out = probs.max(axis=axis)
    else:
        raise ValueError(f"Unknown clip_pooling: {method}")
    return out.astype(np.float32)


def clip_pool_torch(probs: "torch.Tensor", mask: "torch.Tensor | None" = None,
                    method: str = "linear_softmax") -> "torch.Tensor":
    """帧概率 [B, C, T] (+ 有效帧 mask [B, T]) -> 片段概率 [B, C]。padding 帧不参与。"""
    import torch

    if mask is None:
        valid = torch.ones_like(probs[:, :1, :])
    else:
        valid = mask[:, None, :].to(probs)
    p = probs * valid
    if method == "mean":
        return p.sum(-1) / valid.sum(-1).clamp_min(1)
    if method == "linear_softmax":
        return (p * p).sum(-1) / p.sum(-1).clamp_min(1e-7)
    if method == "max":
        return p.amax(-1)
    raise ValueError(f"Unknown clip_pooling: {method}")
