"""评估：在数据集上跑推理并计算指标、调阈值。供训练与独立评估复用。

提供帧级与片段级两套指标：
- 帧级：逐帧判定（苛刻，瞬态类易混）。
- 片段级：帧概率按 clip_pooling（mean / linear_softmax / max）聚合为片段概率、帧标签 max 聚合为片段标签，
  即「整段判定」的产品语义指标。
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.pooling import clip_pool_np
from src.training.metrics import compute_metrics, format_metrics, tune_thresholds


def _cat_flat(p_list, y_list) -> tuple[np.ndarray, np.ndarray]:
    """把若干 [B,C,T] batch 展平为 [N,C] 拼接，返回 (probs, labels)。"""
    ps = [p.transpose(0, 2, 1).reshape(-1, p.shape[1]) for p in p_list]
    ys = [y.transpose(0, 2, 1).reshape(-1, y.shape[1]) for y in y_list]
    return np.concatenate(ps, axis=0), np.concatenate(ys, axis=0)


def _cat_clip(p_list, y_list, pooling: str = "mean") -> tuple[np.ndarray, np.ndarray]:
    """聚合到片段级：prob 按 pooling 沿时间聚合、label 沿时间 max -> [N_clips, C]。"""
    ps = [clip_pool_np(p, pooling, axis=2) for p in p_list]   # [B,C]
    ys = [y.max(axis=2) for y in y_list]    # [B,C]
    return np.concatenate(ps, axis=0), np.concatenate(ys, axis=0)


@torch.no_grad()
def collect_predictions_dual(model, dataloader, device, smooth_window=1, clip_pooling="mean"):
    from src.inference.streaming import smooth_probs
    model.eval()
    all_p, all_y = [], []
    for batch in dataloader:
        mel, labels = batch[:2]
        lengths = batch[2] if len(batch) == 3 else [mel.shape[-1]] * len(mel)
        mel = mel.to(device).float()
        logits, _ = model(mel.unsqueeze(1) if mel.dim() == 3 else mel)
        probs = torch.sigmoid(logits).cpu().numpy()
        for p, y, n in zip(probs, labels.numpy(), lengths):
            n = int(n)
            all_p.append(smooth_probs(p[:, :n], smooth_window)[None, ...])
            all_y.append(y[:, :n][None, ...])
    fp, fl = _cat_flat(all_p, all_y)
    cp, cl = _cat_clip(all_p, all_y, clip_pooling)
    return fp, fl, cp, cl


def collect_predictions(model, dataloader, device):
    fp, fl, _, _ = collect_predictions_dual(model, dataloader, device)
    return fp, fl


def evaluate_model(
    model,
    dataloader: DataLoader,
    device: torch.device,
    thresholds: np.ndarray | None = None,
    class_names: list[str] | None = None,
    clip_thresholds: np.ndarray | None = None,
    smooth_window: int = 1,
    clip_pooling: str = "mean",
) -> tuple[dict, np.ndarray, np.ndarray]:
    """评估帧级 + 片段级。

    thresholds=None -> 在帧级验证集调优；clip_thresholds=None -> 在片段级调优。
    返回 (metrics, frame_thresholds, clip_thresholds)。
    metrics 兼容旧格式：含 frame/clip 两组 + 顶层 per_class/macro 指向帧级（训练早停用）。
    """
    fp, fl, cp, cl = collect_predictions_dual(model, dataloader, device, smooth_window, clip_pooling)
    if thresholds is None:
        thresholds = tune_thresholds(fp, fl)
    if clip_thresholds is None:
        clip_thresholds = tune_thresholds(cp, cl)
    frame_metrics = compute_metrics(fp, fl, thresholds)
    clip_metrics = compute_metrics(cp, cl, clip_thresholds)
    metrics = {
        "frame": frame_metrics,
        "clip": clip_metrics,
        "per_class": frame_metrics["per_class"],
        "macro": frame_metrics["macro"],
    }
    return metrics, thresholds, clip_thresholds


def print_metrics(metrics: dict, class_names: list[str] | None = None) -> None:
    if "frame" in metrics:
        print("--- 帧级 ---")
        print(format_metrics(metrics["frame"], class_names))
        print("--- 片段级（整段判定）---")
        print(format_metrics(metrics["clip"], class_names))
    else:
        print(format_metrics(metrics, class_names))
