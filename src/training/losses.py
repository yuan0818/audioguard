"""Multi-label losses with explicit exclusion of padded frames."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

from src.models.pooling import clip_pool_torch


def masked_mean(loss, mask=None):
    if mask is None:
        return loss.mean()
    weights = mask[:, None, :].to(loss).expand_as(loss)
    return (loss * weights).sum() / weights.sum().clamp_min(1)


class BCELoss(nn.Module):
    def __init__(self, pos_weight=None):
        super().__init__()
        self.register_buffer("pos_weight", None if pos_weight is None else pos_weight.reshape(1, -1, 1).float())

    def forward(self, logits, targets, mask=None):
        loss = F.binary_cross_entropy_with_logits(logits, targets.float(), pos_weight=self.pos_weight, reduction="none")
        return masked_mean(loss, mask)


class FocalLoss(BCELoss):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__(alpha)
        self.gamma = gamma

    def forward(self, logits, targets, mask=None):
        bce = F.binary_cross_entropy_with_logits(logits, targets.float(), pos_weight=self.pos_weight, reduction="none")
        p = torch.sigmoid(logits)
        pt = torch.where(targets > 0.5, p, 1 - p)
        return masked_mean(bce * (1 - pt) ** self.gamma, mask)


class FrameSegmentLoss(BCELoss):
    """帧级 BCE + 片段级弱监督。

    pooling="mean" 保留旧行为（时间均值 logit + 帧级 pos_weight），仅用于复现旧实验。
    其他 pooling 在概率上聚合（与评估/推理一致），片段项不再套用帧级 pos_weight：
    片段级正负比例与帧级完全不同，重复加权会把概率整体推高、阈值挤到 0.9+。
    """

    def __init__(self, pos_weight=None, seg_weight=1.0, pooling="mean"):
        super().__init__(pos_weight)
        self.seg_weight = seg_weight
        self.pooling = pooling

    def forward(self, logits, targets, mask=None):
        frame = super().forward(logits, targets, mask)
        if self.pooling != "mean":
            if mask is None:
                mask = torch.ones_like(logits[:, 0, :], dtype=torch.bool)
            clip_p = clip_pool_torch(torch.sigmoid(logits), mask, self.pooling)
            labels = (targets * mask[:, None, :].to(targets)).amax(-1)
            seg = F.binary_cross_entropy(clip_p.clamp(1e-6, 1 - 1e-6), labels.float())
            return frame + self.seg_weight * seg
        if mask is None:
            mask = torch.ones_like(logits[:, 0, :], dtype=torch.bool)
        valid = mask[:, None, :].to(logits)
        pooled = (logits * valid).sum(-1) / valid.sum(-1).clamp_min(1)
        labels = (targets * valid).amax(-1)
        pw = None if self.pos_weight is None else self.pos_weight.reshape(1, -1)
        return frame + self.seg_weight * F.binary_cross_entropy_with_logits(pooled, labels.float(), pos_weight=pw)


def build_loss(cfg, pos_weight=None):
    kind = cfg["train"]["loss"]
    if kind == "bce":
        return BCELoss(pos_weight)
    if kind == "focal":
        return FocalLoss(cfg["train"].get("focal_gamma", 2.0), pos_weight)
    if kind == "frame_seg":
        return FrameSegmentLoss(pos_weight, cfg["train"].get("seg_weight", 1.0),
                                cfg["train"].get("clip_pooling", "mean"))
    raise ValueError(f"unknown loss: {kind}")
