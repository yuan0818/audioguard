"""Multi-label losses with explicit exclusion of padded frames."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F


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
    def __init__(self, pos_weight=None, seg_weight=1.0):
        super().__init__(pos_weight)
        self.seg_weight = seg_weight

    def forward(self, logits, targets, mask=None):
        frame = super().forward(logits, targets, mask)
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
        return FrameSegmentLoss(pos_weight, cfg["train"].get("seg_weight", 1.0))
    raise ValueError(f"unknown loss: {kind}")
