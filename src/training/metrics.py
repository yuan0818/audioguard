"""逐帧多标签评估指标与阈值调优。

所有计算在 [N, C] 的展平概率/标签上进行（N = sum(batch*T)）。
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def flatten(probs: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[B,C,T] -> [N,C]。"""
    p = probs.transpose(0, 2, 1).reshape(-1, probs.shape[1])
    y = labels.transpose(0, 2, 1).reshape(-1, labels.shape[1])
    return p, y


def tune_thresholds(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """对每个类别在验证集上选取使 F1 最大的阈值。返回 [n_classes]。"""
    n_classes = probs.shape[1]
    th = np.full(n_classes, 0.5, dtype=np.float32)
    for c in range(n_classes):
        y = labels[:, c]
        if y.sum() == 0 or y.sum() == len(y):
            continue
        try:
            prec, rec, prc = precision_recall_curve(y, probs[:, c])
            f1 = 2 * prec * rec / (prec + rec + 1e-9)
            best = int(np.argmax(f1))
            # precision_recall_curve 末尾阈值是任意大，对应 recall=0；取稳健点
            th[c] = float(prc[min(best, len(prc) - 1)]) if best < len(prc) else 0.5
        except Exception:
            th[c] = 0.5
    return np.clip(th, 0.01, 0.99)


def compute_metrics(probs: np.ndarray, labels: np.ndarray, thresholds: np.ndarray) -> dict:
    """返回 per-class 与 macro 指标。

    Args:
        probs: [N, C] 概率（已 sigmoid）。
        labels: [N, C] 0/1。
        thresholds: [C] 阈值。
    """
    n_classes = probs.shape[1]
    pred = (probs >= thresholds[None, :]).astype(np.int32)
    out: dict = {"per_class": {}, "macro": {}}

    aps, aucs, f1s, ps, rs = [], [], [], [], []
    for c in range(n_classes):
        y = labels[:, c].astype(np.int32)
        yp = pred[:, c]
        f1 = f1_score(y, yp, zero_division=0)
        p = precision_score(y, yp, zero_division=0)
        r = recall_score(y, yp, zero_division=0)
        try:
            ap = average_precision_score(y, probs[:, c]) if y.sum() > 0 else float("nan")
        except Exception:
            ap = float("nan")
        try:
            auc = roc_auc_score(y, probs[:, c]) if 0 < y.sum() < len(y) else float("nan")
        except Exception:
            auc = float("nan")
        out["per_class"][c] = {"precision": p, "recall": r, "f1": f1, "ap": ap, "auc": auc, "threshold": float(thresholds[c])}
        if not np.isnan(ap):
            aps.append(ap)
        if not np.isnan(auc):
            aucs.append(auc)
        f1s.append(f1); ps.append(p); rs.append(r)

    out["macro"] = {
        "f1": float(np.mean(f1s)),
        "precision": float(np.mean(ps)),
        "recall": float(np.mean(rs)),
        "mAP": float(np.mean(aps)) if aps else float("nan"),
        "AUC": float(np.mean(aucs)) if aucs else float("nan"),
    }
    return out


def format_metrics(metrics: dict, class_names: list[str] | None = None) -> str:
    lines = ["Macro: F1={f1:.3f} P={precision:.3f} R={recall:.3f} mAP={mAP:.3f} AUC={AUC:.3f}".format(**metrics["macro"])]
    for c, m in metrics["per_class"].items():
        name = class_names[c] if class_names else f"class{c}"
        lines.append(
            f"  [{c}] {name:18s} F1={m['f1']:.3f} P={m['precision']:.3f} R={m['recall']:.3f} "
            f"AP={m['ap']:.3f} AUC={m['auc']:.3f} th={m['threshold']:.2f}"
        )
    return "\n".join(lines)
