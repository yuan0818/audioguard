"""误例分析：在验证集上找出高置信误判样本，绘制 mel 谱图 + 预测/真实时间轴。

用法:
    python -m src.evaluation.analyze_errors --checkpoint checkpoints/best.pt --out logs/errors
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from src.data.dataset import AnomalyDataset, collate_variable
from src.evaluation.evaluate import collect_predictions
from src.features.mel import MelConfig
from src.models.crnn import build_model
from src.inference.offline import load_checkpoint
from src.utils.config import load_config_resolved

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default=str(ROOT / "logs" / "errors"))
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib 未安装，跳过可视化")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, model, thresholds, _ = load_checkpoint(args.checkpoint, device)
    if thresholds is None:
        thresholds = np.full(cfg["model"]["n_classes"], 0.5)

    from src.training.train import build_loaders
    _, ds, _, _ = build_loaders(cfg)
    mel_cfg = MelConfig.from_cfg(cfg)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    plotted = 0
    for idx in range(len(ds)):
        if plotted >= args.n:
            break
        mel, labels = ds[idx]
        mel = mel.unsqueeze(0).unsqueeze(0).to(device).float()
        with torch.no_grad():
            logits, _ = model(mel)
        probs = torch.sigmoid(logits)[0].cpu().numpy()
        labels = labels.numpy()
        pred = (probs >= thresholds.reshape(-1, 1)).astype(np.int32)
        # 选有误判的样本
        fp = ((pred == 1) & (labels == 0)).sum()
        fn = ((pred == 0) & (labels == 1)).sum()
        if fp + fn == 0:
            continue
        fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
        axes[0].imshow(mel[0, 0].cpu().numpy(), aspect="auto", origin="lower", cmap="magma")
        axes[0].set_title(f"sample {idx}  FP={fp} FN={fn}")
        axes[0].set_ylabel("mel bin")
        t = np.arange(probs.shape[1])
        axes[1].plot(t, labels.T, alpha=0.4, lw=1)
        axes[1].plot(t, probs.T, lw=1.2)
        axes[1].set_ylabel("prob / label")
        axes[1].set_xlabel("frame")
        axes[1].legend(cfg["classes"], fontsize=6, ncol=4, loc="upper right")
        fig.tight_layout()
        fig.savefig(out_dir / f"error_{idx:03d}.png", dpi=80)
        plt.close(fig)
        plotted += 1

    print(f"Saved {plotted} error plots -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
