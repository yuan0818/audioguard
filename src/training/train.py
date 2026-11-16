"""训练循环：配置驱动，含 pos_weight 估计、早停、断点续训、验证评估。

用法:
    python -m src.training.train --config configs/default.yaml
    python -m src.training.train --config configs/default.yaml --resume checkpoints/last.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.data import synth_anomalies as synth
from src.data.dataset import AnomalyDataset, collate_variable
from src.evaluation.evaluate import evaluate_model, print_metrics
from src.models.crnn import build_model
from src.training.losses import build_loss
from src.utils.config import load_config_resolved, set_seed, ensure_dirs

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def estimate_pos_weight(dataset: AnomalyDataset, n_batches: int = 20, batch_size: int = 16) -> torch.Tensor:
    """采样若干 batch 估计正负样本比，pos_weight = neg/pos per class。"""
    n_classes = len(synth.CLASS_NAMES)
    pos = np.zeros(n_classes, dtype=np.float64)
    neg = np.zeros(n_classes, dtype=np.float64)
    idxs = np.random.default_rng(0).integers(0, len(dataset), size=n_batches * batch_size)
    for i in idxs:
        _, labels = dataset[int(i)]
        arr = labels.numpy()
        pos += arr.sum(axis=1)
        neg += (1 - arr).sum(axis=1)
    pw = (neg + 1) / (pos + 1)
    return torch.tensor(pw, dtype=torch.float32)


def partition_sources(records, seed, val_ratio=0.2, path_key="path"):
    """Keep recordings, declared source groups and byte-identical files together."""
    import hashlib
    groups, seen_content = {}, {}
    for rec in records:
        path = Path(rec[path_key]).resolve()
        key = str(rec.get("source_id") or path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        # Union explicit provenance with duplicate content across records.
        groups.setdefault(key, key)
        def root(k):
            while groups[k] != k:
                k = groups[k]
            return k
        if digest in seen_content:
            groups[root(key)] = root(seen_content[digest])
        seen_content[digest] = key
        rec["_group"] = key
    for rec in records:
        rec["_group"] = root(rec["_group"])
    keys = sorted({r["_group"] for r in records})
    if len(keys) < 2:
        raise ValueError("Need at least two independent source groups")
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between zero and one")
    np.random.default_rng(seed).shuffle(keys)
    n_val = min(len(keys) - 1, max(1, round(len(keys) * val_ratio)))
    selected = set(keys[:n_val])
    return ([r for r in records if root(r["_group"]) not in selected],
            [r for r in records if root(r["_group"]) in selected])


def build_loaders(cfg: dict):
    import csv
    from src.data.dataset import LabeledDataset, load_label_records
    source = cfg.get("data", {}).get("source", "synth")
    data = cfg.get("data", {})
    if source in {"real", "labeled"}:
        if source == "real":
            records = []
            for manifest in data.get("real_manifests", []):
                with open(manifest, newline="", encoding="utf-8") as f:
                    records.extend(r for r in csv.DictReader(f) if not r.get("category", "").startswith("ERROR:"))
            if not records:
                raise ValueError("source=real requires nonempty real_manifests")
            key = "path"
        else:
            records = load_label_records(data["labeled_jsonl"])
            key = "file"
        tr, va = partition_sources(records, cfg.get("seed", 42), data.get("val_ratio", 0.2), key)
        cfg["split_manifest"] = {"method": "source_group_v1",
                                 "train_groups": sorted({r["_group"] for r in tr}),
                                 "val_groups": sorted({r["_group"] for r in va})}
        if source == "labeled":
            train_ds, val_ds = LabeledDataset(cfg, records=tr), LabeledDataset(cfg, records=va)
        else:
            train_ds = AnomalyDataset(cfg, split="train", source="real", real_paths=[r[key] for r in tr])
            val_ds = AnomalyDataset(cfg, split="val", source="real", real_paths=[r[key] for r in va], augment=False)
    elif source == "synth":
        cfg["split_manifest"] = {"method": "synthetic_seed_index_v1", "seed": cfg.get("seed", 42),
                                 "train_size": cfg["train"].get("train_size", 2048),
                                 "val_size": cfg["train"].get("val_size", 256)}
        train_ds = AnomalyDataset(cfg, split="train", size=cfg["train"].get("train_size", 2048))
        val_ds = AnomalyDataset(cfg, split="val", size=cfg["train"].get("val_size", 256), augment=False)
    else:
        raise ValueError(f"Unknown data source: {source}")
    nw = cfg["train"].get("num_workers", 4)
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
                             num_workers=nw, collate_fn=collate_variable)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
                           num_workers=nw, collate_fn=collate_variable)
    return train_ds, val_ds, train_loader, val_loader


def save_checkpoint(state: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = dict(state)
    for key in ("thresholds", "clip_thresholds"):
        if isinstance(state.get(key), np.ndarray):
            state[key] = state[key].tolist()
    torch.save(state, str(path))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--resume", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config_resolved(args.config, root=ROOT)
    set_seed(int(cfg.get("seed", 42)))
    ensure_dirs(cfg)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    train_ds, val_ds, train_loader, val_loader = build_loaders(cfg)
    print(f"Train size={len(train_ds)}  Val size={len(val_ds)}")

    model = build_model(cfg).to(device)
    print(f"Model params: {model.n_parameters()/1e6:.2f}M")

    pw = cfg["train"].get("pos_weight")
    if pw is None:
        pw_tensor = estimate_pos_weight(train_ds)
        print(f"Estimated pos_weight: {pw_tensor.numpy().round(2).tolist()}")
    else:
        pw_tensor = torch.tensor(pw, dtype=torch.float32)
    loss_fn = build_loss(cfg, pos_weight=pw_tensor).to(device)

    tr = cfg["train"]
    if tr["optimizer"] == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])
    else:
        opt = torch.optim.Adam(model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])
    sched = (ReduceLROnPlateau(opt, patience=3, factor=0.5)
             if tr["scheduler"] == "plateau"
             else CosineAnnealingLR(opt, T_max=tr["epochs"]))

    start_epoch = 0
    best_f1 = -1.0
    patience = 0
    ckpt_dir = Path(cfg["paths"]["checkpoints"])

    if args.resume:
        from src.inference.offline import read_checkpoint
        state = read_checkpoint(args.resume, device)
        model.load_state_dict(state["model"])
        if state["config"].get("split_manifest") != cfg.get("split_manifest"):
            raise ValueError("Resume requires the original recorded source split")
        if state["config"]["features"] != cfg["features"]:
            raise ValueError("Resume feature config differs from checkpoint; start a new training run")
        opt.load_state_dict(state["optimizer"])
        if "scheduler" in state:
            sched.load_state_dict(state["scheduler"])
        start_epoch = state.get("epoch", 0)
        best_f1 = state.get("best_f1", -1.0)
        print(f"Resumed from {args.resume} @ epoch {start_epoch}")

    epochs = args.epochs or tr["epochs"]
    for epoch in range(start_epoch, epochs):
        model.train()
        running = 0.0
        for mel, labels, lengths in train_loader:
            mel = mel.to(device).float().unsqueeze(1)
            labels = labels.to(device).float()
            opt.zero_grad()
            logits, _ = model(mel)
            mask = torch.arange(labels.shape[-1], device=device)[None, :] < lengths.to(device)[:, None]
            loss = loss_fn(logits, labels, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tr["clip_grad"])
            opt.step()
            running += loss.item()

        if isinstance(sched, ReduceLROnPlateau):
            sched.step(running / len(train_loader))
        else:
            sched.step()

        metrics, thresholds, clip_thresholds = evaluate_model(
            model, val_loader, device, thresholds=None, class_names=cfg["classes"],
            smooth_window=cfg.get("inference", {}).get("streaming", {}).get("smooth_window", 1))
        f1 = metrics["macro"]["f1"]          # 帧级 macro F1（早停依据）
        cf1 = metrics["clip"]["macro"]["f1"]
        print(f"\n[Epoch {epoch+1}/{epochs}] loss={running/len(train_loader):.4f}  "
              f"frameF1={f1:.4f}  clipF1={cf1:.4f}")
        print_metrics(metrics, cfg["classes"])

        if f1 > best_f1:
            best_f1 = f1
            patience = 0
            save_checkpoint(
                {"model": model.state_dict(), "optimizer": opt.state_dict(),
                 "epoch": epoch + 1, "best_f1": best_f1, "thresholds": thresholds,
                 "clip_thresholds": clip_thresholds, "config": cfg, "scheduler": sched.state_dict()},
                ckpt_dir / "best.pt")
        else:
            patience += 1
            if patience >= tr["early_stop_patience"]:
                print(f"Early stopping @ epoch {epoch+1}")
                break

        save_checkpoint(
            {"model": model.state_dict(), "optimizer": opt.state_dict(),
             "epoch": epoch + 1, "best_f1": best_f1, "thresholds": thresholds,
             "clip_thresholds": clip_thresholds, "config": cfg, "scheduler": sched.state_dict()},
            ckpt_dir / "last.pt")

    print(f"\nTraining done. Best macro F1 = {best_f1:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
