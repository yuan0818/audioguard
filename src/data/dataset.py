"""数据集：合成异常 + 真实音频混合。

AnomalyDataset 支持：
    source="synth"  : 用 generate_clean_audio 生成的干净音源 + 随机注入异常（自带标签）
    source="real"  : 从 data/raw 加载真实音频切分后注入异常（label 来自合成）
                     真实标注（带 label 文件）后续扩展。

返回 (feat[n_mels (+n_td), T], labels[n_classes, T]) 的 torch 张量。
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data import synth_anomalies as synth
from src.data.augmentations import apply_augmentations
from src.data.clean_sources import generate_clean_audio
from src.features.mel import TD_DB_SCALE, TD_LEVEL_ROWS, MelConfig, _mel_filterbank, compute_mel, gain_scale
from src.utils.audio import load_audio, segment_audio


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class AnomalyDataset(Dataset):
    """多标签 per-frame 异常检测数据集。

    Args:
        cfg: 完整配置 dict。
        split: "train" | "val"（仅影响是否启用增强）。
        source: "synth" | "real"。
        real_paths: source="real" 时，干净音源文件列表。
        size: 合成模式下样本数（每 epoch 即时生成 size 条）。
        augment: 是否启用增强（train=True）。
    """

    def __init__(
        self,
        cfg: Mapping,
        split: str = "train",
        source: str = "synth",
        real_paths: list[str] | None = None,
        size: int = 1024,
        augment: bool | None = None,
    ) -> None:
        self.cfg = cfg
        self.split = split
        self.source = source
        self.size = size
        self.augment = augment if augment is not None else (split == "train")

        sr = cfg["audio"]["sample_rate"]
        self.sr = sr
        self.seg_dur = float(cfg["audio"]["segment_duration"])
        self.synth_cfg = cfg["synth"]
        self.synth_cfg = {**self.synth_cfg, "sr": sr, "segment_duration": self.seg_dur}
        self.mel_cfg = MelConfig.from_cfg(cfg, sample_rate=sr)
        self._fb = _mel_filterbank(self.mel_cfg)

        self._real_paths = real_paths or []
        if source == "real" and not self._real_paths:
            raise ValueError("source='real' 需提供 real_paths")

        classes = list(cfg.get("classes", synth.CLASS_NAMES))
        self._aug_kwargs = {
            "n_mels": self.mel_cfg.n_mels,
            "params": cfg.get("train", {}).get("augment"),
            "mel_gain_scale": gain_scale(self.mel_cfg),
            "level_rows": TD_LEVEL_ROWS if self.mel_cfg.time_features else (),
            "level_scale": TD_DB_SCALE,
            "level_class": classes.index("loudness_anomaly") if "loudness_anomaly" in classes else None,
        }

        # 合成模式的确定性子种子由 __getitem__ 的 index 决定
        self._base_seed = int(cfg.get("seed", 42))

    def __len__(self) -> int:
        return self.size if self.source == "synth" else len(self._real_paths)

    def _get_clean(self, idx: int, rng: np.random.Generator) -> np.ndarray:
        if self.source == "synth":
            return generate_clean_audio(self.sr, self.seg_dur, rng)
        path = self._real_paths[idx % len(self._real_paths)]
        audio = load_audio(path, self.sr, mono=True)
        segs = segment_audio(audio, self.sr, self.seg_dur)
        if not segs:
            return generate_clean_audio(self.sr, self.seg_dur, rng)
        return segs[rng.integers(0, len(segs))]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        rng = np.random.default_rng(self._base_seed + idx * 7919 + (0 if self.split == "train" else 1))
        clean = self._get_clean(idx, rng)
        # 合成异常（即使 source=real 也注入异常以获得标签；真实标注后续替换）
        audio, labels, _ = synth.synthesize(clean, self.sr, self.synth_cfg, self.mel_cfg, rng)
        # 特征
        mel = compute_mel(audio, self.mel_cfg, fb=self._fb)
        if self.augment:
            mel, labels = apply_augmentations(mel, labels, rng, **self._aug_kwargs)
        # 长度对齐（防御性）
        T = min(mel.shape[1], labels.shape[1])
        mel = mel[:, :T]
        labels = labels[:, :T]
        return torch.from_numpy(np.ascontiguousarray(mel)), torch.from_numpy(np.ascontiguousarray(labels))


def load_manifest_paths(manifest_csv: str | Path) -> list[str]:
    """从 download_datasets.py 生成的 manifest.csv 读取可用音频路径（跳过错误行）。"""
    import csv

    paths = []
    with open(manifest_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cat = row.get("category", "")
            if cat.startswith("ERROR:"):
                continue
            paths.append(row["path"])
    return paths


def load_label_records(label_jsonl: str | Path) -> list[dict]:
    """读取 gradio 标注导出的 JSONL 记录列表。"""
    import json

    records = []
    with open(label_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


class LabeledDataset(Dataset):
    """由人工标注 JSONL 构建的数据集（真实录音 + 真实标签）。

    每条记录含整段 labels 和/或 time-range segments：
    - 每个 segment 裁剪为 (mel, one-hot 该类) 正样本
    - 整段 labels（无 segment 时）作为整段正样本
    - 未被标注为异常的其余部分（可选 clean_p）作为干净样本

    产出与 AnomalyDataset 一致的 (mel[n_mels,T], labels[n_classes,T])。
    """

    def __init__(self, cfg: Mapping, label_jsonl: str | Path | None = None, clean_p: float = 0.3, records=None) -> None:
        self.cfg = cfg
        self.sr = cfg["audio"]["sample_rate"]
        self.seg_dur = float(cfg["audio"]["segment_duration"])
        self.mel_cfg = MelConfig.from_cfg(cfg, sample_rate=self.sr)
        self._fb = _mel_filterbank(self.mel_cfg)
        self.class_names = list(cfg.get("classes", synth.CLASS_NAMES))
        self._items: list[tuple[np.ndarray, np.ndarray]] = []
        self._build(records if records is not None else load_label_records(label_jsonl))

    def _build(self, records) -> None:
        # Each recording is exhaustively annotated. Build all class masks jointly
        # before segmentation so overlapping labels and clean regions stay correct.
        for rec in records:
            audio = load_audio(rec["file"], self.sr, mono=True)
            if not len(audio):
                raise ValueError(f"Empty recording: {rec['file']}")
            masks = np.zeros((len(self.class_names), len(audio)), dtype=np.float32)
            for c in rec.get("labels", []):
                if c not in self.class_names:
                    raise ValueError(f"Unknown label: {c}")
                masks[self.class_names.index(c), :] = 1
            for seg in rec.get("segments", []):
                c = seg["class"]
                if c not in self.class_names:
                    raise ValueError(f"Unknown label: {c}")
                a, b = float(seg["start_s"]), float(seg["end_s"])
                if not 0 <= a < b <= len(audio) / self.sr + 1e-6:
                    raise ValueError(f"Invalid annotation range: {a}, {b}")
                masks[self.class_names.index(c), int(a * self.sr):int(b * self.sr)] = 1
            n = max(1, round(self.seg_dur * self.sr))
            for start in range(0, len(audio), n):
                chunk = audio[start:start + n]
                mel = compute_mel(chunk, self.mel_cfg, fb=self._fb)
                labels = np.zeros((len(self.class_names), mel.shape[1]), dtype=np.float32)
                for t in range(mel.shape[1]):
                    a = start + t * self.mel_cfg.hop_length
                    b = min(a + self.mel_cfg.n_fft, start + len(chunk))
                    labels[:, t] = masks[:, a:b].max(axis=1)
                self._items.append((mel, labels))

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int):
        mel, labels = self._items[idx]
        return torch.from_numpy(np.ascontiguousarray(mel)), torch.from_numpy(np.ascontiguousarray(labels))


def collate_variable(batch):
    """按 batch 中最大 T 填零对齐（支持变长片段）。"""
    mels, labels = zip(*batch)
    B = len(mels)
    n_mels = mels[0].shape[0]
    n_classes = labels[0].shape[0]
    T = max(m.shape[1] for m in mels)
    mel_b = torch.zeros(B, n_mels, T)
    lab_b = torch.zeros(B, n_classes, T)
    for i, (m, l) in enumerate(zip(mels, labels)):
        mel_b[i, :, : m.shape[1]] = m
        lab_b[i, :, : l.shape[1]] = l
    lengths = torch.tensor([m.shape[1] for m in mels], dtype=torch.long)
    return mel_b, lab_b, lengths
