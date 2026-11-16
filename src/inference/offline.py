"""离线文件推理：对音频文件/目录输出每类异常的时间轴标注（JSON）。

复用 StreamingDetector 引擎（与流式同一权重、同一特征），整段喂入后聚合为时间轴段落。

用法:
    python -m src.inference.offline --checkpoint checkpoints/best.pt --input file.wav
    python -m src.inference.offline --checkpoint checkpoints/best.pt --input dir/ --out results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from src.features.mel import MelConfig, compute_mel
from src.inference.streaming import StreamingDetector, probs_to_segments, smooth_probs
from src.models.crnn import build_model
from src.models.pooling import clip_pool_np, clip_pooling_method
from src.utils.audio import load_audio
from src.utils.config import load_config_resolved

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def read_checkpoint(path, device):
    # Narrow compatibility with local legacy NumPy threshold arrays, while
    # retaining PyTorch's restricted loader (never arbitrary pickle execution).
    import importlib
    reconstruct = importlib.import_module("numpy._core.multiarray" if np.__version__.split(".")[0] != "1" else "numpy.core.multiarray")._reconstruct
    allowed = [reconstruct, np.ndarray, np.dtype, type(np.dtype("float32")), type(np.dtype("float64"))]
    with torch.serialization.safe_globals(allowed):
        return torch.load(path, map_location=device, weights_only=True)


def load_checkpoint(path: str, device: torch.device):
    state = read_checkpoint(path, device)
    cfg = state["config"]
    model = build_model(cfg).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    thresholds = state.get("thresholds")
    clip_thresholds = state.get("clip_thresholds")
    if thresholds is not None:
        thresholds = np.asarray(thresholds, dtype=np.float32)
    if clip_thresholds is not None:
        clip_thresholds = np.asarray(clip_thresholds, dtype=np.float32)
    return cfg, model, thresholds, clip_thresholds


def infer_audio(audio: np.ndarray, sr: int, cfg: dict, model, thresholds, device,
                clip_thresholds=None, mel_cfg=None, chunk_samples: int | None = None) -> dict:
    """对一维音频数组做异常检测，返回与 infer_file 相同结构的 dict（无 file 字段）。

    服务层直接调用本函数，避免落盘。
    """
    if mel_cfg is None:
        mel_cfg = MelConfig.from_cfg(cfg, sample_rate=sr)
    if audio.ndim != 1 or audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError("Expected nonempty finite mono audio")
    if sr != cfg["audio"]["sample_rate"]:
        raise ValueError("Audio must be resampled to the checkpoint sample rate")
    if mel_cfg.normalize and mel_cfg.normalization == "clip":
        # Preserve legacy preprocessing exactly, without pretending it is causal.
        mel = torch.from_numpy(compute_mel(audio, mel_cfg))[None, None].to(device)
        with torch.no_grad():
            probs = torch.sigmoid(model(mel)[0])[0].cpu().numpy()
        # Legacy thresholds were calibrated without smoothing.
    else:
        det = StreamingDetector.from_cfg(model, cfg, thresholds=thresholds, device=device)
        chunk = chunk_samples or int(sr * 0.2)
        if chunk <= 0:
            raise ValueError("chunk_samples must be positive")
        for i in range(0, len(audio), chunk):
            det.push_samples(audio[i:i + chunk])
        det.flush()
        probs = det.all_probs()
    thresholds = np.full(model.n_classes, 0.5) if thresholds is None else np.asarray(thresholds)
    preds = (probs >= thresholds.reshape(-1, 1)).astype(np.int32)
    min_ms = cfg.get("inference", {}).get("offline", {}).get("min_segment_ms", 200)
    segs = probs_to_segments(preds, mel_cfg, min_segment_ms=min_ms)

    # 片段级整段判定：帧概率按训练时同一 pooling 聚合 -> 每类一个片段概率，按片段阈值判
    clip_probs = clip_pool_np(probs, clip_pooling_method(cfg), axis=1)  # [n_classes]
    if clip_thresholds is not None:
        clip_pred = clip_probs >= np.asarray(clip_thresholds)
    else:
        clip_pred = clip_probs >= 0.5
    class_names = cfg.get("classes", [])
    clip_classes = [class_names[c] for c in range(len(clip_pred)) if clip_pred[c]]
    for s in segs:
        s["class_name"] = class_names[s["class_id"]] if s["class_id"] < len(class_names) else str(s["class_id"])
    return {
        "duration_s": round(len(audio) / sr, 3),
        "n_frames": int(preds.shape[1]),
        "anomaly_segments": segs,
        "n_segments": len(segs),
        "is_anomaly": len(clip_classes) > 0,
        "clip_level": {"classes": clip_classes,
                       "probs": {class_names[c]: round(float(clip_probs[c]), 3) for c in range(len(clip_probs))}},
    }


def infer_file(path: str, cfg: dict, model, thresholds, device,
               clip_thresholds=None, chunk_samples: int | None = None) -> dict:
    sr = cfg["audio"]["sample_rate"]
    audio = load_audio(path, sr, mono=True)
    mel_cfg = MelConfig.from_cfg(cfg, sample_rate=sr)
    r = infer_audio(audio, sr, cfg, model, thresholds, device,
                    clip_thresholds=clip_thresholds, mel_cfg=mel_cfg, chunk_samples=chunk_samples)
    r["file"] = str(path)
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--input", required=True, help="音频文件或目录")
    ap.add_argument("--out", default=None, help="输出 JSON 路径")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg, model, thresholds, clip_thresholds = load_checkpoint(args.checkpoint, device)

    src = Path(args.input)
    files = []
    if src.is_dir():
        files = sorted([p for p in src.iterdir() if p.suffix.lower() in {".wav", ".flac", ".ogg", ".mp3"}])
    else:
        files = [src]

    results = []
    for f in files:
        r = infer_file(str(f), cfg, model, thresholds, device, clip_thresholds=clip_thresholds)
        tag = "ANOMALY" if r["is_anomaly"] else "clean"
        clip_str = ",".join(r["clip_level"]["classes"]) or "none"
        print(f"[{tag}] {f.name}  ({r['duration_s']}s, {r['n_segments']} segs)  clip=[{clip_str}]")
        for s in r["anomaly_segments"]:
            print(f"    {s['class_name']:18s} {s['start_s']:7.2f}-{s['end_s']:7.2f}s")
        results.append(r)

    out_path = Path(args.out) if args.out else (ROOT / "results.json")
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved {len(results)} result(s) -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
