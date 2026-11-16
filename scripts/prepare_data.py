"""生成最小合成数据样本，验证数据管线端到端可用。

用法:
    python scripts/prepare_data.py --config configs/default.yaml \
        --n-samples 200 --out data/synth

对每个样本：合成干净音源 -> 随机注入 0..max_classes 类异常 ->
计算 log-mel + per-frame 多标签 -> 保存为 .npz（mel, labels, chosen）。

输出统计：各类别正帧占比、样本数、mel 形状。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# 让脚本可直接运行（无需安装包）
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.clean_sources import generate_clean_audio  # noqa: E402
from src.data import synth_anomalies as synth  # noqa: E402
from src.features.mel import MelConfig, _mel_filterbank, compute_mel  # noqa: E402
from src.utils.config import load_config_resolved, set_seed, ensure_dirs  # noqa: E402
from src.utils.audio import peak_normalize  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--n-samples", type=int, default=200)
    ap.add_argument("--out", default=str(ROOT / "data" / "synth"))
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config_resolved(args.config, root=ROOT)
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 42))
    set_seed(seed)
    rng = np.random.default_rng(seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ensure_dirs(cfg)

    sr = cfg["audio"]["sample_rate"]
    seg_dur = float(cfg["audio"]["segment_duration"])
    mel_cfg = MelConfig.from_cfg(cfg, sample_rate=sr)
    fb = _mel_filterbank(mel_cfg)
    synth_cfg = {**cfg["synth"], "sr": sr, "segment_duration": seg_dur}

    # per-class 正帧累计
    pos_frames = np.zeros(len(synth.CLASS_NAMES), dtype=np.float64)
    total_frames = 0
    chosen_counts = {n: 0 for n in synth.CLASS_NAMES}
    n_clean = 0
    shapes = set()

    for i in range(args.n_samples):
        clean = generate_clean_audio(sr, seg_dur, rng)
        clean = peak_normalize(clean, target_dbfs=-3.0)
        audio, labels, chosen = synth.synthesize(clean, sr, synth_cfg, mel_cfg, rng)
        mel = compute_mel(audio, mel_cfg, fb=fb)
        # 对齐
        T = min(mel.shape[1], labels.shape[1])
        mel = mel[:, :T]
        labels = labels[:, :T]
        shapes.add(mel.shape)

        np.savez_compressed(
            out_dir / f"sample_{i:05d}.npz",
            mel=mel.astype(np.float32),
            labels=labels.astype(np.float32),
            chosen=",".join(chosen),
            sr=sr,
        )
        pos_frames += labels.sum(axis=1)
        total_frames += T
        if not chosen:
            n_clean += 1
        for c in chosen:
            chosen_counts[c] += 1

    print("\n===== 合成数据准备完成 =====")
    print(f"输出目录     : {out_dir}")
    print(f"样本总数     : {args.n_samples}（其中干净样本 {n_clean}）")
    print(f"mel 形状      : {shapes}")
    print(f"总帧数       : {total_frames}")
    print("\n各类别注入次数（按样本计）:")
    for name, cnt in chosen_counts.items():
        print(f"  {name:20s}: {cnt}")
    print("\n各类别正帧占比:")
    if total_frames > 0:
        for i, name in enumerate(synth.CLASS_NAMES):
            ratio = pos_frames[i] / total_frames
            print(f"  {name:20s}: {ratio*100:6.2f}%  ({int(pos_frames[i])} frames)")
    print("\n下一步: 运行 tests/test_synth.py 与 tests/test_model.py 验证管线。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
