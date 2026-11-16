"""公开数据集下载与统一转换（真实干净音源 + 噪声源）。

用途：为异常合成提供真实干净载波（语音/音乐），并为 noise 类提供真实噪声底噪。

用法:
    python -m src.data.download_datasets --dataset esc50 --out data/raw
    python -m src.data.download_datasets --dataset librispeech_dev_clean --out data/raw
    python -m src.data.download_datasets --all --out data/raw     # 含大包，谨慎

转换规则：统一 -> 44.1kHz / mono / float32 wav，编号存放于 <out>/<name>_44k/xxxx.wav，
并生成 <out>/<name>_44k/manifest.csv（path,category,duration_s）。
下载支持断点续传（Range）；已转换文件跳过。
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import shutil
import sys
import tarfile
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SR_TARGET = 44100

# --------------------------------------------------------------------------- #
# 数据集注册表
# --------------------------------------------------------------------------- #
DATASETS = {
    "esc50": {
        "url": "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip",
        "file": "esc50.zip",
        "size_hint": "~600MB",
        "desc": "环境声 50 类（含空调/风扇/水/交通等噪声类，含 category 元数据）",
        "audio_dir": "ESC-50-master/audio",
        "meta_csv": "ESC-50-master/meta/esc50.csv",
        "cat_col": "category",
    },
    "librispeech_dev_clean": {
        "url": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
        "file": "dev-clean.tar.gz",
        "size_hint": "~330MB",
        "desc": "LibriSpeech dev-clean 干净朗读语音",
        "audio_dir": "LibriSpeech/dev-clean",
        "meta_csv": None,
        "cat_col": None,
    },
    "urbansound8k": {
        "url": "https://zenodo.org/records/1203745/files/UrbanSound8K.tar.gz",
        "file": "UrbanSound8K.tar.gz",
        "size_hint": "~5.6GB",
        "desc": "城市环境声 10 类（含噪声类），需注意下载大",
        "audio_dir": "UrbanSound8K/audio",
        "meta_csv": "UrbanSound8K/metadata/UrbanSound8K.csv",
        "cat_col": "class",
    },
    "musan": {
        "url": "https://www.openslr.org/resources/17/musan.tar.gz",
        "file": "musan.tar.gz",
        "size_hint": "~11GB",
        "desc": "MUSAN 噪声/语音/音乐语料（大型，可选）",
        "audio_dir": "musan",
        "meta_csv": None,
        "cat_col": None,
    },
}

AUDIO_EXTS = {".wav", ".flac", ".ogg", ".mp3", ".m4a"}


# --------------------------------------------------------------------------- #
# 下载 / 解压
# --------------------------------------------------------------------------- #
def download(url: str, dest: Path, chunk: int = 1 << 20) -> None:
    """断点续传下载（requests + certifi，避免 python.org 安装的 SSL 证书缺失问题）。"""
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    have = dest.stat().st_size if dest.exists() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    with requests.get(url, headers=headers, stream=True, timeout=60) as resp:
        if have and resp.status_code != 206:
            raise RuntimeError(f"服务器不支持断点续传 (status={resp.status_code})")
        total = have + int(resp.headers.get("Content-Length") or 0)
        with open(dest, "ab") as f, tqdm(
            total=total, initial=have, unit="B", unit_scale=True, desc=dest.name, leave=False
        ) as bar:
            for buf in resp.iter_content(chunk_size=chunk):
                if not buf:
                    continue
                f.write(buf)
                bar.update(len(buf))


def extract(archive: Path, workdir: Path) -> None:
    """按扩展名解压 tar/zip。"""
    if str(archive).endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            z.extractall(workdir)
    else:
        with tarfile.open(archive) as t:
            t.extractall(workdir)


# --------------------------------------------------------------------------- #
# 统一转换
# --------------------------------------------------------------------------- #
def _convert_one(args: tuple) -> tuple[str | None, str | None, float | None]:
    src, dst = args
    try:
        audio, sr = sf.read(str(src), dtype="float32", always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if sr != SR_TARGET:
            audio = resample_poly(audio, SR_TARGET, sr).astype(np.float32)
        audio = np.clip(audio, -1.0, 1.0)
        sf.write(str(dst), audio, SR_TARGET)
        return str(dst), None, float(len(audio) / SR_TARGET)
    except Exception as e:  # 单文件失败跳过
        return None, f"{src}: {e}", None


def convert_dataset(extract_dir: Path, out_dir: Path, name: str, spec: dict, workers: int = 4) -> list[dict]:
    """把解压后的原始音频转换为统一格式。返回 manifest 行（含 category）。

    extract_dir 是解压后的顶层目录（原归档根目录被重命名为此），
    audio_dir/meta_csv 取首段之后的相对路径。
    """
    rel_audio = spec["audio_dir"].split("/", 1)
    src_audio = extract_dir / (rel_audio[1] if len(rel_audio) > 1 else "")
    dst_dir = out_dir / f"{name}_44k"
    dst_dir.mkdir(parents=True, exist_ok=True)

    # category 映射（若有元数据 csv）
    cat_map: dict[str, str] = {}
    if spec.get("meta_csv"):
        rel_meta = spec["meta_csv"].split("/", 1)
        meta = extract_dir / (rel_meta[1] if len(rel_meta) > 1 else "")
        if meta.exists():
            with open(meta, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    fname = row.get("filename") or row.get("slice_file_name") or ""
                    cat_map[fname] = row[spec["cat_col"]]

    files = sorted(p for p in src_audio.rglob("*") if p.suffix.lower() in AUDIO_EXTS)
    jobs = []
    pending_rows = []
    manifest = []
    for i, f in enumerate(files):
        dst = dst_dir / f"{i:06d}.wav"
        # Preserve source provenance before conversion replaces original filenames.
        source_id = f"{name}:{f.stem.split('-')[1]}" if name == "esc50" else f"{name}:{f.relative_to(src_audio)}"
        if dst.exists():
            manifest.append({"path": str(dst.resolve()), "source_id": source_id, "category": cat_map.get(f.name, ""),
                             "duration_s": None})
            continue
        jobs.append((str(f), str(dst)))
        pending_rows.append(len(manifest))
        manifest.append({"path": str(dst.resolve()), "source_id": source_id, "category": cat_map.get(f.name, ""),
                         "duration_s": None})

    if jobs:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for (dst, err, dur), row in zip(tqdm(ex.map(_convert_one, jobs), total=len(jobs), desc=f"convert {name}"), (manifest[i] for i in pending_rows)):
                if err:
                    row["category"] = f"ERROR:{err}"
                row["duration_s"] = dur

    # 写 manifest
    manifest_path = dst_dir / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "source_id", "category", "duration_s"])
        w.writeheader()
        for row in manifest:
            w.writerow(row)
    return manifest


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="下载并转换公开数据集")
    ap.add_argument("--dataset", default=None, help=f"可选: {', '.join(DATASETS)}")
    ap.add_argument("--all", action="store_true", help="下载全部（含大包，谨慎）")
    ap.add_argument("--out", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    ap.add_argument("--no-convert", action="store_true", help="只下载解压，不转换")
    args = ap.parse_args()

    names = list(DATASETS) if args.all else ([args.dataset] if args.dataset else None)
    if names is None:
        ap.error("需指定 --dataset 或 --all")

    out_dir = Path(args.out)
    workdir = out_dir / "_downloads"
    workdir.mkdir(parents=True, exist_ok=True)

    for name in names:
        spec = DATASETS[name]
        print(f"\n===== {name}: {spec['desc']} ({spec['size_hint']}) =====")
        archive = workdir / spec["file"]
        if not archive.exists() or archive.stat().st_size == 0:
            download(spec["url"], archive)
        else:
            print(f"  archive 已存在: {archive}")
        extract_dir = workdir / name
        marker = extract_dir / ".extracted"
        if not marker.exists():
            extract(archive, workdir)
            # 找到实际解压根目录（如 ESC-50-master / LibriSpeech）
            actual = workdir / spec["audio_dir"].split("/")[0]
            if actual.exists() and actual != extract_dir:
                shutil.move(str(actual), str(extract_dir))
            marker.touch()
        if args.no_convert:
            continue
        manifest = convert_dataset(extract_dir, out_dir, name, spec, workers=args.workers)
        n_ok = sum(1 for r in manifest if not str(r["category"]).startswith("ERROR:"))
        print(f"  {name}: 转换完成 {n_ok}/{len(manifest)} -> {out_dir}/{name}_44k/")
    print("\n全部完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
