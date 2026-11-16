"""真实播放器录音人工标注界面（gradio）。

功能：
- 选择音频目录，逐个文件试听（波形 + 谱图）
- 7 类异常多标签勾选（整段）
- 可选：为某类别标注精确时间区间（起止秒），支持多段
- 保存为 JSONL：{file, labels, segments, annotator}

用法:
    python scripts/label_ui.py --dir data/recordings --out data/raw/labels.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import librosa
import librosa.display
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import gradio as gr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.synth_anomalies import CLASS_NAMES  # noqa: E402
from src.utils.audio import load_audio  # noqa: E402

AUDIO_EXTS = {".wav", ".flac", ".ogg", ".mp3", ".m4a"}


def list_files(directory: str) -> list[str]:
    d = Path(directory)
    if not d.is_dir():
        return []
    return sorted(str(p) for p in d.iterdir() if p.suffix.lower() in AUDIO_EXTS)


def load_labels(path: str) -> dict[str, dict]:
    p = Path(path)
    if not p.exists():
        return {}
    out = {}
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                out[rec["file"]] = rec
            except json.JSONDecodeError:
                continue
    return out


def save_label(path: str, rec: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    labels = load_labels(path)
    labels[rec["file"]] = rec
    with open(p, "w", encoding="utf-8") as f:
        for r in labels.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def make_spectrogram(file: str) -> str | None:
    """生成谱图 png 返回路径。"""
    if not file:
        return None
    try:
        audio, sr = librosa.load(file, sr=44100, mono=True, duration=30.0)
    except Exception:
        return None
    fig = plt.figure(figsize=(12, 3))
    librosa.display.specshow(
        librosa.amplitude_to_db(np.abs(librosa.stft(audio, n_fft=2048, hop_length=441)), ref=np.max),
        sr=sr, hop_length=441, x_axis="time", y_axis="hz", cmap="magma",
    )
    plt.colorbar(format="%+2.0f dB")
    plt.tight_layout()
    tmp = Path("/tmp") / "label_spec.png"
    fig.savefig(tmp, dpi=72)
    plt.close(fig)
    return str(tmp)


def refresh_filelist(directory: str) -> gr.Dropdown:
    files = list_files(directory)
    labeled = set(load_labels(GRAB["label_path"]).keys())
    choices = [(f"{'✓' if f not in labeled else '✓✓'} {Path(f).name}", f) for f in files]
    return gr.Dropdown(choices=choices, value=files[0] if files else None)


GRAB: dict = {}


def on_select(file: str):
    """选中文件 -> 显示谱图/音频/已有标签。"""
    if not file:
        return [None] * 5
    labels = load_labels(GRAB["label_path"]).get(file, {})
    prev_labels = labels.get("labels", [])
    prev_segs = labels.get("segments", [])
    segs_txt = "\n".join(f"{s['class']}: {s['start_s']:.2f}-{s['end_s']:.2f}s" for s in prev_segs)
    return [
        make_spectrogram(file),       # 谱图
        file,                          # 音频
        prev_labels,                   # 勾选
        prev_segs,                     # 状态
        segs_txt or "（无）",           # 段列表显示
    ]


def add_segment(segments, class_name, start_s, end_s, segs_display):
    if not class_name or end_s <= start_s:
        return segments, segs_display
    segs = list(segments or [])
    segs.append({"class": class_name, "start_s": round(float(start_s), 2), "end_s": round(float(end_s), 2)})
    txt = "\n".join(f"{s['class']}: {s['start_s']:.2f}-{s['end_s']:.2f}s" for s in segs)
    return segs, txt


def clear_segments():
    return [], "（无）"


def save_current(file, labels, segments, annotator):
    if not file:
        return "未选中文件", gr.Dropdown()
    rec = {"file": file, "labels": labels or [], "segments": segments or [], "annotator": annotator or "anonymous"}
    save_label(GRAB["label_path"], rec)
    total = len(list_files(GRAB["directory"]))
    done = len(load_labels(GRAB["label_path"]))
    return f"已保存 {Path(file).name}（{done}/{total}）", refresh_filelist(GRAB["directory"])


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="播放器音频异常标注") as app:
        gr.Markdown("# 播放器音频异常标注\n试听并勾选异常类别（可整段，也可为类别标注时间区间）")
        with gr.Row():
            directory = gr.Textbox(label="音频目录", value=GRAB["directory"])
            label_path = gr.Textbox(label="标注输出 (JSONL)", value=GRAB["label_path"])
            annotator = gr.Textbox(label="标注人", value="anonymous")
        file_dd = gr.Dropdown(label="文件（✓ 已标）", choices=[])
        with gr.Row():
            with gr.Column(scale=3):
                spect = gr.Image(label="谱图（前 30s）", interactive=False)
                audio = gr.Audio(label="试听")
            with gr.Column(scale=2):
                labels_gr = gr.CheckboxGroup(label="整段异常类别（多选）", choices=CLASS_NAMES)
                class_dd = gr.Dropdown(label="区间标注类别", choices=CLASS_NAMES, value=None)
                with gr.Row():
                    start_s = gr.Slider(0, 30, 0, step=0.1, label="开始(秒)")
                    end_s = gr.Slider(0, 30, 30, step=0.1, label="结束(秒)")
                with gr.Row():
                    add_btn = gr.Button("添加区间")
                    clear_btn = gr.Button("清空区间")
                segs_display = gr.Textbox(label="已标区间", lines=5, interactive=False)
                segs_state = gr.State([])
                save_btn = gr.Button("保存并下一个", variant="primary")
                status = gr.Textbox(label="状态", interactive=False)

        directory.change(lambda d: refresh_filelist(d), inputs=directory, outputs=file_dd)
        file_dd.change(on_select, inputs=file_dd,
                       outputs=[spect, audio, labels_gr, segs_state, segs_display])
        add_btn.click(add_segment, inputs=[segs_state, class_dd, start_s, end_s, segs_display],
                      outputs=[segs_state, segs_display])
        clear_btn.click(clear_segments, outputs=[segs_state, segs_display])
        save_btn.click(save_current, inputs=[file_dd, labels_gr, segs_state, annotator],
                       outputs=[status, file_dd])
        app.load(lambda: refresh_filelist(GRAB["directory"]), outputs=file_dd)
    return app


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "data" / "recordings"))
    ap.add_argument("--out", default=str(ROOT / "data" / "raw" / "labels.jsonl"))
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    GRAB["directory"] = args.dir
    GRAB["label_path"] = args.out
    Path(args.dir).mkdir(parents=True, exist_ok=True)
    app = build_ui()
    app.launch(server_port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
