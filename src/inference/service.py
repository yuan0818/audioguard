"""音频异常检测 HTTP 服务（FastAPI + 线程池并发）。

启动时 CPU 加载模型一次，eval + no_grad 下 forward 线程安全；
请求由固定数量的工作线程执行。

接口:
    POST /detect   multipart file=<音频字节>  -> 整段判定 + 时间轴段落 JSON
    GET  /health   健康检查
    GET  /metrics  QPS / 平均延迟 / 在途请求数

用法:
    python -m src.inference.service --checkpoint checkpoints/best.pt \
        --host 0.0.0.0 --port 8000 --workers 8
    curl -F "file=@data/test_sample.wav" http://localhost:8000/detect
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from src.features.mel import MelConfig
from src.inference.offline import infer_audio, load_checkpoint
from src.utils.audio import decode_audio

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

AUDIO_EXTS = {".wav", ".flac", ".ogg", ".mp3", ".m4a"}


class AppState:
    """服务运行态：模型/配置/线程池/指标计数器。"""

    def __init__(self) -> None:
        self.model = None
        self.cfg = None
        self.mel_cfg = None
        self.thresholds = None
        self.clip_thresholds = None
        self.device = None
        self.pool: ThreadPoolExecutor | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        # 指标
        self.n_total = 0
        self.n_inflight = 0
        self.latency_sum = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动
    state = app.state.runtime
    state.loop = asyncio.get_running_loop()
    yield
    # 关闭
    if state.pool is not None:
        state.pool.shutdown(wait=True)


def build_app(checkpoint: str, workers: int, num_threads: int) -> FastAPI:
    app = FastAPI(title="AudioGuard", version="0.1.0", lifespan=lifespan)
    state = AppState()
    app.state.runtime = state

    device = torch.device("cpu")
    torch.set_num_threads(num_threads)
    cfg, model, thresholds, clip_thresholds = load_checkpoint(checkpoint, device)
    model.eval()
    sr = cfg["audio"]["sample_rate"]
    state.model = model
    state.cfg = cfg
    state.thresholds = thresholds
    state.clip_thresholds = clip_thresholds
    state.mel_cfg = MelConfig.from_cfg(cfg, sample_rate=sr)
    state.device = device
    state.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="infer")

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "model_loaded": state.model is not None,
            "device": str(state.device),
            "workers": state.pool._max_workers if state.pool else 0,
        }

    @app.get("/metrics")
    async def metrics():
        n = max(state.n_total, 1)
        return {
            "requests_total": state.n_total,
            "requests_in_flight": state.n_inflight,
            "avg_latency_ms": round(state.latency_sum / n * 1000, 2) if state.n_total else 0.0,
        }

    @app.post("/detect")
    async def detect(file: UploadFile = File(...)):
        if state.model is None:
            raise HTTPException(503, "model not loaded")
        name = Path(file.filename or "").suffix.lower()
        if name and name not in AUDIO_EXTS:
            raise HTTPException(415, f"unsupported file type: {name}")
        buf = await file.read()
        if not buf:
            raise HTTPException(400, "empty file")

        sr = state.cfg["audio"]["sample_rate"]
        loop = state.loop or asyncio.get_running_loop()
        # 解码 + 推理都在线程池中执行（CPU 密集，释放事件循环）
        state.n_inflight += 1
        t0 = time.perf_counter()
        try:
            result = await loop.run_in_executor(state.pool, _run_inference, state, buf, sr)
        except Exception as e:
            raise HTTPException(422, f"inference failed: {e}")
        finally:
            state.n_inflight -= 1
            state.latency_sum += time.perf_counter() - t0
            state.n_total += 1
        result["file"] = file.filename
        return JSONResponse(result)

    return app


def _run_inference(state: AppState, buf: bytes, sr: int) -> dict:
    """工作线程内：解码音频 -> 推理。model 在 eval+no_grad 下并发安全。"""
    audio = decode_audio(buf, sr, mono=True)
    with torch.no_grad():
        return infer_audio(
            audio, sr, state.cfg, state.model, state.thresholds, state.device,
            clip_thresholds=state.clip_thresholds, mel_cfg=state.mel_cfg,
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "best.pt"))
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--workers", type=int, default=8, help="推理线程池大小")
    ap.add_argument("--num-threads", type=int, default=1, help="每请求 torch 算子线程数")
    args = ap.parse_args()

    import uvicorn

    app = build_app(args.checkpoint, args.workers, args.num_threads)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
