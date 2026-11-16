# AudioGuard · Experimental v0.1

仓库与目录名：`audioguard`。

播放器音频异常检测实验项目：log-mel + 因果卷积 + 单向 GRU，输出逐帧多标签
概率。支持训练、离线文件检测、流式检测和 HTTP 服务。代码采用 [MIT](LICENSE)。

7 类：noise、harsh、clip_intermittent、distortion、hum、silence_loss、
loudness_anomaly。没有类别超过阈值时判为 clean。

## 安装与验证

在仓库根目录操作，使用 Python 3.14。`constraints.txt` 固定本次验证的直接依赖版本：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q
python scripts/smoke.py
```

测试自行生成临时权重和音频，不依赖 `checkpoints/best.pt` 或本地数据。
冒烟执行数据生成、一轮训练、文件推理和误例绘图。CI 使用 Linux CPU；实际
执行结果见 [验证记录](docs/VALIDATION.md)。标注 UI 是可选功能，另安装
`requirements-labeling.txt`。

## 最小训练示例

```bash
python -m src.training.train --config configs/demo.yaml --device cpu
python scripts/prepare_data.py --config configs/demo.yaml --n-samples 2 --out data/demo
```

训练默认即时合成，不读取 prepare_data 导出的样本；后者仅用于检查和试听。
`configs/demo.yaml` 使用 small 模型和极小数据量，输出到 `checkpoints/demo/`。
正式实验使用 `configs/default.yaml`，默认从头训练，不覆盖 demo 模型。

```bash
python -m src.inference.offline --checkpoint checkpoints/demo/best.pt --input your_recording.wav --out results.json
python -m src.evaluation.analyze_errors --checkpoint checkpoints/demo/best.pt --out logs/demo/errors --n 4
python -m src.inference.service --checkpoint checkpoints/demo/best.pt --host 127.0.0.1 --port 8000 --workers 4
curl -F 'file=@your_recording.wav' http://localhost:8000/detect
```

HTTP 接口还包括 `/health` 和 `/metrics`。服务用于本地实验，未提供认证或公网
部署配置。误例分析使用 checkpoint 保存的配置和对应验证分区。

## 流式使用

新配置采用固定仿射归一化（mean=-40 dB、std=20 dB），训练、整段和分块特征
保持一致。每个检测器独立保留卷积上下文和 GRU hidden，模型权重可共享。

```python
import torch
from src.inference.offline import load_checkpoint
from src.inference.streaming import StreamingDetector

cfg, model, thresholds, _ = load_checkpoint("checkpoints/demo/best.pt", torch.device("cpu"))
detector = StreamingDetector.from_cfg(model, cfg, thresholds, retain_history=False)
# chunk 必须是 checkpoint 采样率的 mono float32 numpy 数组
# probs = detector.push_samples(chunk)   # [7, 新增帧数]
# tail = detector.flush()                # 结束后不能继续 push；重复 flush 返回空
```

`retain_history=False` 只保留推理状态，调用者自行消费结果；默认 True 可通过
`all_probs()` 获取历史。默认 100ms 推理步进，特征窗口为 2048/44100 秒；首次
可输出时间约 140ms，另加计算和输入缓冲时间，不能作为设备延迟保证。
帧段起止按 hop 网格给出；最后不足完整 FFT 窗口的尾部丢弃，短于一帧的非空
录音补零成一帧。可选平滑只使用过去帧，不回写已经输出的结果。

## 数据与评估

`data.source` 可选 synth、real、labeled。real 使用公开音源后注入合成异常，
仍不等于真实异常标注。manifest 支持 source_id；相关录音/说话人/场景应共享
分组 ID。labeled 使用完整标注 JSONL，先分录音，再切片，重叠异常保留多标签。
填充帧不参与损失或评估。详见 [数据格式和授权说明](docs/DATA.md)。

验证集用于选 checkpoint 和调阈值。发布质量成绩必须另做固定阈值的独立测试。
默认合成样本由 seed/index 固定生成，不会每个 epoch 自动换一批。

## 旧权重兼容与限制

已有本地 best.pt 使用 per-clip 归一化，保留整段离线推理兼容；实时检测器会
拒绝这种特征配置，避免以错误的窗口归一化悄悄使用旧权重。要使用新的流式
路径，需要按 fixed 配置重新训练；不能把旧权重直接改配置后当成新模型。
旧权重历史约 0.859 的帧级 macro F1 来自合成验证集，不能代表真实播放器表现。

仓库不分发音频、数据下载包或权重。详情见 [模型说明](docs/MODEL_CARD.md)。
