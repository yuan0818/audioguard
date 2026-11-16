"""程序化合成 7 类音频异常。

每个注入器返回 (audio_out, mask_samples)：
    audio_out     —— 修改后波形（float32，长度可与输入相同）
    mask_samples  —— bool 数组，标记该异常在样本级生效的位置（用于生成 per-frame 标签）

顶层 `synthesize()` 随机选 0..max_classes 类顺序注入，产出 (audio, label_frames[n_classes, T])。
合成即带 ground truth，作为无数据起步阶段的训练主力。

标签帧粒度与 features.mel (center=False) 对齐：帧 i 覆盖样本 [i*hop, i*hop+n_fft)，
若该区间内 mask 任一为真则该帧为正。
"""

from __future__ import annotations

from typing import Callable, Mapping, Tuple

import numpy as np
from scipy.signal import butter, iirpeak, lfilter, sosfilt, tf2sos

from src.features.mel import MelConfig

# 类别顺序须与 configs/default.yaml 的 classes 一致
CLASS_NAMES = [
    "noise",
    "harsh",
    "clip_intermittent",
    "distortion",
    "hum",
    "silence_loss",
    "loudness_anomaly",
]
CLASS_TO_IDX = {n: i for i, n in enumerate(CLASS_NAMES)}

AudioMask = Tuple[np.ndarray, np.ndarray]  # (audio, mask_samples)


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def _full_mask(n: int) -> np.ndarray:
    return np.ones(n, dtype=bool)


def _random_region(n: int, rng: np.random.Generator, min_frac: float = 0.2) -> Tuple[int, int]:
    """在 [0,n) 中选一段长度 >= min_frac*n 的区间。"""
    lo = int(rng.integers(0, max(1, int(n * (1 - min_frac) + 1))))
    max_hi = n
    min_hi = lo + int(n * min_frac)
    hi = int(rng.integers(min(min_hi, max_hi), max_hi + 1))
    hi = max(hi, min_hi)
    return lo, min(hi, n)


def _colored_noise(n: int, color: str, rng: np.random.Generator) -> np.ndarray:
    """生成白/粉/棕噪声。"""
    white = rng.standard_normal(n).astype(np.float32)
    if color == "white":
        return white
    if color == "pink":
        # 简易粉噪：1/f 滤波
        b = np.array([0.049922035, -0.095993541, 0.050612256, -0.004408786], dtype=np.float32)
        a = np.array([1.0, -2.49195479, 2.07009966, -0.57180957], dtype=np.float32)
        return lfilter(b, a, white).astype(np.float32)
    if color == "brown":
        # 棕噪：积分（低通）
        out = np.cumsum(white)
        out = out / (np.max(np.abs(out)) + 1e-9)
        return out.astype(np.float32)
    return white


def _set_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """缩放噪声使 signal/noise 达到目标 SNR（按 RMS）。"""
    s_p = np.sqrt(np.mean(signal ** 2) + 1e-12)
    n_p = np.sqrt(np.mean(noise ** 2) + 1e-12)
    if n_p < 1e-12:
        return noise
    gain = s_p / (n_p * 10.0 ** (snr_db / 20.0))
    return noise * gain


# --------------------------------------------------------------------------- #
# 注入器
# --------------------------------------------------------------------------- #
def inject_noise(audio, sr, rng, p) -> AudioMask:
    """宽带背景噪声（白/粉/棕），随机 SNR，作用于整段。"""
    n = len(audio)
    color = p["colors"][rng.integers(0, len(p["colors"]))]
    snr = float(rng.uniform(*p["snr_db"]))
    noise = _colored_noise(n, color, rng)
    noise = _set_snr(audio, noise, snr)
    return audio + noise, _full_mask(n)


def inject_harsh(audio, sr, rng, p) -> AudioMask:
    """尖锐刺耳：多个高频共振峰 + 高频段谐波激励（波形整形只作用于高频）。"""
    n = len(audio)
    out = audio.copy()
    n_peaks = int(rng.integers(*p.get("n_peaks", [2, 4])))
    for _ in range(n_peaks):
        f0 = float(rng.uniform(*p["freq_range"]))
        q = float(rng.uniform(*p["q"]))
        gain_db = float(rng.uniform(*p["gain_db"]))
        b, a = iirpeak(f0, q, sr)
        sos = tf2sos(b, a)
        out = sosfilt(sos, out).astype(np.float32)
    # 高频谐波激励：高通提取高频成分 -> 硬削波整形 -> 混回（制造金属感刺耳泛音）
    # 用硬削波而非 tanh 饱和，与 distortion 的全带 tanh 失真区分开
    if rng.random() < p.get("p_excite", 0.5):
        cutoff = float(rng.uniform(*p.get("excite_cutoff", [3000.0, 6000.0])))
        sos_hp = butter(4, cutoff, btype="highpass", fs=sr, output="sos")
        hi = sosfilt(sos_hp, audio)
        drive = float(rng.uniform(*p.get("excite_drive", [2.0, 6.0])))
        excited = np.clip(hi * drive, -1.0, 1.0)
        mix = float(rng.uniform(*p.get("excite_mix", [0.3, 1.0])))
        out = out + mix * (excited - hi)
    # 峰值归一到原峰值，避免与 loudness 类混淆
    peak = np.max(np.abs(out)) + 1e-9
    out = out / peak * min(1.0, np.max(np.abs(audio)))
    return out.astype(np.float32), _full_mask(n)


def inject_clip_intermittent(audio, sr, rng, p) -> AudioMask:
    """爆音/断续：强削波 + 零填充卡顿 + 爆音 clicks + 缓冲卡顿重复(stutter)。"""
    n = len(audio)
    out = audio.copy()
    mask = np.zeros(n, dtype=bool)

    # A. 强削波（整段或长区间，模拟功放/数字过载）
    if rng.random() < p.get("p_hard_clip", 0.6):
        thresh = float(rng.uniform(*p["clip_thresh"]))
        lo, hi = _random_region(n, rng, min_frac=0.5)
        seg = out[lo:hi]
        out[lo:hi] = np.clip(seg, -thresh, thresh)
        mask[lo:hi] = True  # 整个区间为正（削波是区间级缺陷，标签加密）

    # B. dropout（信号丢失）
    drop_p = float(rng.uniform(*p["dropout_p"]))
    if drop_p > 0:
        chunk_ms = int(rng.uniform(*p["dropout_chunk_ms"]))
        chunk = max(int(sr * chunk_ms / 1000.0), 1)
        i = 0
        while i < n - chunk:
            if rng.random() < drop_p:
                out[i : i + chunk] = 0.0
                mask[i : i + chunk] = True
                i += chunk * 2  # 跳过避免重叠
            else:
                i += chunk

    # C. clicks 爆音（短噪声脉冲，区别于静音类）
    n_clicks = int(rng.integers(*p.get("n_clicks", [1, 6])))
    click = max(int(sr * float(rng.uniform(*p.get("click_ms", [1, 8]))) / 1000.0), 1)
    for _ in range(n_clicks):
        pos = int(rng.integers(0, max(1, n - click)))
        amp = float(rng.uniform(*p.get("click_amp", [0.3, 1.0])))
        out[pos : pos + click] += amp * rng.standard_normal(click)
        mask[pos : pos + click] = True

    # D. stutter（缓冲卡顿：小段重复 2-3 次）
    if rng.random() < p.get("p_stutter", 0.4):
        rep = max(int(sr * float(rng.uniform(*p.get("stutter_ms", [30, 120]))) / 1000.0), 1)
        n_rep = int(rng.integers(2, 4))
        start = int(rng.integers(0, max(1, n - rep * n_rep)))
        seg = audio[start : start + rep]
        for k in range(1, n_rep):
            s1 = start + k * rep
            e1 = min(s1 + rep, n)
            out[s1:e1] = seg[: e1 - s1]
            mask[s1:e1] = True
    return out.astype(np.float32), mask


def inject_distortion(audio, sr, rng, p) -> AudioMask:
    """失真/破音：waveshater 非线性饱和（tanh/cubic/soft），整段或局部。"""
    n = len(audio)
    drive = float(rng.uniform(*p["drive"]))
    kind = p["kind"][rng.integers(0, len(p["kind"]))]
    lo, hi = _random_region(n, rng, min_frac=0.4)
    seg = audio[lo:hi]
    driven = seg * drive
    if kind == "tanh":
        shaper = np.tanh(driven)
    elif kind == "cubic":
        shaper = driven - (driven ** 3) / 3.0
    else:  # soft（折叠）
        shaper = np.sin(driven)
    # 归一回并补偿增益
    shaper = shaper / (np.max(np.abs(shaper)) + 1e-9) * np.max(np.abs(seg))
    out = audio.copy()
    out[lo:hi] = shaper.astype(np.float32)
    mask = np.zeros(n, dtype=bool)
    mask[lo:hi] = True
    return out, mask


def inject_hum(audio, sr, rng, p) -> AudioMask:
    """工频干扰：50/60Hz + 谐波叠加。"""
    n = len(audio)
    f0 = float(rng.uniform(*p["freq"]))
    n_harm = int(rng.integers(*p["n_harmonics"]))
    amp = float(rng.uniform(*p["amp"]))
    t = np.arange(n) / sr
    hum = np.zeros(n, dtype=np.float32)
    for k in range(1, n_harm + 1):
        # 高次谐波幅度衰减
        a = amp / k
        phase = float(rng.uniform(0, 2 * np.pi))
        hum += a * np.sin(2 * np.pi * f0 * k * t + phase)
    return audio + hum, _full_mask(n)


def inject_silence_loss(audio, sr, rng, p) -> AudioMask:
    """静音/信号丢失：整段或局部置零。"""
    n = len(audio)
    out = audio.copy()
    mask = np.zeros(n, dtype=bool)
    if rng.random() < p["whole_p"]:
        out[:] = 0.0
        mask[:] = True
    else:
        chunk_p = float(rng.uniform(*p["chunk_p"]))
        i = 0
        while i < n:
            if rng.random() < chunk_p:
                length = int(rng.integers(int(sr * 0.05), max(int(sr * 0.05) + 1, int(sr * 0.3))))
                end = min(i + length, n)
                out[i:end] = 0.0
                mask[i:end] = True
                i = end
            else:
                i += int(sr * 0.02)
    return out, mask


def inject_loudness_anomaly(audio, sr, rng, p) -> AudioMask:
    """音量异常：极端增益 + 可选过压缩（限幅器）。"""
    n = len(audio)
    gain_db = float(rng.uniform(*p["gain_db"]))
    out = audio * (10.0 ** (gain_db / 20.0))
    if rng.random() < p["p_overcompress"]:
        # 简易限幅：硬限幅到 -3dBFS 后再放大，制造过压缩刺耳感
        out = np.tanh(out) * 0.7
    # 防数值溢出
    out = np.clip(out, -1.0, 1.0).astype(np.float32)
    return out, _full_mask(n)


INJECTORS: dict[str, Callable] = {
    "noise": inject_noise,
    "harsh": inject_harsh,
    "clip_intermittent": inject_clip_intermittent,
    "distortion": inject_distortion,
    "hum": inject_hum,
    "silence_loss": inject_silence_loss,
    "loudness_anomaly": inject_loudness_anomaly,
}


# --------------------------------------------------------------------------- #
# 帧级标签转换
# --------------------------------------------------------------------------- #
def masks_to_frames(
    masks_by_class: dict[str, np.ndarray], n: int, mel_cfg: MelConfig
) -> np.ndarray:
    """样本级 mask -> 帧级标签 [n_classes, T]，与 features.mel (center=False) 对齐。"""
    n_fft, hop = mel_cfg.n_fft, mel_cfg.hop_length
    if n < n_fft:
        T = 1
        frames_start = np.array([0])
    else:
        T = (n - n_fft) // hop + 1
        frames_start = np.arange(T) * hop
    labels = np.zeros((len(CLASS_NAMES), T), dtype=np.float32)
    for name, mask in masks_by_class.items():
        c = CLASS_TO_IDX[name]
        for i in range(T):
            s = frames_start[i]
            e = min(s + n_fft, n)
            if np.any(mask[s:e]):
                labels[c, i] = 1.0
    return labels


# --------------------------------------------------------------------------- #
# 顶层合成
# --------------------------------------------------------------------------- #
def synthesize(
    audio: np.ndarray,
    sr: int,
    synth_cfg: Mapping,
    mel_cfg: MelConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """对干净 audio 随机注入 0..max_classes 类异常。

    Returns:
        audio_out: 修改后波形 float32（长度与输入相同）。
        labels: per-frame 多标签 [n_classes, T]。
        chosen: 实际注入的类别名列表（空 = 干净）。
    """
    audio = np.asarray(audio, dtype=np.float32)
    n = len(audio)
    max_classes = int(synth_cfg.get("max_classes", 3))
    p_clean = float(synth_cfg.get("p_clean", 0.15))

    out = audio.copy()
    masks: dict[str, np.ndarray] = {}

    if rng.random() < p_clean:
        chosen: list[str] = []
    else:
        k = int(rng.integers(0, max_classes + 1))  # 注入类数（可含 0）
        k = min(k, len(CLASS_NAMES))
        chosen = list(rng.choice(CLASS_NAMES, size=k, replace=False))
        for name in chosen:
            injector = INJECTORS[name]
            params = synth_cfg.get(name, {})
            try:
                out, mask = injector(out, sr, rng, params)
            except Exception as e:  # 单类失败不阻断整体
                print(f"[synth] injector {name} failed: {e}")
                continue
            # 合并 mask：多类叠加取并
            if name in masks:
                masks[name] = masks[name] | mask
            else:
                masks[name] = mask

    labels = masks_to_frames(masks, n, mel_cfg)
    return out.astype(np.float32), labels, chosen


def make_clean_label(n: int, mel_cfg: MelConfig) -> np.ndarray:
    """构造全零（干净）per-frame 标签。"""
    masks = {}
    labels = masks_to_frames(masks, n, mel_cfg)
    return labels
