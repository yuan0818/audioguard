"""时域统计特征、增强与片段聚合（纯 NumPy，不依赖 torch）。"""
from __future__ import annotations

import copy

import numpy as np
import pytest

from src.data.augmentations import apply_augmentations
from src.features.mel import (N_TD, TD_DB_SCALE, TD_LEVEL_ROWS, TD_NAMES, MelConfig, MelStream,
                              _mel_filterbank, compute_mel, normalize_mel, time_domain_features)
from src.models.pooling import clip_pool_np
from src.utils.config import load_config

SR = 44100


def _cfg(time_features=True):
    cfg = copy.deepcopy(load_config("configs/default.yaml"))
    cfg["features"]["time_features"] = time_features
    return cfg


def _sine(sec=1.0, amp=0.5, f=440.0):
    t = np.arange(int(SR * sec)) / SR
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)


def _rows(name):
    return TD_NAMES.index(name)


@pytest.mark.parametrize("samples", [1, 2047, 2048, 2489, 30000])
def test_time_features_stream_matches_offline(samples):
    mc = MelConfig.from_cfg(_cfg())
    assert mc.n_features == mc.n_mels + N_TD
    audio = np.random.default_rng(0).normal(0, 0.1, samples).astype(np.float32)
    audio[samples // 2] = 0.9  # 一个点击
    fb = _mel_filterbank(mc)
    offline = compute_mel(audio, mc, fb=fb)
    stream = MelStream(mc, fb=fb)
    parts = [stream.push(c) for c in np.array_split(audio, 11)] + [stream.flush()]
    online = normalize_mel(np.concatenate(parts, axis=1), mc)
    assert online.shape == offline.shape
    assert offline.shape[0] == mc.n_mels + N_TD
    np.testing.assert_allclose(online, offline, atol=1e-4)


def test_legacy_config_has_no_time_rows():
    mc = MelConfig.from_cfg(_cfg(time_features=False))
    assert mc.n_features == mc.n_mels
    assert compute_mel(_sine(0.2), mc).shape[0] == mc.n_mels


def test_time_features_flag_clipping_and_clicks():
    n = 2048
    clean = _sine(n / SR, amp=0.9)[None, :n]
    clipped = np.clip(clean * 3.0, -0.5, 0.5)
    click = clean * 0.2
    click[0, 1000] += 0.8
    f_clean, f_clip, f_click = (time_domain_features(x)[:, 0] for x in (clean, clipped, click))
    assert f_clip[_rows("top_frac")] > f_clean[_rows("top_frac")] + 1.0
    assert f_clip[_rows("flat_frac")] > f_clean[_rows("flat_frac")] + 0.5
    assert f_clip[_rows("crest_db")] < f_clean[_rows("crest_db")]
    assert f_click[_rows("jump")] > f_clean[_rows("jump")] + 0.5
    gap = clean.copy()
    gap[0, 512:1024] = 0.0
    f_gap = time_domain_features(gap)[:, 0]
    assert f_gap[_rows("env_range")] > f_clean[_rows("env_range")] + 1.0
    assert f_gap[_rows("zero_frac")] > f_clean[_rows("zero_frac")]


def test_time_features_silence_is_finite():
    f = time_domain_features(np.zeros((3, 2048), dtype=np.float32))
    assert f.shape == (N_TD, 3) and np.isfinite(f).all()


ONLY_GAIN = {"p": 1.0, "gain_db": [-4.0, 4.0], "crop_max_frac": 0.0, "freq_masks": 0, "time_mask_p": 0.0}


def test_gain_is_additive_offset_in_log_domain():
    rng = np.random.default_rng(1)
    n_mels = 8
    feat = rng.normal(size=(n_mels + N_TD, 50)).astype(np.float32)
    labels = np.zeros((7, 50), dtype=np.float32)
    out, _ = apply_augmentations(feat, labels, np.random.default_rng(2), n_mels=n_mels, params=ONLY_GAIN,
                                 mel_gain_scale=1 / 20, level_rows=TD_LEVEL_ROWS, level_scale=TD_DB_SCALE,
                                 level_class=6)
    delta = out[:n_mels] - feat[:n_mels]
    assert np.ptp(delta) < 1e-5 and 0 < abs(delta[0, 0]) <= 4 / 20 + 1e-6
    for r in range(N_TD):
        d = out[n_mels + r] - feat[n_mels + r]
        if r in TD_LEVEL_ROWS:
            np.testing.assert_allclose(d, delta[0, 0] * 20 * TD_DB_SCALE, atol=1e-5)
        else:
            np.testing.assert_array_equal(d, 0)


def test_gain_skipped_for_loudness_anomaly_and_clip_normalization():
    feat = np.random.default_rng(1).normal(size=(10, 30)).astype(np.float32)
    labels = np.zeros((7, 30), dtype=np.float32)
    labels[6, 5:9] = 1
    out, _ = apply_augmentations(feat, labels, np.random.default_rng(0), n_mels=10, params=ONLY_GAIN,
                                 mel_gain_scale=1 / 20, level_class=6)
    np.testing.assert_array_equal(out, feat)
    out, _ = apply_augmentations(feat, labels * 0, np.random.default_rng(0), n_mels=10, params=ONLY_GAIN,
                                 mel_gain_scale=0.0, level_class=6)
    np.testing.assert_array_equal(out, feat)


def test_start_crop_never_wraps_around():
    T = 100
    feat = np.tile(np.arange(T, dtype=np.float32), (4, 1))
    labels = np.zeros((7, T), dtype=np.float32)
    labels[2, 90:] = 1
    params = {"p": 1.0, "crop_max_frac": 0.1, "freq_masks": 0, "time_mask_p": 0.0, "gain_db": [0.0, 0.0]}
    for seed in range(30):
        out, lab = apply_augmentations(feat, labels, np.random.default_rng(seed), n_mels=4, params=params)
        k = T - out.shape[1]
        assert 0 <= k <= 10
        np.testing.assert_array_equal(out, feat[:, k:])
        np.testing.assert_array_equal(lab, labels[:, k:])
        assert np.all(np.diff(out[0]) == 1)  # 单调：无首尾拼接断点


def test_freq_mask_fills_with_band_mean_and_spares_time_rows():
    rng = np.random.default_rng(3)
    n_mels = 16
    feat = rng.normal(size=(n_mels + N_TD, 40)).astype(np.float32)
    params = {"p": 1.0, "crop_max_frac": 0.0, "freq_masks": 1, "freq_mask_max": 6, "time_mask_p": 0.0,
              "gain_db": [0.0, 0.0]}
    out, _ = apply_augmentations(feat, np.zeros((7, 40), np.float32), np.random.default_rng(4),
                                 n_mels=n_mels, params=params)
    changed = np.where(np.any(out != feat, axis=1))[0]
    assert len(changed) and changed.max() < n_mels
    for r in changed:
        np.testing.assert_allclose(out[r], feat[r].mean(), atol=1e-6)


def test_linear_softmax_is_not_diluted_by_clean_frames():
    p = np.full(300, 0.01, dtype=np.float32)
    p[100:103] = 0.95  # 30ms 爆音
    assert clip_pool_np(p, "mean") < 0.03
    assert clip_pool_np(p, "linear_softmax") > 0.45
    assert clip_pool_np(p, "max") == pytest.approx(0.95)
    assert clip_pool_np(np.zeros(10), "linear_softmax") == 0.0
