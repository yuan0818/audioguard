"""CRNN 模型测试（需 torch）。

运行: python -m pytest tests/test_model.py
或  : python tests/test_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.crnn import build_model
from src.utils.config import load_config_resolved

_CFG = load_config_resolved(ROOT / "configs" / "default.yaml", root=ROOT)


def test_forward_shape():
    model = build_model(_CFG).eval()
    B, n_mels, T = 2, build_model(_CFG).n_features, 64
    x = torch.randn(B, 1, n_mels, T)
    logits, hidden = model(x)
    assert logits.shape == (B, _CFG["model"]["n_classes"], T), logits.shape
    assert hidden.shape[0] == model.n_layers
    assert hidden.shape[2] == model.gru_hidden


def test_variable_length():
    model = build_model(_CFG).eval()
    for T in [1, 8, 33, 128]:
        x = torch.randn(1, 1, model.n_features, T)
        logits, _ = model(x)
        assert logits.shape[2] == T, (T, logits.shape)


def test_backward_step():
    model = build_model(_CFG).train()
    x = torch.randn(2, 1, model.n_features, 32)
    labels = torch.randint(0, 2, (2, _CFG["model"]["n_classes"], 32)).float()
    logits, _ = model(x)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert all(g is not None for g in grads)
    assert all(not torch.isnan(g).any() for g in grads)


def test_param_count_base():
    model = build_model(_CFG)
    n = model.n_parameters()
    assert 1e5 < n < 1e7, n  # base 约 0.1M-10M 合理


def test_stateful_hidden():
    """分两段喂入（携带 hidden）应与一次性整段喂入在公共帧上接近（GRU 因果）。"""
    torch.manual_seed(0)
    model = build_model(_CFG).eval()
    T = 64
    x = torch.randn(1, 1, model.n_features, T)
    logits_full, _ = model(x)
    # 分两段：前 40 帧带初始 hidden，后 24 帧携带返回 hidden
    logits_a, state = model.forward_stream(x[..., :40])
    logits_b, _ = model.forward_stream(x[..., 40:], state)
    actual = torch.cat([logits_a, logits_b], dim=-1)
    torch.testing.assert_close(actual, logits_full, atol=1e-5, rtol=1e-5)




def test_time_feature_branch_and_input_check():
    import copy
    import pytest
    from src.features.mel import N_TD
    cfg = copy.deepcopy(_CFG)
    cfg["features"]["time_features"] = True
    model = build_model(cfg).eval()
    assert model.n_td == N_TD and model.n_features == cfg["model"]["n_mels"] + N_TD
    x = torch.randn(1, 1, model.n_features, 20)
    logits, _ = model(x)
    assert logits.shape == (1, cfg["model"]["n_classes"], 20)
    # 时域行确实参与计算
    y = x.clone()
    y[:, :, model.n_mels:] += 1.0
    assert not torch.allclose(model(y)[0], logits)
    with pytest.raises(ValueError, match="feature rows"):
        model(torch.randn(1, 1, cfg["model"]["n_mels"], 20))
    cfg["features"]["time_features"] = False
    legacy = build_model(cfg)
    assert legacy.n_td == 0 and legacy.td_proj is None


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                print(f"FAIL {name}: {e}")
                failed += 1
            except Exception as e:
                print(f"ERROR {name}: {type(e).__name__}: {e}")
                failed += 1
    print(f"\n{'ALL PASS' if failed == 0 else f'{failed} FAILED'}")
    sys.exit(1 if failed else 0)
