"""配置加载与路径工具。

提供 YAML 配置加载、深合并、路径解析与随机种子设置。
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml


def load_config(path: str | os.PathLike) -> dict:
    """加载 YAML 配置文件为字典。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        cfg = {}
    return cfg


def deep_merge(base: Mapping, override: Mapping) -> dict:
    """递归合并 override 到 base（override 优先）。"""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], Mapping) and isinstance(v, Mapping):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def resolve_paths(cfg: dict, root: str | os.PathLike | None = None) -> dict:
    """将 cfg['paths'] 下的相对路径转为绝对路径（相对 root，默认项目根）。"""
    root = Path(root).resolve() if root is not None else Path.cwd().resolve()
    paths = cfg.get("paths", {})
    resolved = {}
    for k, v in paths.items():
        p = Path(v)
        resolved[k] = str(p if p.is_absolute() else (root / p))
    cfg["paths"] = resolved
    data = cfg.get("data", {})
    for key in ("labeled_jsonl",):
        if data.get(key):
            value = Path(data[key])
            data[key] = str(value if value.is_absolute() else root / value)
    data["real_manifests"] = [str(Path(v) if Path(v).is_absolute() else root / v)
                              for v in data.get("real_manifests", [])]
    return cfg


def load_config_resolved(path: str | os.PathLike, root: str | os.PathLike | None = None) -> dict:
    """加载并解析路径的便捷入口。"""
    cfg = load_config(path)
    return resolve_paths(cfg, root=root)


def set_seed(seed: int) -> None:
    """固定 Python / NumPy / PyTorch 随机种子以保证可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def ensure_dirs(cfg: dict) -> None:
    """按 cfg['paths'] 创建所需目录。"""
    for p in cfg.get("paths", {}).values():
        Path(p).mkdir(parents=True, exist_ok=True)


def get_project_root() -> Path:
    """返回项目根目录（src 的上两级）。"""
    return Path(__file__).resolve().parents[2]
