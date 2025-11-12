from __future__ import annotations
from typing import Dict, Any
import os

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

from .utils import build_default_benchmark_mapping

DEFAULT_CONFIG: Dict[str, Any] = {
    "benchmark_mapping": build_default_benchmark_mapping(),
    "datasets": {
        # 可以在这里覆盖某些数据集的 metric 与归一化策略
        "MATH": {"metric": "acc", "scale": "x100", "higher_is_better": True},
        "SQuAD": {"metric": "f1", "higher_is_better": True},
    }
}

def load_config(config_path: str | None) -> Dict[str, Any]:
    if config_path is None:
        return DEFAULT_CONFIG
    if not os.path.exists(config_path):
        return DEFAULT_CONFIG
    if yaml is None:
        return DEFAULT_CONFIG
    with open(config_path, "r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}
    # 合并用户配置
    cfg = DEFAULT_CONFIG.copy()
    # 合并 benchmark mapping
    bm = cfg.get("benchmark_mapping", {}).copy()
    bm.update(user_cfg.get("benchmark_mapping", {}))
    cfg["benchmark_mapping"] = bm
    # 合并 datasets 策略
    ds = cfg.get("datasets", {}).copy()
    ds.update(user_cfg.get("datasets", {}))
    cfg["datasets"] = ds
    return cfg
