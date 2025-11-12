from __future__ import annotations
from typing import Dict, Tuple

# 说明：把不同评测框架输出的指标，统一归一到 [0, 100] 之间。
# 默认规则：
# - 命中类（acc、accuracy、f1、em、pass@1、bleu、rouge*、exact_match）若在 [0,1] 则乘 100；若在 [0,100] 直接使用；
# - 其他特殊指标可在 config 里覆盖：{ dataset: { "metric": "f1", "scale": "x100", "higher_is_better": true } }

def to_0_100(value: float, higher_is_better: bool = True) -> float:
    # 自动把 [0,1] 拉到 [0,100]
    if 0 <= value <= 1:
        v = value * 100.0
    else:
        v = value
    # 如果是 “越小越好”，则转为得分（简单反转）
    if not higher_is_better:
        # 假设最小值 0，最大值 100
        v = 100.0 - max(0.0, min(100.0, v))
    return float(max(0.0, min(100.0, v)))

# 针对常见指标名的默认 higher_is_better 策略
DEFAULT_HIGHER_IS_BETTER = {
    "acc": True, "accuracy": True, "f1": True, "em": True, "exact_match": True,
    "bleu": True, "rouge": True, "rouge1": True, "rouge2": True, "rougeL": True,
    "pass@1": True, "code_pass@1": True,
    "loss": False, "perplexity": False, "ppl": False,
}

def normalize_metric(dataset: str, metric_name: str, value: float, cfg: Dict) -> float:
    ds_cfg = cfg.get("datasets", {}).get(dataset, {})
    # 优先使用配置里指定的 higher_is_better
    hib = ds_cfg.get("higher_is_better")
    if hib is None:
        hib = DEFAULT_HIGHER_IS_BETTER.get(metric_name.lower(), True)
    # scale 规则（目前支持 x100 / identity）
    scale = ds_cfg.get("scale")
    if scale == "x100":
        value = value * 100.0
    # 常规归一
    return to_0_100(value, higher_is_better=hib)
