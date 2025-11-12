# router/router_scores.py
# -*- coding: utf-8 -*-
"""
Routing score utilities.

total score (minimize):
    total = W1(model_cap, task_vec)
            + gamma * cost_term
            + alpha * BP_penalty(queue_i | all_queues)
            + beta  * (1 - match_val)     # label mismatch

Notes
-----
- BP_penalty 采用“相对拥挤度”惩罚（比平均队列更长的部分才处罚），
  并支持刻度模式：'linear' / 'log' / 'sqrt'。
- 你可以把 alpha / beta / gamma / delta 作为超参从 config 或命令行传入。
- 选择得分最小的模型。
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Iterable
import math

# 依赖你项目中已有的 Wasserstein 计算（多维、带类型和权重）
from .wasserstein import w1_multidim


# ---------- 小工具 ----------

def _safe_float(x, default: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def _check_same_len(a: Iterable, b: Iterable, name_a: str = "a", name_b: str = "b"):
    la, lb = len(a), len(b)
    if la != lb:
        raise ValueError(f"{name_a} dim={la} != {name_b} dim={lb}")


# ---------- 成本项 ----------

def cost_term(cost: Dict, cost_w: Dict) -> float:
    """
    汇总成本：对 keys 取权重加权和；缺省为 0。

    cost      : 例如 {"tp":1.0, "price":0.2, "latency":0.5}
    cost_w    : 例如 {"tp":1.0, "price":0.0, "latency":0.0}
    """
    if not cost_w:
        return 0.0
    s = 0.0
    for k, w in cost_w.items():
        s += _safe_float(cost.get(k, 0.0)) * _safe_float(w, 0.0)
    return float(s)


# ---------- 难度映射（兼容旧接口，可按需使用） ----------

_DIFFICULTY_MAP = {"easy": 0.3, "medium": 0.6, "hard": 0.9}

def map_difficulty(x: str) -> float:
    """
    把 'easy'/'medium'/'hard' 映射到一个 0~1 的系数（可按需改）。
    """
    if not isinstance(x, str):
        return _DIFFICULTY_MAP["medium"]
    return _DIFFICULTY_MAP.get(x.strip().lower(), _DIFFICULTY_MAP["medium"])


# ---------- 负载均衡：BP 惩罚 ----------

def _relative_overload(q_i: float, q_all: List[float]) -> float:
    """
    只惩罚“高于平均”的部分：max(0, q_i - avg_q)。
    """
    if not q_all:
        return max(0.0, _safe_float(q_i))
    avg_q = sum(_safe_float(x) for x in q_all) / max(1, len(q_all))
    return max(0.0, _safe_float(q_i) - avg_q)

def bp_penalty(
    queue_len: float,
    all_queues: List[float],
    mode: str = "linear",
    eps: float = 1e-9
) -> float:
    """
    负载均衡惩罚。与“鼓励拥挤”的 shadow price 相反，我们**惩罚**相对拥挤度。

    Parameters
    ----------
    queue_len : 当前模型的队列长度或代理（也可以是利用率*时间窗）
    all_queues: 所有模型的队列长度（用于算平均）
    mode      : 'linear' | 'log' | 'sqrt'
    eps       : 稳定项

    Returns
    -------
    非负惩罚值，越拥挤越大。
    """
    rel = _relative_overload(queue_len, all_queues)  # 只看超过平均的部分
    if mode == "linear":
        return rel
    if mode == "log":
        return math.log1p(rel)
    if mode == "sqrt":
        return math.sqrt(rel + eps)
    # 不认识的就退化为线性
    return rel


# 兼容旧代码里可能引用的名字：
shadow_price = bp_penalty  # 注意：语义已变为惩罚（不是奖励）


# ---------- 匹配项（标签匹配） ----------

def label_mismatch_from_match(match_val: float) -> float:
    """
    将 match_val（越大越好）变为 label_mismatch（越小越好）。
    """
    mv = _safe_float(match_val, 0.0)
    # 截断在 [0,1]，防止训练误差或上游“相似度 > 1”
    mv = max(0.0, min(1.0, mv))
    return 1.0 - mv


# ---------- 单模型打分 ----------

def route_score_single(
    model_vec: List[float],
    task_vec: List[float],
    dim_types: List[str],
    dim_weights: List[float],
    cost: Dict,
    cost_w: Dict,
    queue_len: float,
    all_queues: List[float],
    alpha: float,   # BP 权重
    beta: float,    # 标签不匹配权重
    gamma: float,   # 成本权重
    delta: float,   # （保留参数，可用作额外缩放；当前未单独使用）
    match_val: float,
    bp_mode: str = "linear",
) -> Dict[str, float]:
    """
    计算单个模型对给定任务的各项得分与总分（越小越好）。
    """
    # 维度一致性校验
    _check_same_len(model_vec, task_vec, "model_vec", "task_vec")
    _check_same_len(dim_types, dim_weights, "dim_types", "dim_weights")
    if len(model_vec) != len(dim_types):
        raise ValueError(f"capability dim={len(model_vec)} != dim_types dim={len(dim_types)}")

    # 1) Wasserstein 距离（多维）
    w1 = float(w1_multidim(
        model_vec=model_vec,
        task_vec=task_vec,
        dim_types=dim_types,
        dim_weights=dim_weights
    ))

    # 假设每维权重和 ≤ 1，或者直接按维数归一化
    w1_scale = max(1.0, sum(abs(x) for x in dim_weights))  # 或用 len(dim_weights)
    w1 = float(w1) / w1_scale

    # 2) 成本
    c = float(cost_term(cost, cost_w))

    # 3) 负载均衡惩罚（只惩罚高于平均的相对拥挤度）
    bp = float(bp_penalty(queue_len, all_queues, mode=bp_mode))

    # 4) 标签不匹配
    lm = float(label_mismatch_from_match(match_val))

    total = w1 + gamma * c + alpha * bp + beta * lm

    return {
        "w1": w1,
        "cost": c,
        "bp": bp,
        "label_mismatch": lm,
        "total": total
    }


# ---------- 多模型选择（取 total 最小） ----------

def pick_model(
    models: List[Dict],
    task_vec: List[float],
    dim_types: List[str],
    dim_weights: List[float],
    cost_w: Dict,
    queues: Dict[str, float],
    alpha: float, beta: float, gamma: float, delta: float,
    match_vals: Optional[Dict[str, float]] = None,
    bp_mode: str = "linear",
) -> Tuple[str, float, Dict]:
    """
    从多个模型中选择“总分最小”的那个。

    Parameters
    ----------
    models : 列表，每个元素至少包含：
             {"name": str, "capability": [..], "cost": {...}}
             可选：{"queue": {...}}（若不提供则从 queues 取）
    task_vec : 任务向量
    dim_types, dim_weights : 路由维度的类型与权重
    cost_w : 成本项权重
    queues : 模型名 -> 队列长度（或等效载荷度量）
    match_vals : 模型名 -> match_val（若不提供，默认为 0）
    bp_mode : 'linear' / 'log' / 'sqrt'

    Returns
    -------
    (chosen_name, chosen_total, detail_dict)
    """
    # 组装 all_queues
    all_q = []
    for m in models:
        name = m["name"]
        q = None
        # 优先从参数 queues 获取；其次尝试 m.get("queue", {}).get("len")
        if queues is not None and name in queues:
            q = queues[name]
        else:
            q = m.get("queue", {}).get("len", 0.0)
        all_q.append(_safe_float(q, 0.0))

    # 逐模型打分
    best = None
    last_detail = None
    for idx, m in enumerate(models):
        name = m["name"]
        mvec = m["capability"]
        mcost = m.get("cost", {})
        q_i = all_q[idx]

        # match 值
        mv = 0.0
        if match_vals and (name in match_vals):
            mv = _safe_float(match_vals[name], 0.0)

        comp = route_score_single(
            model_vec=mvec,
            task_vec=task_vec,
            dim_types=dim_types,
            dim_weights=dim_weights,
            cost=mcost,
            cost_w=cost_w,
            queue_len=q_i,
            all_queues=all_q,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
            match_val=mv,
            bp_mode=bp_mode,
        )

        if best is None or comp["total"] < best[1]:
            best = (name, comp["total"])
            last_detail = comp

    if best is None:
        raise RuntimeError("pick_model: no models or failed to score.")

    return best[0], best[1], last_detail
