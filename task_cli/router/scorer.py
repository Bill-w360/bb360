# router/scorer.py
# -*- coding: utf-8 -*-
from typing import Dict, List, Tuple, Union
from .route_score import route_score_single

Number = Union[int, float]

def route_all_models(
    task_category: str,
    task_difficulty: Union[str, Number],
    domain_index: Dict[str, int],
    models_cap: Dict[str, List[float]],
    queues: Dict[str, Number],
    k: float = 1.0,
    shadow_mode: str = "log",
    shadow_alpha: float = 1.0,
) -> Tuple[str, float, Dict[str, Dict[str, float]]]:
    """
    返回: (best_model, best_score, detail_by_model)
    detail_by_model[name] = {"w1":..., "shadow":..., "score":..., "q":..., "cap_cat":...}
    """
    if task_category not in domain_index:
        raise KeyError(f"unknown task_category='{task_category}'")
    cat_idx = domain_index[task_category]

    best_name, best_score = None, float("inf")
    detail: Dict[str, Dict[str, float]] = {}

    from .route_score import map_difficulty, w1_1d, shadow_price
    td = map_difficulty(task_difficulty)

    for name, cap in models_cap.items():
        q = float(queues.get(name, 0.0))
        w1 = w1_1d(cap[cat_idx], td)
        sp = shadow_price(q, mode=shadow_mode, alpha=shadow_alpha)
        score = w1 - k * sp
        detail[name] = {"w1": w1, "shadow": sp, "score": score, "q": q, "cap_cat": float(cap[cat_idx])}
        if score < best_score:
            best_name, best_score = name, score

    return best_name, best_score, detail
