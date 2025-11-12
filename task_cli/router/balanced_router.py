# router/balanced_router.py
# -*- coding: utf-8 -*-
from typing import Dict, List, Tuple, Union
from .route_score import pick_model

Number = Union[int, float]

def assign_batch(
    tasks: List[Dict],                    # 每条: {"category":str, "difficulty":str|float}
    domain_index: Dict[str, int],
    models_cap: Dict[str, List[float]],
    init_queues: Dict[str, Number] = None,
    k: float = 0.2,
    shadow_mode: str = "log",
    shadow_alpha: float = 1.0,
    service_tokens: Dict[str, Number] = None,  # 每个任务在该模型上的“占用量”，默认=1
) -> Tuple[List[Dict], Dict[str, Number]]:
    """
    返回:
      - assigns: 每条 {"task_idx":i, "chosen":model, "score":x, "detail":{...}}
      - queues : 最终队列（或累计负载）
    说明:
      - 我们把 'queues' 当“累积负载”用：每领一个任务，chosen 的队列 +load
      - 如果不同任务的占用不同，可在 service_tokens 里指定每个模型的单位代价
    """
    queues = dict(init_queues) if init_queues else {m: 0.0 for m in models_cap}
    assigns: List[Dict] = []
    for i, t in enumerate(tasks):
        # 选模型（内部用“相对拥塞”的背压）
        chosen, score, detail = pick_model(
            task_category=t["category"],
            task_difficulty=t.get("difficulty", "medium"),
            domain_index=domain_index,
            models_cap=models_cap,
            queues=queues,
            k=k, shadow_mode=shadow_mode, shadow_alpha=shadow_alpha,
        )
        assigns.append({"task_idx": i, "chosen": chosen, "score": score, "detail": detail})

        # 更新该模型的“负载/队列”
        inc = 1.0
        if service_tokens and chosen in service_tokens:
            inc = float(service_tokens[chosen])
        queues[chosen] = float(queues.get(chosen, 0.0)) + inc
    return assigns, queues
