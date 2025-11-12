# extras/make_assign_history.py
# -*- coding: utf-8 -*-
"""
把 tasks_*.jsonl (+ models.yaml + config.yaml) 变成训练 label_matcher 的配对数据：
每行：{"task_vec":[...], "task_label":"...", "model_cap":[...], "model_labels":[...], "y":0/1}

弱监督打标策略（可改）：
1) 用 router.classifier.make_task_vector() 生成 task_vec，并得到 task_label (domain-difficulty)。
2) 计算模型与任务的“匹配分” s = 1 - W1(model_cap, task_vec) （越大越匹配）。
3) 排序后：前 top_p% 标 1，后面采样一些负样本标 0。
"""

# ==== ✅ 关键补丁：确保能 import 到项目内的 router/* ====
import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# ==========================================================

import json, argparse, random
from typing import List, Dict

from router.utils import load_yaml, read_jsonl, write_jsonl
from router.schemas import Task
from router.classifier import make_task_vector, classify_task_label
from router.wasserstein import w1_multidim


def load_models(models_yaml: str) -> List[Dict]:
    y = load_yaml(models_yaml)
    out = []
    for m in y["models"]:
        out.append({
            "name": m["name"],
            "labels": m.get("labels", []),
            "capability": m["capability"]
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", required=True,
                    help="一个或多个 tasks_*.jsonl（可通配），来自 convert_*.py 的输出")
    ap.add_argument("--models", required=True, help="models.yaml")
    ap.add_argument("--config", required=True, help="config.yaml（提供 cap_dims 定义）")
    ap.add_argument("--out", required=True, help="输出：assign_history.synth.jsonl")
    ap.add_argument("--top-p", type=float, default=0.2, help="按相似度前 p 比例打正样本")
    ap.add_argument("--neg-per-pos", type=int, default=2, help="每个正样本配多少个负样本")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    cfg = load_yaml(args.config)
    cap_dims = cfg["cap_dims"]
    dim_types = [d["type"] for d in cap_dims]
    dim_weights = [float(d["weight"]) for d in cap_dims]
    cap_dim_spec = {d["name"]: {"type": d["type"], "weight": d["weight"]} for d in cap_dims}

    models = load_models(args.models)

    # 收集 tasks
    all_tasks: List[Task] = []
    for path in args.tasks:
        for r in read_jsonl(path):
            all_tasks.append(Task(**r))

    synth_rows = []
    for t in all_tasks:
        tv = make_task_vector(t, cap_dim_spec)
        task_vec = tv["vec"]
        tlabel = classify_task_label(t)

        scored = []
        for m in models:
            w1 = w1_multidim(
                model_vec=m["capability"],
                task_vec=task_vec,
                dim_types=dim_types,
                dim_weights=dim_weights
            )
            sim = 1.0 - float(w1)  # 越大越好
            scored.append((sim, m))

        scored.sort(key=lambda x: x[0], reverse=True)
        k_pos = max(1, int(len(scored) * args.top_p))
        pos_pairs = scored[:k_pos]
        neg_cands = scored[k_pos:]
        random.shuffle(neg_cands)
        neg_pairs = neg_cands[:len(pos_pairs) * max(1, args.neg_per_pos)]

        for sim, m in pos_pairs:
            synth_rows.append({
                "task_vec": task_vec,
                "task_label": tlabel,
                "model_cap": m["capability"],
                "model_labels": m["labels"],
                "y": 1,
                "weak_supervision": {"sim": sim}
            })
        for sim, m in neg_pairs:
            synth_rows.append({
                "task_vec": task_vec,
                "task_label": tlabel,
                "model_cap": m["capability"],
                "model_labels": m["labels"],
                "y": 0,
                "weak_supervision": {"sim": sim}
            })

    write_jsonl(args.out, synth_rows)
    print("[synth] wrote", args.out, "n=", len(synth_rows), "from tasks:", len(all_tasks), "models:", len(models))


if __name__ == "__main__":
    main()
