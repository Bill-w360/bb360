# router/cli.py
# -*- coding: utf-8 -*-
"""
命令行工具（统一入口）
- assign        : 离线批量打分与分配（本地计算，兼容你现有流程）
- offer         : 向 Broker 投放单条任务（在线路由，只有空闲模型会参与计算并抢单）
- offer-batch   : 批量向 Broker 投放 tasks.jsonl
- dump-models   : 打印 models.yaml 里的模型及能力维度长度（便于自检）
"""
import argparse
from typing import List, Dict, Tuple, Union
import requests
import sys

from .utils import read_jsonl, write_jsonl, load_yaml
from .schemas import Task, ModelInfo, RouteConfig
from .classifier import make_task_vector, classify_task_label
from .scorer import score_one
from .label_matcher import BilinearMatcher, SemanticMatcher


# ----------------------
# 配置与模型加载
# ----------------------
def load_config(cfg_path: str) -> RouteConfig:
    raw = load_yaml(cfg_path)
    cap_dims = raw["cap_dims"]  # list of {name,type,weight}
    for d in cap_dims:
        assert "name" in d and "type" in d and "weight" in d, "cap_dims item needs name/type/weight"
    return RouteConfig(
        cap_dims=cap_dims,
        alpha=float(raw["alpha"]),
        beta=float(raw["beta"]),
        gamma=float(raw["gamma"]),
        delta=float(raw.get("delta", 0.7)),
        cost_weights=raw["cost_weights"],
        backpressure=raw["backpressure"],
        label_matcher=raw.get("label_matcher", {}),
    )


def parse_models(models_yaml: str) -> List[ModelInfo]:
    y = load_yaml(models_yaml)
    out = []
    for m in y["models"]:
        out.append(
            ModelInfo(
                name=m["name"],
                labels=m.get("labels", []),
                capability=m["capability"],
                cost=m["cost"],
                queue=m.get("queue", {}),
            )
        )
    return out


# ----------------------
# 子命令：离线 assign（本地计算）
# ----------------------
def cmd_assign(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    dim_names = [d["name"] for d in cfg.cap_dims]
    dim_types = [d["type"] for d in cfg.cap_dims]
    dim_weights = [float(d["weight"]) for d in cfg.cap_dims]

    # label matcher
    lm_conf = cfg.label_matcher or {}
    lm_mode = lm_conf.get("mode", "bilinear")
    matcher = None
    if lm_mode == "bilinear":
        dim = len(dim_names)
        weights_path = lm_conf.get("weights_path", "checkpoints/label_matcher_bilinear.json")
        matcher = BilinearMatcher(dim=dim, weights_path=weights_path)
    elif lm_mode == "semantic":
        matcher = SemanticMatcher()

    models = parse_models(args.models)
    rows = read_jsonl(args.inp)
    tasks = [Task(**r) for r in rows]

    cap_dim_spec = {d["name"]: {"type": d["type"], "weight": d["weight"]} for d in cfg.cap_dims}

    assigns: List[Dict] = []
    for t in tasks:
        tv = make_task_vector(t, cap_dim_spec)
        task_vec = tv["vec"]
        task_label = classify_task_label(t)

        best: Tuple[str, Dict, float] = None  # (model_name, comp_dict, match_val)
        per_model: List[Dict] = []

        for m in models:
            if len(m.capability) != len(task_vec):
                raise ValueError(f"model {m.name} capability dim != task dim")

            # 模型“自算匹配值”的等价实现：这里仍由 router 离线代入 matcher，
            # 线上场景则由 worker 本地计算；两者公式一致。
            if isinstance(matcher, BilinearMatcher):
                match_val = matcher.match(task_vec, m.capability)
            else:
                ttxt = task_label
                mtxt = " ".join(m.labels)
                match_val = matcher.match(ttxt, mtxt)

            comp = score_one(
                model_vec=m.capability,
                task_vec=task_vec,
                dim_types=dim_types,
                dim_weights=dim_weights,
                cost=m.cost,
                cost_w=cfg.cost_weights,
                queue=m.queue,
                bp_cfg=cfg.backpressure,
                alpha=cfg.alpha,
                beta=cfg.beta,
                gamma=cfg.gamma,
                delta=cfg.delta,
                match_val=match_val,
            )
            per_model.append({"model": m.name, "match_val": match_val, **comp})
            if (best is None) or (comp["total"] < best[1]["total"]):
                best = (m.name, comp, match_val)

        assigns.append(
            {
                "task_id": t.id,
                "task_label": task_label,
                "task_domain": tv["domain"],
                "task_difficulty": tv["difficulty"],
                "task_vec": task_vec,
                "chosen_model": best[0],
                "score": best[1]["total"],
                "components": {
                    "w1": best[1]["w1"],
                    "cost": best[1]["cost"],
                    "shadow": best[1]["shadow"],
                    "label_mismatch": best[1]["label_mismatch"],
                },
                "match_val": best[2],
                "per_model": per_model,
            }
        )

    write_jsonl(args.out, assigns)
    print(f"[done] wrote {args.out}; {len(assigns)} assignments")


# ----------------------
# 子命令：向 Broker 投单（在线）
# ----------------------
def cmd_offer(args: argparse.Namespace) -> None:
    base = args.broker.rstrip("/")
    payload = {"category": args.category, "difficulty": args.difficulty}
    try:
        r = requests.post(f"{base}/task/offer", json=payload, timeout=5)
        r.raise_for_status()
    except requests.RequestException as e:
        print(f"[error] offer failed: {e}", file=sys.stderr)
        sys.exit(2)
    print(r.json())


def cmd_offer_batch(args: argparse.Namespace) -> None:
    base = args.broker.rstrip("/")
    rows = read_jsonl(args.inp)
    ok, fail = 0, 0
    for row in rows:
        # 允许两种输入：
        # 1) 你的 Task JSON 结构（含 domain/difficulty）
        # 2) 极简 {"category": "...", "difficulty": "..."}
        category = row.get("domain") or row.get("category")
        difficulty = row.get("difficulty") or row.get("task_difficulty") or "medium"
        if not category:
            print(f"[warn] skip row without category/domain: {row}", file=sys.stderr)
            fail += 1
            continue
        try:
            r = requests.post(
                f"{base}/task/offer", json={"category": category, "difficulty": difficulty}, timeout=5
            )
            r.raise_for_status()
            ok += 1
        except requests.RequestException as e:
            print(f"[warn] offer failed: {e} row={row}", file=sys.stderr)
            fail += 1
    print(f"[done] offered {ok} tasks; failed {fail}")


# ----------------------
# 子命令：自检 models.yaml
# ----------------------
def cmd_dump_models(args: argparse.Namespace) -> None:
    models = parse_models(args.models)
    dim_lens = {m.name: len(m.capability) for m in models}
    print("models:", ", ".join(m.name for m in models))
    for m in models:
        print(f"- {m.name:20s} dim={len(m.capability)} labels={m.labels}")


# ----------------------
# 入口
# ----------------------
def main():
    ap = argparse.ArgumentParser(
        description="Router CLI (offline assign + online broker offers)"
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    # assign（离线批处理，兼容你现有脚本）
    ap_assign = sub.add_parser("assign", help="offline batch assign with local scoring")
    ap_assign.add_argument("--in", dest="inp", required=True, help="input tasks.jsonl")
    ap_assign.add_argument("--config", default="config.yaml")
    ap_assign.add_argument("--models", default="models.yaml")
    ap_assign.add_argument("--out", default="assign.jsonl")
    ap_assign.set_defaults(func=cmd_assign)

    # offer（单条在线投单）
    ap_offer = sub.add_parser("offer", help="offer one task to broker (online)")
    ap_offer.add_argument("--broker", default="http://127.0.0.1:8000")
    ap_offer.add_argument("--category", required=True)
    ap_offer.add_argument("--difficulty", required=True)
    ap_offer.set_defaults(func=cmd_offer)

    # offer-batch（批量在线投单）
    ap_offer_b = sub.add_parser("offer-batch", help="offer tasks.jsonl to broker (online)")
    ap_offer_b.add_argument("--broker", default="http://127.0.0.1:8000")
    ap_offer_b.add_argument("--in", dest="inp", required=True, help="tasks.jsonl")
    ap_offer_b.set_defaults(func=cmd_offer_batch)

    # dump-models（自检）
    ap_dm = sub.add_parser("dump-models", help="print models and capability dims")
    ap_dm.add_argument("--models", default="models.yaml")
    ap_dm.set_defaults(func=cmd_dump_models)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
