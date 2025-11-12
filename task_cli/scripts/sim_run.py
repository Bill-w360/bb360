# scripts/sim_run.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json, math, random, re, os
from typing import Dict, List, Tuple, Callable, Optional

# 项目内工具
from router.utils import load_yaml, read_jsonl
from router.route_score import pick_model, map_difficulty

# -------------------------
# 工具函数
# -------------------------

def pquant(xs: List[float], q: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = min(len(xs) - 1, max(0, int(q * (len(xs) - 1))))
    return float(xs[k])

def jain_index(vals: List[float]) -> float:
    if not vals:
        return 1.0
    s = sum(vals)
    s2 = sum(v * v for v in vals)
    n = len(vals)
    if s2 <= 0:
        return 1.0
    return (s * s) / (n * s2)

# -------------------------
# service 分布解析（更鲁棒）
# -------------------------

def parse_service_specs(specs: List[str]):
    """
    支持三种形式（可多次传参；也可一次给多个）：
      default:<dist>:<p1>[:<p2>...]
      model:<model_name>:<dist>:<p1>[:<p2>...]
      pair:<model_name>,<category>:<dist>:<p1>[:<p2>...]

    dist: const / expo / normal
    - const:v
    - expo:lambda
    - normal:mu:sigma
    """
    def make_sampler(dist: str, *args) -> Callable[[random.Random], float]:
        dist = dist.lower()
        if dist == "const":
            v = float(args[0])
            return lambda rng: v
        if dist == "expo":
            lam = float(args[0])
            return lambda rng: rng.expovariate(lam)
        if dist == "normal":
            mu = float(args[0]); sigma = float(args[1])
            return lambda rng: max(0.0, rng.gauss(mu, sigma))
        raise ValueError(f"unknown dist '{dist}' with args={args}")

    default_sampler = lambda rng: 1.0
    by_model: Dict[str, Callable] = {}
    by_pair: Dict[Tuple[str, str], Callable] = {}

    if not specs:
        return default_sampler, by_model, by_pair

    for raw in specs:
        s = str(raw).strip()
        if not s:
            continue
        if ":" not in s:
            raise ValueError(f"[--service] bad spec (missing ':'): '{s}'")

        head, rest = s.split(":", 1)
        head = head.strip().lower()
        parts = rest.split(":")
        if len(parts) < 2:
            raise ValueError(f"[--service] bad spec (need dist:param): '{s}'")

        if head == "default":
            sampler = make_sampler(parts[0], *parts[1:])
            default_sampler = sampler

        elif head.startswith("model"):
            # model:<name>:<dist>:...
            tmp = s.split(":", 2)
            if len(tmp) < 3:
                raise ValueError(f"[--service] bad model spec: '{s}' (expect model:<name>:<dist>:...)")
            mname = tmp[1].strip()
            rest2 = tmp[2].strip()
            parts2 = rest2.split(":")
            dist = parts2[0].strip()
            params = parts2[1:]
            sampler = make_sampler(dist, *params)
            if not mname:
                raise ValueError(f"[--service] empty model name in '{s}'")
            by_model[mname] = sampler

        elif head.startswith("pair"):
            # pair:<model,category>:<dist>:...
            mc = parts[0].strip()
            mname, cat = None, None
            spl = re.split(r"[,/|]", mc)
            if len(spl) == 2:
                mname = spl[0].strip()
                cat = spl[1].strip()
            else:
                raise ValueError(
                    f"[--service] bad pair head '{mc}' in '{s}'. Expect 'pair:<model,category>:<dist>:...'"
                )
            dist = parts[1].strip()
            params = parts[2:]
            if not mname or not cat:
                raise ValueError(f"[--service] empty model/category in '{s}'")
            sampler = make_sampler(dist, *params)
            by_pair[(mname, cat)] = sampler

        else:
            raise ValueError(f"[--service] unknown head '{head}' in '{s}'")

    return default_sampler, by_model, by_pair

# -------------------------
# 解析 domain 映射
# -------------------------

def parse_domain_mapping(spec: str) -> Tuple[List[str], Dict[str, int]]:
    """
    形如: "math_reasoning:0,commonsense:1,reading_comprehension:2,stem:3"
    返回：names(list in index order), index dict
    """
    if not spec:
        raise ValueError("--domain is required")
    pairs = [p.strip() for p in spec.split(",") if p.strip()]
    idx2name = {}
    for p in pairs:
        if ":" not in p:
            raise ValueError(f"bad domain item '{p}', expect name:index")
        name, idx = p.split(":")
        idx2name[int(idx)] = name.strip()
    names = [idx2name[i] for i in sorted(idx2name.keys())]
    index = {n: i for i, n in enumerate(names)}
    return names, index

# -------------------------
# 统一写行到 dump 文件
# -------------------------

def _write_assign_line(
    fh,
    task: Dict,
    chosen_model_name: str,
    score: Optional[float],
    detail: Optional[Dict],
    task_vec: List[float],
):
    rec = {
        "task_id": task.get("id"),
        "dataset": task.get("dataset"),
        "question": task.get("question") or task.get("input"),
        "options": task.get("options"),
        "gold": task.get("gold"),
        "category": task.get("category"),
        "difficulty": task.get("difficulty"),
        "arrival_time": task.get("arrival_time"),
        "task_vec": task_vec,
        "chosen_model": chosen_model_name,
    }
    # 尽量保留可用的细节
    if score is not None:
        rec["score"] = float(score)
    if isinstance(detail, dict):
        # 兼容 route_score.pick_model 可能返回的 keys
        comps = detail.get("components") or {}
        if comps:
            rec["components"] = comps
        # 如果 detail 里有 per_model（有些实现会带），也可以保留以便分析
        if "per_model" in detail and isinstance(detail["per_model"], list):
            rec["per_model"] = detail["per_model"]

    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

# -------------------------
# 模拟主流程
# -------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Offline router simulation with W1 + cost + k*BP + beta*(1-match) (minimize total)"
    )
    ap.add_argument("--models", required=True, help="models.yaml")
    ap.add_argument("--tasks", required=True, help="unified tasks jsonl")
    ap.add_argument("--domain", required=True,
                    help="name:index mapping, e.g., math_reasoning:0,commonsense:1,... ; "
                         "order must match capability vector")
    
    # 负载均衡与惩罚项
    ap.add_argument("--rho-fair", type=float, default=0.0,
                    help="fairness surcharge weight on queues: q_i += rho * max(0, served_i - avg_served)")
    ap.add_argument("--k", type=float, default=None, help="BP penalty weight (alpha). If set, overrides --alpha.")
    ap.add_argument("--alpha", type=float, default=0.1, help="BP penalty weight (fallback if --k not set)")
    ap.add_argument("--beta", type=float, default=0.5, help="label mismatch weight")
    ap.add_argument("--gamma", type=float, default=0.0, help="cost weight")
    ap.add_argument("--delta", type=float, default=0.7, help="reserved scaling (kept for compatibility)")
    ap.add_argument("--bp-mode", dest="bp_mode", choices=["linear","log","sqrt"], default="linear",
                    help="BP penalty scale")
    ap.add_argument("--lambda-util", type=float, default=0.0)


    # 兼容历史参数名 --shadow
    ap.add_argument("--shadow", choices=["linear","log","sqrt"], help="alias of --bp-mode")
    ap.add_argument("--cost-w", default="tp=1.0",
                    help="comma pairs, e.g. 'tp=1.0,price=0.0,latency=0.0'")
    ap.add_argument("--seed", type=float, default=42)
    ap.add_argument("--service", nargs="+", action="extend",
                    help="service specs: default:expo:1.8 model:xxx:const:1.2 pair:m1,commonsense:normal:1.4:0.3")
    ap.add_argument("--out", default="sim_result.json")
    ap.add_argument("--dump-assign", default=None,
                    help="jsonl path to dump per-task assignment (assign.jsonl style)")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    alpha = args.k if args.k is not None else args.alpha
    bp_mode = args.shadow if args.shadow else args.bp_mode

    # cost 权重解析
    cost_w = {}
    if args.cost_w:
        for item in args.cost_w.split(","):
            item = item.strip()
            if not item:
                continue
            if "=" not in item:
                raise ValueError(f"[--cost-w] bad item '{item}', expect key=val")
            k, v = item.split("=")
            cost_w[k.strip()] = float(v)

    # 载入模型
    y = load_yaml(args.models)
    models = []
    for m in y["models"]:
        models.append({
            "name": m["name"],
            "capability": list(m["capability"]),
            "cost": m.get("cost", {}),
            "queue": {"len": 0.0},  # 仅占位
        })
    if not models:
        raise ValueError("no models in models.yaml")

    dim = len(models[0]["capability"])
    for m in models:
        if len(m["capability"]) != dim:
            raise ValueError(f"model '{m['name']}' capability dim mismatch")

    # 解析 domain 名称与索引，并检查与 dim 一致
    domain_names, domain_index = parse_domain_mapping(args.domain)
    if len(domain_names) != dim:
        raise ValueError(f"--domain has {len(domain_names)} dims but model capability has {dim} dims")

    # 载入任务
    tasks = [r for r in read_jsonl(args.tasks)]
    if not tasks:
        raise ValueError("no tasks loaded")

    # 类别校验
    unknown = [t for t in tasks if t.get("category") not in domain_index]
    if unknown:
        bad_keys = sorted({t.get("category") for t in unknown})
        raise RuntimeError(f"[error] {len(unknown)} tasks use unknown category keys: {bad_keys[:4]} ... "
                           f"Please align --domain with your task 'category'.")

    # service 分布
    default_sampler, by_model, by_pair = parse_service_specs(args.service or [])

    # 初始化状态
    avail_time: Dict[str, float] = {m["name"]: 0.0 for m in models}  # 单服务器可用时间（时刻）
    busy_time: Dict[str, float] = {m["name"]: 0.0 for m in models}   # 总忙碌时长
    served: Dict[str, int] = {m["name"]: 0 for m in models}

    waits: List[float] = []
    sojs: List[float] = []

    # 维度类型/权重（与 capability 等长；仿真里统一 cap/1.0）
    dim_types = ["cap"] * dim
    dim_weights = [1.0] * dim

    # 可选：逐任务分配 dump
    dump_f = None
    if args.dump_assign:
        os.makedirs(os.path.dirname(args.dump_assign), exist_ok=True)
        dump_f = open(args.dump_assign, "w", encoding="utf-8")

    # 仿真（按 arrival_time 已经是非降序；如果无则按顺序）
    tasks_sorted = sorted(tasks, key=lambda x: float(x.get("arrival_time", 0.0)))

    for t in tasks_sorted:
        now = float(t.get("arrival_time", 0.0))
        cat = t["category"]
        diff = t.get("difficulty", "medium")

        # 任务向量（one-hot*难度系数）
        tv = [0.0] * dim
        idx = domain_index[cat]
        if isinstance(diff, (int, float)):
            dscore = float(diff)
        else:
            dscore = map_difficulty(str(diff))
        tv[idx] = dscore

        # 队列度量（以“时间积压”表示）：q_i = max(0, avail_time - now)
        queues = {m["name"]: max(0.0, avail_time[m["name"]] - now) for m in models}

        # === 公平附加队列：对“接单数高于平均”的模型，增加一点等价队列 ===
        if args.rho_fair and len(models) > 1:
            avg_served = sum(served.values()) / max(1, len(served))
            queues = {
                name: q + args.rho_fair * max(0.0, served[name] - avg_served)
                for name, q in queues.items()
            }
        # === 这样后续的 bp_penalty(queues) 会自动更惩罚“过度服务”的模型 ===

        # 利用率均衡附加（很小的 λ，默认 0，按需开启）
        if args.lambda_util and len(models) > 1:
            # 近似当前时刻的“已知利用率”：busy_time / max(now, 1)
            denom = max(1e-6, now)
            util_now = {n: (busy_time[n] / denom) for n in busy_time}
            avg_util = sum(util_now.values()) / max(1, len(util_now))
            queues = {name: q + args.lambda_util * max(0.0, util_now[name] - avg_util)
                    for name, q in queues.items()}

        # （可选）标签匹配，这里离线模拟默认不用 -> 传 None
        chosen, total, detail = pick_model(
            models=models,
            task_vec=tv,
            dim_types=dim_types,
            dim_weights=dim_weights,
            cost_w=cost_w,
            queues=queues,
            alpha=alpha, beta=args.beta, gamma=args.gamma, delta=args.delta,
            match_vals=None,
            bp_mode=bp_mode,
        )

        # 选择服务时间分布
        sampler = by_pair.get((chosen, cat), None) or by_model.get(chosen, None) or default_sampler
        st = float(sampler(rng))

        # 调度：单服务器 FIFO
        start = max(now, avail_time[chosen])
        finish = start + st
        wait = start - now
        soj = finish - now

        # 更新
        avail_time[chosen] = finish
        busy_time[chosen] += st
        served[chosen] += 1
        waits.append(wait)
        sojs.append(soj)

        # >>> 关键：逐任务分配 dump（assign.jsonl 风格） <<<
        if dump_f:
            _write_assign_line(
                dump_f, task=t, chosen_model_name=chosen,
                score=total, detail=detail, task_vec=tv
            )


    # 汇总
    t_end = max(avail_time.values()) if avail_time else 0.0
    util = {k: (busy_time[k] / t_end if t_end > 0 else 0.0) for k in busy_time}
    jain = jain_index(list(served.values()))
    qvar_end = 0.0  # 以时间积压为队列度量，结束时刻都被“消化”为 0（离线批处理）

    out = {
        "tasks_total": len(tasks),
        "fairness_jain": jain,
        "queue_variance_end": qvar_end,
        "wait_p50": pquant(waits, 0.50),
        "wait_p90": pquant(waits, 0.90),
        "wait_p99": pquant(waits, 0.99),
        "soj_p50": pquant(sojs, 0.50),
        "soj_p90": pquant(sojs, 0.90),
        "soj_p99": pquant(sojs, 0.99),
        "per_model_served": served,
        "per_model_busy": busy_time,
        "per_model_util": util,
        "t_end": t_end,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    if dump_f:
        dump_f.close()

    print(f"[sim] wrote {args.out}")
    if args.dump_assign:
        print(f"[sim] dumped per-task assignment to {args.dump_assign}")
    print(json.dumps(out, ensure_ascii=False, indent=2))



if __name__ == "__main__":
    main()
