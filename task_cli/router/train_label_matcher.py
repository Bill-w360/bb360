# router/train_label_matcher.py
# -*- coding: utf-8 -*-
"""
Train learnable label matcher (bilinear) from:
  (A) paired jsonl files (fields: task_vec/model_cap/y)
  (B) assign.jsonl + models.yaml produced by router.cli

Enhancements:
- Auto sys.path injection (no PYTHONPATH pains)
- Auto infer dim from config.yaml (--from-config)
- Vector dim guard + optional pad/trunc to fit
- Safe sigmoid, grad clipping, cosine lr scheduler
- Optional validation split with logloss report
- Detailed data loading diagnostics
"""

# ===== sys.path 自救：保证能 import 项目内模块 =====
import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# =================================================

import json, argparse, glob, random, math
from typing import Iterable, Dict, List, Tuple, Optional

from .label_matcher import BilinearMatcher
from .utils import load_yaml

# ---------- I/O 基础 ----------

def read_jsonl(path: str) -> Iterable[Dict]:
    bad = 0
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except json.JSONDecodeError as e:
                bad += 1
                if bad <= 3:
                    print(f"[warn] bad json at {path}:{ln}: {e} -> {s[:120]}...")
                elif bad == 4:
                    print(f"[warn] ...more bad json lines suppressed for {path}")

def expand_paths(patterns: Optional[List[str]]) -> List[str]:
    if not patterns:
        return []
    files = []
    for p in patterns:
        hits = glob.glob(p)
        if hits:
            files.extend(sorted(hits))
        elif os.path.isfile(p):
            files.append(p)
        else:
            print(f"[warn] no match for: {p}")
    # 去重并保持顺序
    seen, uniq = set(), []
    for fp in files:
        if fp not in seen:
            uniq.append(fp); seen.add(fp)
    return uniq

# ---------- 从 paired jsonl 读监督对 ----------

def load_paired_dataset(paths: List[str], max_samples: int = 0, shuffle: bool = True) -> List[Dict]:
    data, skipped = [], 0
    for p in paths:
        for ex in read_jsonl(p):
            if "task_vec" in ex and "model_cap" in ex and "y" in ex:
                try:
                    yv = float(ex["y"])
                except Exception:
                    yv = 1.0 if str(ex["y"]).lower() in ("true", "1") else 0.0
                data.append({
                    "task_vec": ex["task_vec"],
                    "model_cap": ex["model_cap"],
                    "y": yv
                })
            else:
                skipped += 1
    if shuffle:
        random.shuffle(data)
    if max_samples and len(data) > max_samples:
        data = data[:max_samples]
    print(f"[info] paired: loaded {len(data)} examples from {len(paths)} files (skipped={skipped})")
    return data

# ---------- 从 assign.jsonl + models.yaml 生成监督对 ----------

def load_models_capability(models_yaml: str) -> Dict[str, List[float]]:
    y = load_yaml(models_yaml)
    name2cap = {}
    for m in y.get("models", []):
        name2cap[m["name"]] = m["capability"]
    return name2cap

def paired_from_assign(assign_files: List[str],
                       models_yaml: str,
                       hard_neg_k: int = 0,
                       pos_weight: float = 1.0,
                       neg_weight: float = 1.0) -> List[Dict]:
    """
    正样本：chosen_model
    负样本：per_model 中除 chosen 之外的模型；若 hard_neg_k>0，取 total 最小的前 k 个
    """
    name2cap = load_models_capability(models_yaml)
    pairs = []
    total_tasks, bad_lines = 0, 0

    for fp in assign_files:
        for ex in read_jsonl(fp):
            total_tasks += 1
            task_vec = ex.get("task_vec")
            chosen = ex.get("chosen_model")
            per_model = ex.get("per_model", [])
            if not isinstance(task_vec, list) or not chosen or not per_model:
                bad_lines += 1
                continue

            # 正样本
            cap = name2cap.get(chosen)
            if cap is not None:
                pairs.append({"task_vec": task_vec, "model_cap": cap, "y": 1.0, "_w": pos_weight})
            else:
                print(f"[warn] chosen_model '{chosen}' not in models.yaml; skip pos")

            # 负样本候选
            negs: List[Tuple[float, Dict]] = []
            for pm in per_model:
                mname = pm.get("model")
                if not mname or mname == chosen:
                    continue
                mcap = name2cap.get(mname)
                if not mcap:
                    print(f"[warn] model '{mname}' not in models.yaml; skip neg")
                    continue
                total = float(pm.get("total", 0.0))
                negs.append((total, {"task_vec": task_vec, "model_cap": mcap, "y": 0.0, "_w": neg_weight}))

            if hard_neg_k and len(negs) > hard_neg_k:
                negs = sorted(negs, key=lambda x: x[0])[:hard_neg_k]
            for _, item in negs:
                pairs.append(item)

    print(f"[info] assign: built {len(pairs)} examples from {len(assign_files)} files (tasks={total_tasks}, bad={bad_lines})")
    return pairs

# ---------- 维度处理与校验 ----------

def infer_dim_from_config(config_yaml: str) -> int:
    y = load_yaml(config_yaml)
    cap_dims = y.get("cap_dims", [])
    if not cap_dims:
        raise ValueError("config.yaml missing 'cap_dims'")
    return len(cap_dims)

def ensure_dim(v: List[float], dim: int, mode: str) -> List[float]:
    """
    mode = 'strict' | 'pad' | 'trunc' | 'pad_or_trunc'
    """
    n = len(v)
    if n == dim:
        return v
    if mode == "strict":
        raise ValueError(f"vector dim mismatch: got {n}, expect {dim}")
    if n < dim and mode in ("pad", "pad_or_trunc"):
        return v + [0.0] * (dim - n)
    if n > dim and mode in ("trunc", "pad_or_trunc"):
        return v[:dim]
    # fallback：仍抛错
    raise ValueError(f"vector dim mismatch: got {n}, expect {dim} (mode={mode})")

def harmonize_dims(data: List[Dict], dim: int, mode: str) -> List[Dict]:
    fixed, bad = [], 0
    for ex in data:
        try:
            tv = ensure_dim([float(x) for x in ex["task_vec"]], dim, mode)
            mv = ensure_dim([float(x) for x in ex["model_cap"]], dim, mode)
            y = float(ex["y"])
            w = float(ex.get("_w", 1.0))
            fixed.append({"task_vec": tv, "model_cap": mv, "y": y, "_w": w})
        except Exception as e:
            bad += 1
            if bad <= 3:
                print(f"[warn] drop one example for dim issue: {e}")
    if bad > 3:
        print(f"[warn] ... dropped {bad} examples due to dim issues")
    return fixed

# ---------- 训练与评估 ----------

def safe_sigmoid(z: float) -> float:
    # 数值稳定 sigmoid
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    else:
        ez = math.exp(z)
        return ez / (1.0 + ez)

def logloss(y: float, p: float, eps: float = 1e-7) -> float:
    p = min(max(p, eps), 1.0 - eps)
    return -(y * math.log(p) + (1.0 - y) * math.log(1.0 - p))

def evaluate_logloss(model: BilinearMatcher, data: List[Dict]) -> float:
    if not data:
        return float("nan")
    s = 0.0
    for ex in data:
        z = model.score_raw(ex["task_vec"], ex["model_cap"])
        p = safe_sigmoid(z)
        s += logloss(ex["y"], p)
    return s / len(data)

def cosine_lr(base_lr: float, step: int, total_steps: int, min_lr_ratio: float = 0.05) -> float:
    if total_steps <= 0:
        return base_lr
    cos_inner = math.pi * min(step, total_steps) / total_steps
    return (base_lr - base_lr * min_lr_ratio) * (1 + math.cos(cos_inner)) / 2 + base_lr * min_lr_ratio

# ---------- 主流程 ----------

def main():
    ap = argparse.ArgumentParser(description="Train bilinear label matcher from paired data and/or assign.jsonl")
    # 数据来源
    ap.add_argument("--assign", nargs="*", help="assign.jsonl file(s) or globs from router.cli output")
    ap.add_argument("--models", default=None, help="models.yaml (required if --assign is used)")
    ap.add_argument("--data", nargs="*", help="paired jsonl files or globs (fields: task_vec/model_cap/y)")
    # 维度
    ap.add_argument("--dim", type=int, default=0, help="embedding dimension; if 0 and --from-config is set, inferred from config")
    ap.add_argument("--from-config", default=None, help="config.yaml to infer dim from cap_dims (used if --dim==0)")
    ap.add_argument("--pad-or-trunc", action="store_true", help="auto fix dim mismatch by padding/truncating")
    # 训练
    ap.add_argument("--weights", default="checkpoints/label_matcher_bilinear.json")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--no-shuffle", action="store_true")
    ap.add_argument("--grad-clip", type=float, default=1.0, help="L-infty grad clip per weight (0=disable)")
    ap.add_argument("--cosine-lr", action="store_true", help="enable cosine lr schedule")
    # assign 负样本控制
    ap.add_argument("--hard-neg-k", type=int, default=0, help="use top-k hardest negatives per task when building from assign")
    ap.add_argument("--pos-weight", type=float, default=1.0)
    ap.add_argument("--neg-weight", type=float, default=1.0)
    # 验证
    ap.add_argument("--val-ratio", type=float, default=0.0, help="0~0.5; if >0, split a validation set")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    # 解析路径与加载数据
    paired_files = expand_paths(args.data)
    assign_files = expand_paths(args.assign)

    data: List[Dict] = []
    if assign_files:
        if not args.models:
            raise ValueError("--models is required when using --assign")
        data.extend(paired_from_assign(
            assign_files=assign_files,
            models_yaml=args.models,
            hard_neg_k=args.hard_neg_k,
            pos_weight=args.pos_weight,
            neg_weight=args.neg_weight,
        ))
    if paired_files:
        data.extend(load_paired_dataset(
            paths=paired_files,
            max_samples=0,  # 合并后再统一截断
            shuffle=not args.no_shuffle
        ))

    if not data:
        raise FileNotFoundError("No training data: provide --assign and/or --data")

    # 维度推断
    dim = args.dim
    if dim <= 0:
        if not args.from_config:
            raise ValueError("--dim is 0 and --from-config not provided; one of them must be set")
        dim = infer_dim_from_config(args.from_config)
        print(f"[info] infer dim={dim} from {args.from_config}")

    # 统一处理维度
    mode = "pad_or_trunc" if args.pad_or_trunc else "strict"
    data = harmonize_dims(data, dim, mode=mode)

    # 打乱与采样上限
    if not args.no_shuffle:
        random.shuffle(data)
    if args.max_samples and len(data) > args.max_samples:
        data = data[:args.max_samples]

    # train/val split
    val_set: List[Dict] = []
    if args.val_ratio > 1e-9:
        vr = max(0.0, min(args.val_ratio, 0.5))
        n_val = int(len(data) * vr)
        val_set = data[:n_val]
        data = data[n_val:]
        print(f"[info] split train={len(data)}, val={len(val_set)} (ratio={vr})")

    # 准备模型与路径
    os.makedirs(os.path.dirname(args.weights), exist_ok=True)
    bm = BilinearMatcher(dim, args.weights)

    # 计算总步数（用于余弦退火）
    total_steps = max(1, args.epochs * len(data))

    # 训练
    step = 0
    base_lr = args.lr
    for epoch in range(1, args.epochs + 1):
        if not args.no_shuffle:
            random.shuffle(data)
        running_loss = 0.0

        for ex in data:
            # 学习率调度
            lr = cosine_lr(base_lr, step, total_steps) if args.cosine_lr else base_lr

            t = ex["task_vec"]; m = ex["model_cap"]; y = float(ex["y"])
            z = bm.score_raw(t, m)
            p = safe_sigmoid(z)
            w = float(ex.get("_w", 1.0))
            # dL/dz = (p - y)
            g = (p - y) * w

            # SGD + L2 + grad clip
            for i in range(bm.dim):
                ti = t[i]
                if ti == 0:
                    continue
                row = bm.W[i]
                for j in range(bm.dim):
                    grad = g * ti * m[j] + args.l2 * row[j]
                    if args.grad_clip > 0:
                        if grad > args.grad_clip: grad = args.grad_clip
                        elif grad < -args.grad_clip: grad = -args.grad_clip
                    row[j] -= lr * grad
            bm.b -= lr * g

            running_loss += logloss(y, p)
            step += 1

        train_ll = running_loss / max(1, len(data))
        if val_set:
            val_ll = evaluate_logloss(bm, val_set)
            print(f"[epoch {epoch:02d}] train_logloss={train_ll:.6f}  val_logloss={val_ll:.6f}  lr={lr:.6g}")
        else:
            print(f"[epoch {epoch:02d}] train_logloss={train_ll:.6f}  lr={lr:.6g}")

    bm.save(args.weights)
    print("[train] saved:", args.weights, "with", len(data), "train ex", f"+ {len(val_set)} val ex" if val_set else "")

if __name__ == "__main__":
    main()
