# extras/split_assign_by_model.py
# -*- coding: utf-8 -*-
import os, json, argparse, collections
from typing import Dict, Any, Iterable

def read_jsonl(p: str) -> Iterable[Dict[str, Any]]:
    with open(p, "r", encoding="utf-8") as f:
        for s in f:
            s = s.strip()
            if s:
                yield json.loads(s)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def pack_for_batch(task: Dict[str, Any]) -> Dict[str, Any]:
    """把 assign 中的任务行转换为 infer_local 可直接使用的一行。"""
    ds = (task.get("dataset") or "").lower()
    meta = task.get("meta") or {}

    ex = {
        "id": task["id"],
        "dataset": task.get("dataset",""),
        # 关键：透传 prompt，并兼容 question 字段
        "prompt": task.get("prompt",""),
        "question": task.get("prompt","") or task.get("question",""),
    }

    if ds in ("hellaswag","mmlu"):
        choices = meta.get("choices") or task.get("options") or []
        ex["options"] = choices
        ans_idx = meta.get("answer_idx")
        if isinstance(ans_idx, int) and 0 <= ans_idx < 26:
            ex["gold"] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[ans_idx]
        else:
            # 如已是 "A"/"B"/"C"/"D"
            ex["gold"] = (meta.get("answer") or task.get("gold"))
    elif ds == "boolq":
        gold = meta.get("answer")
        if isinstance(gold, bool):
            ex["gold"] = "yes" if gold else "no"
        else:
            ex["gold"] = task.get("gold")
    elif ds == "gsm8k":
        ex["gold"] = meta.get("answer") or task.get("gold")
    return ex

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="assign.jsonl（每行至少含 id, model）")
    ap.add_argument("--out-dir", required=True, help="输出目录：work/batches/")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    groups = collections.defaultdict(list)

    for row in read_jsonl(args.inp):
        model = row.get("model") or row.get("assign") or row.get("route")
        if not model:
            # 丢弃没有分配模型的样本
            continue
        groups[model].append(pack_for_batch(row))

    for model, items in groups.items():
        safe = model.replace("/", "_").replace(":", "_")
        outp = os.path.join(args.out_dir, f"batch_{safe}.jsonl")
        with open(outp, "w", encoding="utf-8") as f:
            for ex in items:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"[split] {model} -> {outp}  n={len(items)}")

if __name__ == "__main__":
    main()
