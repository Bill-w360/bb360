# tools/fix_assign.py
# -*- coding: utf-8 -*-
import json, argparse
from typing import Dict, Any, Iterable

def load_jsonl(p: str) -> Iterable[Dict[str, Any]]:
    with open(p, "r", encoding="utf-8") as f:
        for s in f:
            s = s.strip()
            if s:
                yield json.loads(s)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True, help="data/tasks_unified.jsonl")
    ap.add_argument("--assign", required=True, help="data/assign.from_sim_*.jsonl")
    ap.add_argument("--out", required=True, help="data/assign.fixed.jsonl")
    args = ap.parse_args()

    tasks: Dict[str, Dict[str, Any]] = {}
    for ex in load_jsonl(args.tasks):
        tid = ex["id"]
        tasks[tid] = {
            "prompt": ex.get("prompt") or ex.get("question") or "",
            "dataset": ex.get("dataset",""),
            "category": ex.get("category",""),
            "arrival_time": ex.get("arrival_time"),
            "meta": ex.get("meta") or {},
        }

    missing = 0
    with open(args.out, "w", encoding="utf-8") as fout:
        for row in load_jsonl(args.assign):
            tid = row.get("id") or row.get("task_id")
            base = tasks.get(tid)
            if not base:
                missing += 1
                continue
            out = {
                "id": tid,
                "model": row.get("model") or row.get("assign") or row.get("route") or row.get("chosen_model"),
                "dataset": base["dataset"],
                "category": base["category"],
                "arrival_time": base["arrival_time"],
                "prompt": base["prompt"],
                # 把上游 meta 也带上（choices/answer_idx 等）
                "meta": row.get("meta") or base["meta"],
                "score": row.get("score"),
                "queue": row.get("queue"),
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")

    if missing:
        print(f"[fix] skipped {missing} rows (no matching task id)")
    print(f"[fix] wrote {args.out}")

if __name__ == "__main__":
    main()
