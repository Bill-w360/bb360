
# -*- coding: utf-8 -*-
"""
score_sup_accuracy_gold.py
Score 8-class predictions against per-row gold labels ('label_sup').
Assumes you labeled the SAME normalized JSONL (line-aligned).

Usage:
  python score_sup_accuracy_gold.py --pred eval_out/preds_boolq.jsonl --jsonl eval_out/boolq.norm.jsonl

If you only labeled a subset (via template CSV), pass --label-csv and we will score only those rows.
"""
import argparse, json, csv
from pathlib import Path
from collections import Counter

def load_jsonl(p):
    rows=[]
    with open(p,"r",encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="pred JSONL (with _pred_super)")
    ap.add_argument("--jsonl", required=True, help="the normalized JSONL used for inference; if it contains label_sup, we'll use it")
    ap.add_argument("--label-csv", default=None, help="optional: a CSV (from make_sup_label_template.py) with a subset gold labels")
    args = ap.parse_args()

    preds = load_jsonl(args.pred)
    data  = load_jsonl(args.jsonl)

    if len(preds) != len(data):
        print(f"[warn] length mismatch: preds={len(preds)} vs jsonl={len(data)}; will align by row_id if CSV provided, else min length.")
    has_inline = any(("label_sup" in r and r["label_sup"]) for r in data)

    subset_idx = None
    gold_map = {}
    if args.label_csv:
        subset_idx = []
        with open(args.label_csv, "r", encoding="utf-8") as f:
            for i, row in enumerate(csv.DictReader(f)):
                if not row.get("label_sup"):
                    continue
                rid = int(row["row_id"])
                subset_idx.append(rid)
                gold_map[rid] = row["label_sup"].strip()
        subset_idx = sorted(set(subset_idx))

    hits = 0
    total = 0
    conf = Counter()

    def get_gold(i):
        if subset_idx is not None:
            return gold_map.get(i)
        if has_inline:
            return data[i].get("label_sup")
        return None

    for i in (subset_idx if subset_idx is not None else range(min(len(preds), len(data)))):
        gold = get_gold(i)
        if not gold:
            continue
        pred = preds[i].get("_pred_super")
        conf[pred] += 1
        total += 1
        if pred == gold:
            hits += 1

    acc = (hits/total) if total>0 else 0.0
    print(f"gold-labeled samples: {total}")
    print(f"accuracy: {acc:.4f}")
    print("--- prediction distribution ---")
    for k, v in conf.most_common():
        print(f"{k:>24s} : {v}")

if __name__ == "__main__":
    main()
