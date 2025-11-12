
# -*- coding: utf-8 -*-
"""
audit_label_gap.py  (fixed & enhanced)
Compare:
  A) model predictions (_pred_super) vs
  B) "silver" labels (auto/mapped) and
  C) "gold" labels (human)

Outputs accuracies, confusion matrices, and a CSV of disagreements (w.r.t. gold).
"""
import argparse, json, csv
from pathlib import Path
from typing import List, Dict, Any, Optional

SUP = [
    "math_reasoning","commonsense","reading_comprehension","general_knowledge",
    "humanities","social_science","stem","other_knowledge"
]

def load_jsonl(p: Path) -> List[Dict[str, Any]]:
    rows=[]
    with p.open("r",encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def confusion(true_labels: List[str], pred_labels: List[str], labels: List[str]):
    mat = {(t,p):0 for t in labels for p in labels}
    for t,p in zip(true_labels, pred_labels):
        if t in labels and p in labels:
            mat[(t,p)] += 1
    return mat

def pretty_conf(mat, labels: List[str]):
    rows = []
    header = ["true\\pred"] + labels
    rows.append(header)
    for t in labels:
        row = [t]
        for p in labels:
            row.append(str(mat[(t,p)]))
        rows.append(row)
    return rows

def pick_field(row: Dict[str, Any], candidates: List[str]) -> Optional[str]:
    for k in candidates:
        if k in row and row[k]:
            return row[k]
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="pred JSONL with _pred_super")
    ap.add_argument("--jsonl", required=True, help="JSONL that may contain silver/gold label fields")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--silver-field", default=None, help="explicit field name for silver label, e.g., label_sup or label_sup_silver")
    ap.add_argument("--gold-field", default=None, help="explicit field name for gold label, e.g., label_sup")
    args = ap.parse_args()

    pred_path = Path(args.pred)
    data_path = Path(args.jsonl)
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    preds = load_jsonl(pred_path)
    data  = load_jsonl(data_path)
    n = min(len(preds), len(data))

    y_pred = [preds[i].get("_pred_super") for i in range(n)]

    silver_candidates = [args.silver_field] if args.silver_field else ["label_sup_silver","label_sup_auto","label_sup"]
    gold_candidates   = [args.gold_field]   if args.gold_field   else ["label_sup"]

    y_silver = [pick_field(data[i], silver_candidates) for i in range(n)]
    y_gold   = [pick_field(data[i], gold_candidates)   for i in range(n)]

    def acc(ytrue, yhat):
        tot = 0; hit = 0
        for t,h in zip(ytrue, yhat):
            if not t:
                continue
            tot += 1
            if t == h:
                hit += 1
        return (hit/tot if tot>0 else 0.0), tot

    acc_silver, tot_silver = acc(y_silver, y_pred)
    acc_gold,   tot_gold   = acc(y_gold,   y_pred)

    def acc_pair(a, b):
        tot=0; hit=0
        for x,y in zip(a,b):
            if not x or not y:
                continue
            tot += 1
            if x == y:
                hit += 1
        return (hit/tot if tot>0 else 0.0), tot
    acc_sg, tot_sg = acc_pair(y_gold, y_silver)

    def conf_with(ytrue, yhat):
        T = [t for t in ytrue if t]
        H = [h for (t,h) in zip(ytrue,yhat) if t]
        return confusion(T, H, SUP) if T else None

    conf_pred_silver = conf_with(y_silver, y_pred) if tot_silver else None
    conf_pred_gold   = conf_with(y_gold,   y_pred) if tot_gold else None
    conf_silver_gold = conf_with(y_gold,   y_silver) if tot_sg else None

    def write_conf(mat, name):
        if not mat: return
        rows = pretty_conf(mat, SUP)
        with (outdir / f"confusion_{name}.csv").open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(rows)

    write_conf(conf_pred_silver, "pred_vs_silver")
    write_conf(conf_pred_gold,   "pred_vs_gold")
    write_conf(conf_silver_gold, "silver_vs_gold")

    with (outdir / "disagreements.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["row_id","gold","silver","pred"])
        for i in range(n):
            g = y_gold[i]; s = y_silver[i]; p = y_pred[i]
            if g and p != g:
                w.writerow([i, g or "", s or "", p or ""])

    with (outdir / "summary.txt").open("w", encoding="utf-8") as f:
        f.write(f"pred vs silver: acc={acc_silver:.4f} (n={tot_silver}) fields={silver_candidates}\n")
        f.write(f"pred vs gold  : acc={acc_gold:.4f} (n={tot_gold}) fields={gold_candidates}\n")
        f.write(f"silver vs gold: acc={acc_sg:.4f} (n={tot_sg})\n")
        f.write(f"aligned rows  : n={n} (preds={len(preds)}, data={len(data)})\n")

    print(f"[done] wrote: {outdir}/summary.txt and confusion_*.csv, disagreements.csv")
if __name__ == "__main__":
    main()
