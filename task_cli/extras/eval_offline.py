# -*- coding: utf-8 -*-
import argparse, json, math, collections

def read_jsonl(p):
    with open(p, 'r', encoding='utf-8') as f:
        for s in f:
            s = s.strip()
            if s: yield json.loads(s)

def norm_bool(x):
    t = (x or "").strip().lower()
    if t in ("true","yes","y","1"): return "yes"
    if t in ("false","no","n","0"): return "no"
    return t

def is_num(s):
    try:
        float(s); return True
    except: return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True, help="tasks_unified.jsonl（需含 id,dataset,gold）")
    ap.add_argument("--pred", nargs="+", required=True, help="pred_*.jsonl（可多个，自动合并）")
    ap.add_argument("--out", default="eval_report.json")
    args = ap.parse_args()

    gold = {}
    for ex in read_jsonl(args.gold):
        gid = ex.get("id")
        if gid is None: continue
        gold[gid] = {"dataset": ex.get("dataset","").lower(),
                     "gold": str(ex.get("gold")) if ex.get("gold") is not None else None}

    pred = {}
    for p in args.pred:
        for r in read_jsonl(p):
            pid = r.get("id")
            if pid is not None: pred[pid] = r

    total = 0; correct = 0
    by_ds = collections.defaultdict(lambda: {"n":0, "c":0})

    for gid, g in gold.items():
        if gid not in pred: continue
        ds = g["dataset"]
        gold_val = g["gold"]
        pv = pred[gid].get("pred")
        if pv is None or gold_val is None: continue

        ok = False
        if ds == "boolq":
            ok = (norm_bool(pv) == norm_bool(gold_val))
        elif ds in ("hellaswag","mmlu"):
            ok = (str(pv).strip().upper() == str(gold_val).strip().upper())
        elif ds == "gsm8k":
            # 数值近似匹配（避免轻微格式差异）
            if is_num(pv) and is_num(gold_val):
                ok = (abs(float(pv) - float(gold_val)) < 1e-6)
            else:
                ok = (str(pv).strip() == str(gold_val).strip())
        else:
            ok = (str(pv).strip() == str(gold_val).strip())

        total += 1
        correct += int(ok)
        by_ds[ds]["n"] += 1
        by_ds[ds]["c"] += int(ok)

    report = {
        "total": total,
        "accuracy": (correct/total) if total else 0.0,
        "per_dataset": {
            ds: {"n":v["n"], "acc": (v["c"]/v["n"] if v["n"] else 0.0)}
            for ds,v in by_ds.items()
        }
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
