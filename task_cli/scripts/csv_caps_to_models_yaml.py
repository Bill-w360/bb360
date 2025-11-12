# scripts/csv_caps_to_models_yaml.py
# -*- coding: utf-8 -*-
import csv, yaml, argparse

DIM_ORDER = [
  "math_reasoning","commonsense","reading_comprehension","general_knowledge",
  "humanities","social_science","stem","other_knowledge"
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="models.yaml")
    # 新增 divide100 模式：把 0~100 直接缩放到 0~1
    ap.add_argument("--norm", choices=["none","divide100","minmax"], default="divide100")
    ap.add_argument("--device-map", default="")
    args = ap.parse_args()

    devs = [d.strip() for d in args.device_map.split(",") if d.strip()]
    models = []

    with open(args.csv, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        rows = list(rdr)

    # minmax 需要的统计（如果你用 divide100 或 none，会被忽略）
    mins, maxs = {}, {}
    if args.norm == "minmax":
        for d in DIM_ORDER:
            vals = [float(r[d]) for r in rows if r.get(d, "") != ""]
            mins[d] = min(vals) if vals else 0.0
            maxs[d] = max(vals) if vals else 1.0

    for i, r in enumerate(rows):
        name = r["model"]
        vec = []
        for d in DIM_ORDER:
            v = float(r.get(d, 0.0))
            if args.norm == "divide100":
                v = v / 100.0
            elif args.norm == "minmax":
                lo, hi = mins[d], maxs[d]
                v = 0.0 if hi == lo else (v - lo) / (hi - lo)
            # else: none -> 原样
            vec.append(round(v, 6))

        models.append({
            "name": name,
            "capability": vec,
            "cost": {"tp": 1.0},
            # 可选：在这里补 infer 配置
        })

    y = {"dim_order": DIM_ORDER, "models": models}
    with open(args.out, "w", encoding="utf-8") as f:
        yaml.safe_dump(y, f, allow_unicode=True, sort_keys=False)
    print(f"[ok] wrote {args.out} (norm={args.norm})")

if __name__ == "__main__":
    main()
