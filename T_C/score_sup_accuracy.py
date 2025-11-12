
# -*- coding: utf-8 -*-

# score_sup_accuracy.py
# 读取 infer_task_classifier.py 产出的预测 JSONL（每条含 _pred_super 和 _prob_super），
# 统计与目标上位域标签（如 math_reasoning, reading_comprehension）的匹配准确率。
# 支持设定拒识阈值（max(_prob_super.values()) < tau 即视为 'reject'）。

# 示例：
#   python score_sup_accuracy.py --pred preds_boolq.jsonl --target reading_comprehension --reject 0.5
#   python score_sup_accuracy.py --pred preds_gsm8k.jsonl --target math_reasoning --reject 0.5

import argparse, json
from pathlib import Path
from collections import Counter

def load_jsonl(p):
    rows=[]
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line: rows.append(json.loads(line))
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="预测 JSONL（含 _pred_super 与 _prob_super）")
    ap.add_argument("--target", required=True, help="目标上位域标签名，例如 math_reasoning / reading_comprehension")
    ap.add_argument("--reject", type=float, default=None, help="拒识阈值（若最大上位域概率 < 阈值，则记为 reject）")
    args = ap.parse_args()

    rows = load_jsonl(args.pred)
    n = len(rows)

    hit = 0
    rej = 0
    conf = Counter()  # 统计各个预测标签计数
    for r in rows:
        pred = r.get("_pred_super")
        prob_map = r.get("_prob_super", {}) or {}
        maxp = max(prob_map.values()) if prob_map else 0.0

        if args.reject is not None and maxp < args.reject:
            rej += 1
            continue

        conf[pred] += 1
        if pred == args.target:
            hit += 1

    used = n - rej
    acc = (hit / used) if used > 0 else 0.0

    print(f"总样本: {n}")
    if args.reject is not None:
        print(f"拒识数: {rej}  使用样本: {used}")
    print(f"目标上位域: {args.target}")
    print(f"8 类准确率: {acc:.4f}")
    print("--- 预测标签分布（不含拒识） ---")
    for k, v in conf.most_common():
        print(f"{k:>24s} : {v}")

if __name__ == "__main__":
    main()
