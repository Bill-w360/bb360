# tools/lm_eval_to_scores.py
# 用法: python tools/lm_eval_to_scores.py <results.json> <model_id> <out_csv>
from __future__ import annotations
import json
import sys
import csv
from pathlib import Path
from numbers import Number

def _as_number(x):
    # 支持 {"value":0.58, "stderr":...} 或 0.58
    if isinstance(x, Number):
        return float(x), None
    if isinstance(x, dict):
        v = x.get("value", None)
        s = x.get("stderr", x.get("acc_stderr", None))
        if isinstance(v, Number):
            return float(v) * 100.0, (float(s) if isinstance(s, Number) else None)
    return None, None

def _normalize_metric_name(name: str) -> str:
    # 把 'flexible-extract/exact_match' -> 'flexible-extract:exact_match'
    return name.replace("/", ":")

def flatten_results(results: dict, model_id: str) -> list[tuple[str,str,str,float]]:
    """
    返回 [(model, dataset, metric, value), ...]
    仅保留数值型 value；忽略 stderr 行。
    """
    rows = []
    res = results.get("results", {})
    if not isinstance(res, dict):
        return rows

    for dataset, metrics in res.items():
        if not isinstance(metrics, dict):
            continue
        for metric_name, metric_obj in metrics.items():
            # 统一命名
            mname = _normalize_metric_name(metric_name)

            # 过滤掉专门的误差键
            if mname.endswith("_stderr") or mname.endswith(":stderr"):
                continue

            val, _stderr = _as_number(metric_obj)
            if val is None:
                # 有些版本把 'acc' 和 'acc_norm' 各自下再嵌一层 {'value':...}
                # 或者把 filter 放在 dict 的子键里，这里尝试再下一层
                if isinstance(metric_obj, dict):
                    for subk, subv in metric_obj.items():
                        subname = _normalize_metric_name(f"{mname}:{subk}")
                        v2, _ = _as_number(subv)
                        if v2 is not None:
                            rows.append((model_id, dataset, subname, v2))
                continue
            rows.append((model_id, dataset, mname, val))
    return rows

def main():
    if len(sys.argv) != 4:
        print("Usage: python tools/lm_eval_to_scores.py <results.json> <model_id> <out_csv>")
        sys.exit(2)

    results_json = Path(sys.argv[1])
    model_id = sys.argv[2]
    out_csv = Path(sys.argv[3])

    data = json.loads(results_json.read_text(encoding="utf-8"))
    rows = flatten_results(data, model_id)

    # 如果空，至少写表头，让上游能检测到“没有可用分数”
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "dataset", "metric", "value"])
        for r in rows:
            w.writerow(r)

    if rows:
        print(f"[ok] wrote {len(rows)} metrics to {out_csv}")
    else:
        print("[warn] No metrics found to write (empty rows)")

if __name__ == "__main__":
    main()
