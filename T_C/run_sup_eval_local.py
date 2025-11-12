
# -*- coding: utf-8 -*-
"""
run_sup_eval_local.py

Batch-evaluate local JSONL datasets (e.g., boolq/{train,validation}.jsonl, gsm8k/{train,test}.jsonl, hellaswag/{train,test}.jsonl)
using your infer_task_classifier.py, then score 8-class (super) accuracy via score_sup_accuracy.py.
It saves a summary CSV and a bar chart.
"""
import argparse, json, subprocess, sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import csv
import matplotlib.pyplot as plt

# default mapping from dataset to target super class
DEF_SUP_MAP = {
    "boolq": "reading_comprehension",
    "gsm8k": "math_reasoning",
    "hellaswag": "commonsense",
}

# candidate split filenames to pick from (priority order)
CAND_SPLITS = ["validation.jsonl", "val.jsonl", "dev.jsonl", "test.jsonl", "train.jsonl"]

def parse_sup_map(s: Optional[str]) -> Dict[str, str]:
    m = dict(DEF_SUP_MAP)
    if not s:
        return m
    for pair in s.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise ValueError(f"--sup-map bad format: {pair}")
        k, v = pair.split("=", 1)
        m[k.strip()] = v.strip()
    return m

def pick_split_file(ds_dir: Path) -> Optional[Path]:
    for name in CAND_SPLITS:
        p = ds_dir / name
        if p.exists():
            return p
    return None

def normalize_record(r: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a record into {question: str, choices?: [str, ...]}"""
    # question candidates
    q = (
        r.get("question") or r.get("prompt") or r.get("input") or
        r.get("query") or r.get("text") or r.get("title") or ""
    )
    if not isinstance(q, str):
        q = str(q)

    # choices candidates
    choices = None
    for key in ["choices", "options", "endings"]:
        if key in r and isinstance(r[key], (list, tuple)):
            choices = []
            for c in r[key]:
                if isinstance(c, str):
                    choices.append(c)
                else:
                    choices.append(str(c))
            break

    out = {"question": " ".join(q.strip().split())}
    if choices:
        out["choices"] = choices
    return out

def normalize_jsonl(src: Path, dst: Path) -> int:
    """Read src JSONL, write normalized dst JSONL, return sample count."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with src.open("r", encoding="utf-8") as f_in, dst.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            rr = normalize_record(r)
            f_out.write(json.dumps(rr, ensure_ascii=False) + "\n")
            n += 1
    return n

def run_infer(model_dir: Path, test_json: Path, out_pred: Path, offline: bool = True) -> None:
    cmd = [
        sys.executable, "infer_task_classifier.py",
        "--model-dir", str(model_dir),
        "--test-json", str(test_json),
        "--out", str(out_pred),
    ]
    if offline:
        cmd.append("--offline")
    print("[cmd]", " ".join(cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(f"infer failed for {test_json}.")

def run_score(pred_path: Path, target_sup: str, tau: Optional[float]) -> Tuple[float, int, int]:
    cmd = [sys.executable, "score_sup_accuracy.py", "--pred", str(pred_path), "--target", target_sup]
    if tau is not None:
        cmd += ["--reject", str(tau)]
    print("[cmd]", " ".join(cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(proc.stdout)

    # parse metrics from stdout
    acc = 0.0
    n_total = None
    n_used = None
    for ln in proc.stdout.splitlines():
        ln = ln.strip()
        if ln.startswith("8 类准确率:") or ln.lower().startswith("8-class accuracy:"):
            try:
                acc = float(ln.split(":")[1].strip())
            except Exception:
                pass
        elif ln.startswith("总样本:") or ln.lower().startswith("total:"):
            try:
                n_total = int(ln.split(":")[1].strip())
            except Exception:
                pass
        elif ln.startswith("拒识数:"):
            try:
                parts = ln.split()
                used = int(parts[3])
                n_used = used
            except Exception:
                pass
    return acc, (n_total or 0), (n_used if n_used is not None else (n_total or 0))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="root folder containing dataset subfolders")
    ap.add_argument("--model-dir", required=True, help="trained classifier dir (tokenizer/model)")
    ap.add_argument("--out-dir", required=True, help="where to save normalized JSONL, predictions, and summary")
    ap.add_argument("--datasets", default="boolq,gsm8k,hellaswag", help="comma-separated dataset names")
    ap.add_argument("--tau", type=float, default=None, help="reject threshold (optional)")
    ap.add_argument("--sup-map", default=None, help='override default super mapping, e.g., "boolq=reading_comprehension,gsm8k=math_reasoning"')
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    sup_map = parse_sup_map(args.sup_map)
    ds_names = [x.strip() for x in args.datasets.split(",") if x.strip()]

    rows_summary: List[Dict[str, Any]] = []
    for ds in ds_names:
        ds_dir = data_root / ds
        if not ds_dir.exists():
            print(f"[warn] dataset dir not found: {ds_dir}, skip.")
            continue

        src = pick_split_file(ds_dir)
        if not src:
            print(f"[warn] {ds}: no split file found among {CAND_SPLITS}, skip.")
            continue

        norm = out_root / f"{ds}.norm.jsonl"
        n = normalize_jsonl(src, norm)
        if n == 0:
            print(f"[warn] {ds}: 0 samples after normalization, skip.")
            continue

        pred = out_root / f"preds_{ds}.jsonl"
        run_infer(Path(args.model_dir), norm, pred, offline=True)

        target_sup = sup_map.get(ds, "commonsense")

        acc, n_total, n_used = run_score(pred, target_sup, args.tau)
        rows_summary.append({
            "dataset": ds,
            "split_file": src.name,
            "samples": n_total,
            "used_after_reject": n_used,
            "target_super": target_sup,
            "acc_super": acc,
            "pred_file": str(pred),
        })

    if rows_summary:
        csv_path = out_root / "sup_eval_summary.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_summary[0].keys()))
            w.writeheader()
            for r in rows_summary:
                w.writerow(r)
        print(f"[save] summary CSV => {csv_path}")

        fig_path = out_root / "sup_eval_bar.png"
        plt.figure(figsize=(7.5, 4.5))
        xs = [r["dataset"] for r in rows_summary]
        ys = [r["acc_super"] * 100.0 for r in rows_summary]
        plt.bar(xs, ys)
        plt.ylim(0, 100)
        plt.ylabel("8-class accuracy (%)")
        plt.title("Super-class accuracy across datasets")
        for x, y in zip(xs, ys):
            plt.text(x, y + 1, f"{y:.1f}%", ha="center", va="bottom")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150)
        print(f"[save] bar chart => {fig_path}")
    else:
        print("[warn] nothing to summarize.")

if __name__ == "__main__":
    main()
