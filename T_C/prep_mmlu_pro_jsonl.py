
# -*- coding: utf-8 -*-
"""
prep_mmlu_pro_jsonl.py  (revised for MMLU-Pro raw rows)

Goal:
- Normalize various MMLU/MMLU-Pro style schemas into a clean JSONL:
    {
      "id": <int or str>,
      "question": "...",
      "choices": ["...", "...", ...],
      "subject": "business_ethics",         # parsed from src or category
      "category": "business",               # original category if present
      "source": "ori_mmlu-business_ethics", # original 'src' if present
      # optional golds if --keep-gold:
      "gold_choice": "I",
      "gold_index": 8,
      "gold_text": "Unsafe practices, Distress, Fear, Serious"
    }
- Works with local json/jsonl/csv, or HF datasets via --hf-id.
- Specifically supports rows like:
  {
    "question_id": 70,
    "question": "...",
    "options": [...],
    "answer": "I",
    "answer_index": 8,
    "category": "business",
    "src": "ori_mmlu-business_ethics"
  }

Usage examples:
  python prep_mmlu_pro_jsonl.py --infile mmlu_pro_validation.jsonl --out data_local/mmlu_pro_val.jsonl --keep-gold
  python prep_mmlu_pro_jsonl.py --hf-id TIGER-Lab/MMLU-Pro --split validation --out data_local/mmlu_pro_val.jsonl
"""
import argparse, json, csv, re
from pathlib import Path
from typing import Dict, Any, List, Optional

LETTER2IDX = {chr(ord('A')+i): i for i in range(26)}

def _read_local(fp: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    suf = fp.suffix.lower()
    if suf == ".jsonl":
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    elif suf == ".json":
        with fp.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            rows = obj
        elif isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], list):
            rows = obj["data"]
        else:
            raise ValueError("JSON must be a list or an object with 'data' list.")
    elif suf == ".csv":
        with fp.open("r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                rows.append(row)
    else:
        raise ValueError(f"Unsupported file: {fp}")
    return rows

def _to_list(v):
    if v is None:
        return None
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        return [v]
    if isinstance(v, dict):
        # keep values in sorted key order for determinism
        return [str(v[k]) for k in sorted(v.keys())]
    return [str(v)]

def _norm_text(s: Any) -> str:
    s = "" if s is None else str(s)
    return " ".join(s.strip().split())

def _parse_subject(src: Optional[str], category: Optional[str]) -> Optional[str]:
    """
    Try to extract a subject slug from src like 'ori_mmlu-business_ethics' -> 'business_ethics'.
    Fallback to category (lowercase, replace spaces with underscore).
    """
    if src and isinstance(src, str):
        # take the last hyphen-separated token: e.g., ori_mmlu-business_ethics -> business_ethics
        m = re.search(r"-([A-Za-z0-9_]+)$", src)
        if m:
            return m.group(1).lower()
        # also handle patterns like mmlu/business_ethics
        m = re.search(r"/([A-Za-z0-9_]+)$", src)
        if m:
            return m.group(1).lower()
    if category and isinstance(category, str) and category.strip():
        return re.sub(r"\s+", "_", category.strip().lower())
    return None

def normalize_record(r: Dict[str, Any], keep_gold: bool) -> Dict[str, Any]:
    # id
    rid = r.get("question_id") or r.get("id") or r.get("uid") or r.get("qid")
    rid = rid if rid is not None else ""

    # question
    q = r.get("question") or r.get("prompt") or r.get("input") or r.get("query") or r.get("text") or ""
    q = _norm_text(q)

    # choices (supports options/choices/answers/endings)
    raw_choices = None
    for k in ["options", "choices", "answers", "endings"]:
        if k in r:
            raw_choices = _to_list(r[k])
            break
    choices = [ _norm_text(c) for c in (raw_choices or []) ]

    # subject/category/src
    category = r.get("category") or r.get("field")
    src = r.get("src") or r.get("source")
    subject = r.get("subject")
    if not subject:
        subject = _parse_subject(src, category)

    # base output
    out = {
        "id": rid,
        "question": q,
    }
    if choices:
        out["choices"] = choices
    if subject:
        out["subject"] = str(subject)
    if category:
        out["category"] = str(category)
    if src:
        out["source"] = str(src)

    # optional golds
    if keep_gold:
        gold_choice = r.get("answer") or r.get("gold") or r.get("label")
        gold_index = r.get("answer_index")
        # reconcile letter/index
        if gold_index is None and isinstance(gold_choice, str) and gold_choice.strip().upper() in LETTER2IDX:
            gold_index = LETTER2IDX[gold_choice.strip().upper()]
        if gold_index is not None:
            try:
                gi = int(gold_index)
            except Exception:
                gi = None
        else:
            gi = None

        out["gold_choice"] = gold_choice if gold_choice is not None else ""
        out["gold_index"] = gi if gi is not None else ""
        if gi is not None and 0 <= gi < len(choices):
            out["gold_text"] = choices[gi]
        elif isinstance(gold_choice, str):
            # try letter mapping to text
            li = LETTER2IDX.get(gold_choice.strip().upper(), None)
            out["gold_text"] = choices[li] if (li is not None and 0 <= li < len(choices)) else ""
        else:
            out["gold_text"] = ""

    return out

def write_jsonl(rows: List[Dict[str, Any]], out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-id", default=None, help="HF datasets id, e.g., TIGER-Lab/MMLU-Pro")
    ap.add_argument("--split", default="validation", help="split for HF datasets")
    ap.add_argument("--infile", default=None, help="local json/jsonl/csv if not using HF")
    ap.add_argument("--out", required=True, help="output JSONL path")
    ap.add_argument("--keep-gold", action="store_true", help="include gold fields if available (answer/answer_index)")
    args = ap.parse_args()

    rows_raw: List[Dict[str, Any]]
    if args.hf_id:
        from datasets import load_dataset  # lazy import
        # some repos require config; allow users to pass "repo/config" in hf-id string.
        try:
            if "/" in args.hf_id:
                repo, config = args.hf_id.split("/", 1)
                ds = load_dataset(repo, config, split=args.split)
            else:
                ds = load_dataset(args.hf_id, split=args.split)
        except Exception:
            ds = load_dataset(args.hf_id, split=args.split)
        rows_raw = [dict(x) for x in ds]
    else:
        if not args.infile:
            raise ValueError("Either --hf-id or --infile must be provided.")
        rows_raw = _read_local(Path(args.infile))

    rows = [normalize_record(r, args.keep_gold) for r in rows_raw]
    n = write_jsonl(rows, Path(args.out))
    print(f"[save] {n} rows -> {args.out}")

if __name__ == "__main__":
    main()
