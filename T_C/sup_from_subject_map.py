
# -*- coding: utf-8 -*-
"""
sup_from_subject_map.py
Attach gold super labels to a normalized JSONL using a subject->super mapping YAML/JSON.

Input JSONL is expected to have "subject" per row (use prep_mmlu_pro_jsonl.py to preserve it).
Mapping file example (YAML):
  mappings:
    high_school_physics: stem
    college_physics: stem
    abstract_algebra: math_reasoning
    world_history: humanities
    us_history: humanities
    macroeconomics: social_science

Usage:
  python sup_from_subject_map.py --jsonl data_local/mmlu_pro_val.jsonl \
     --map subject2sup.yaml --out data_local/mmlu_pro_val.labeled.jsonl

Then evaluate with score_sup_accuracy_gold.py:
  python score_sup_accuracy_gold.py --pred preds_mmlu_pro.jsonl --jsonl data_local/mmlu_pro_val.labeled.jsonl
"""
import argparse, json, re
from pathlib import Path
from typing import Dict, Any
def load_mapping(p: Path) -> Dict[str, str]:
    if p.suffix.lower() in [".yaml", ".yml"]:
        import yaml  # requires pyyaml
        obj = yaml.safe_load(p.read_text(encoding="utf-8"))
    else:
        import json
        obj = json.loads(p.read_text(encoding="utf-8"))
    # support either top-level dict or {"mappings": {...}}
    if isinstance(obj, dict) and "mappings" in obj and isinstance(obj["mappings"], dict):
        return {str(k): str(v) for k, v in obj["mappings"].items()}
    if isinstance(obj, dict):
        return {str(k): str(v) for k, v in obj.items()}
    raise ValueError("Mapping must be a dict or {'mappings': {...}}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="normalized JSONL with 'subject'")
    ap.add_argument("--map", required=True, help="subject->super mapping file (yaml/json)")
    ap.add_argument("--out", required=True, help="output JSONL with 'label_sup' attached")
    ap.add_argument("--regex", action="store_true", help="treat mapping keys as regex patterns")
    args = ap.parse_args()

    mp = load_mapping(Path(args.map))

    n_in = 0
    n_hit = 0
    with open(args.jsonl, "r", encoding="utf-8") as fin, open(args.out, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line: continue
            n_in += 1
            r = json.loads(line)
            subject = str(r.get("subject", "")).strip()
            label = None
            if args.regex:
                for pat, sup in mp.items():
                    if re.fullmatch(pat, subject):
                        label = sup
                        break
            else:
                label = mp.get(subject)
            if label:
                r["label_sup"] = label
                n_hit += 1
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[map] total={n_in} labeled={n_hit} coverage={n_hit/max(1,n_in):.3f} -> {args.out}")

if __name__ == "__main__":
    main()
