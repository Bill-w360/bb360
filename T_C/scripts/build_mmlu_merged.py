# -*- coding: utf-8 -*-
import argparse, json, os, sys
from datasets import load_dataset

SUBJECTS = [
    "abstract_algebra","anatomy","astronomy","business_ethics","clinical_knowledge",
    "college_biology","college_chemistry","college_computer_science","college_mathematics","college_medicine","college_physics",
    "computer_security","conceptual_physics","econometrics","electrical_engineering","elementary_mathematics","formal_logic",
    "global_facts","high_school_biology","high_school_chemistry","high_school_computer_science","high_school_european_history",
    "high_school_geography","high_school_government_and_politics","high_school_macroeconomics","high_school_mathematics",
    "high_school_microeconomics","high_school_physics","high_school_psychology","high_school_statistics","high_school_us_history",
    "high_school_world_history","human_aging","human_sexuality","international_law","jurisprudence","logical_fallacies",
    "machine_learning","management","marketing","medical_genetics","miscellaneous","moral_disputes","moral_scenarios","nutrition",
    "philosophical_ethics","philosophy","prehistory","professional_accounting","professional_law","professional_medicine",
    "professional_psychology","public_relations","security_studies","sociology","us_foreign_policy","virology","world_religions"
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="mmlu_validation_merged.jsonl")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--hf-offline", action="store_true")
    args = ap.parse_args()
    if args.hf_offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    out = open(args.out, "w", encoding="utf-8")
    n = 0
    for s in SUBJECTS:
        try:
            ds = load_dataset("cais/mmlu", s, split="validation", cache_dir=args.cache_dir)
        except Exception as e:
            print(f"[warn] skip {s}: {e}", file=sys.stderr); continue
        for ex in ds:
            rec = {
                "subject": s,
                "question": ex["question"],
                "choices": ex["choices"],
                # 不写答案，避免泄漏；这里只做学科分类。
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
        print(f"[ok] {s}: {len(ds)}")
    out.close()
    print(f"[done] total={n} saved to {args.out}")

if __name__ == "__main__":
    main()
