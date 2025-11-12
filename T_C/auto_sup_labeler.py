
# -*- coding: utf-8 -*-
"""
auto_sup_labeler.py
A dataset-agnostic 8-class (super) labeler that does NOT require a per-dataset mapping.
It assigns label_sup using lightweight heuristics from (subject, question, choices) when available.

Priority:
1) subject-based regex matching (broad, dataset-agnostic)
2) question/choices keyword+signal heuristics
3) fallback: "other_knowledge"

Usage:
  python auto_sup_labeler.py --in data_local/mmlu_pro_val.jsonl --out data_local/mmlu_pro_val.auto.jsonl

Then evaluate your classifier predictions against this silver-standard:
  python score_sup_accuracy_gold.py \
    --pred preds_mmlu_pro.jsonl \
    --jsonl data_local/mmlu_pro_val.auto.jsonl
"""
import argparse, json, re
from pathlib import Path
from typing import Dict, Any, List

SUPS = [
    "math_reasoning",
    "commonsense",
    "reading_comprehension",
    "general_knowledge",
    "humanities",
    "social_science",
    "stem",
    "other_knowledge",
]

# 1) SUBJECT-LEVEL broad regexes (dataset-agnostic)
SUBJECT_PATTERNS = [
    ("math_reasoning", re.compile(r"(math|algebra|calculus|geometry|number[_\-\s]*theory|probabilit|combinatoric|logic|statistics)", re.I)),
    ("stem", re.compile(r"(physics|chemistr|biology|biochem|geolog|astronom|engineering|computer|electrical|mechanical|materials|neuroscience|botany|zoolog)", re.I)),
    ("humanities", re.compile(r"(history|philosophy|linguistic|literature|classics|ethic|aestheti|theolog|art[_\-\s]*history|world[_\-\s]*history|us[_\-\s]*history)", re.I)),
    ("social_science", re.compile(r"(econom|macroeconom|microeconom|business|finance|accounting|sociolog|psycholog|politic|government|law|anthropolog|education)", re.I)),
    ("reading_comprehension", re.compile(r"(reading[_\-\s]*comprehension|rc)", re.I)),
    ("commonsense", re.compile(r"(commonsense|hellaswag|social[_\-\s]*iqa|piqa)", re.I)),
    ("general_knowledge", re.compile(r"(trivia|global[_\-\s]*facts|general[_\-\s]*knowledge)", re.I)),
]

# 2) TEXT HEURISTICS from question/choices
KW = {
    "math_reasoning": [
        r"\b(integer|prime|composite|remainder|probabilit|percent|equation|inequality|sum|product|ratio|mean|median|mode|variance|matrix|vector|derivative|integral|limit)\b",
        r"\b(\d+\s*(?:\+|\-|\*|/|%|=))",
    ],
    "reading_comprehension": [
        r"\b(passage|according to the passage|the author|paragraph|main idea|best title|context)\b",
    ],
    "commonsense": [
        r"\b(plausible|most likely|makes sense|everyday|commonsense)\b",
        r"\b(Which of the following.*?most|best).*?(describe|happen|next)\b",
    ],
    "general_knowledge": [
        r"\b(capital of|located in|invented by|founded in|who wrote|when was)\b",
    ],
    "humanities": [
        r"\b(philosoph|ethic|aestheti|metaphysic|epistemolog|poet|novel|rhetori|archaeolog|linguist)\b",
    ],
    "social_science": [
        r"\b(GDP|inflation|unemploy|market|demand|supply|utility|politic|treaty|constitution|psycholog|cognitive|experiment)\b",
    ],
    "stem": [
        r"\b(velocity|acceleration|force|mass|electron|molecule|enzyme|DNA|RNA|circuit|voltage|current|algorithm|complexity|runtime|compiler)\b",
    ],
}

def pick_by_subject(subject: str) -> str:
    if not subject:
        return ""
    for sup, pat in SUBJECT_PATTERNS:
        if pat.search(subject):
            return sup
    return ""

def pick_by_text(q: str, choices: List[str]) -> str:
    text = q + " " + " ".join(choices or [])
    for sup, patterns in KW.items():
        for p in patterns:
            if re.search(p, text, flags=re.I):
                return sup
    return ""

def label_row(r: Dict[str, Any]) -> str:
    sub = str(r.get("subject", "") or "").strip()
    q = str(r.get("question", "") or "")
    choices = r.get("choices", [])
    # A) subject first
    lab = pick_by_subject(sub)
    if lab:
        return lab
    # B) then text heuristics
    lab = pick_by_text(q, choices if isinstance(choices, list) else [])
    if lab:
        return lab
    # C) weak fallback
    return "other_knowledge"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="input JSONL with question[/choices][/subject]")
    ap.add_argument("--out", required=True, help="output JSONL with label_sup attached")
    args = ap.parse_args()

    inp = Path(args.inp)
    outp = Path(args.out)

    n = 0
    hit_sub = 0
    hit_txt = 0
    with inp.open("r", encoding="utf-8") as fin, outp.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            lab = pick_by_subject(str(r.get("subject", "")))
            if lab:
                hit_sub += 1
            else:
                lab = pick_by_text(str(r.get("question","")), r.get("choices", []))
                if lab:
                    hit_txt += 1
                else:
                    lab = "other_knowledge"
            r["label_sup"] = lab
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    print(f"[auto] total={n} via_subject={hit_sub} via_text={hit_txt} -> {outp}")

if __name__ == "__main__":
    main()
