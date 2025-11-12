# extras/ds_to_tasks.py
# -*- coding: utf-8 -*-
"""
把 data_local/{boolq,gsm8k,hellaswag,mmlu}/*.jsonl 转成统一任务 JSONL：
输出每行包含：
{id, dataset, category, difficulty, arrival_time, prompt, meta}

约定到路由大类(category)：
- boolq      -> reading_comprehension
- gsm8k      -> math_reasoning
- hellaswag  -> commonsense
- mmlu       -> 由 subject 映射到 {stem/humanities/social_science/other_knowledge/general_knowledge}

difficulty：简单启发式（可自己改）
arrival_time：按行号自增 1.0
"""

import os, json, argparse, glob, re

SUBJ2DOMAIN = {
    # STEM
    "abstract_algebra":"stem","college_chemistry":"stem","college_mathematics":"stem",
    "college_physics":"stem","computer_security":"stem","electrical_engineering":"stem",
    "high_school_biology":"stem","high_school_chemistry":"stem","high_school_computer_science":"stem",
    "high_school_mathematics":"stem","high_school_physics":"stem","high_school_statistics":"stem",
    "machine_learning":"stem","elementary_mathematics":"stem","conceptual_physics":"stem",
    # Humanities
    "philosophy":"humanities","world_religions":"humanities","formal_logic":"humanities",
    "humanities":"humanities","jurisprudence":"humanities","professional_law":"humanities",
    # Social Science
    "econometrics":"social_science","high_school_geography":"social_science",
    "high_school_government_and_politics":"social_science","high_school_macroeconomics":"social_science",
    "high_school_microeconomics":"social_science","high_school_psychology":"social_science",
    "high_school_us_history":"social_science","high_school_world_history":"social_science",
    "sociology":"social_science",
    # Other / General
    "business_ethics":"other_knowledge","marketing":"other_knowledge","management":"other_knowledge",
    "logical_fallacies":"other_knowledge","nutrition":"other_knowledge","international_law":"other_knowledge",
    "professional_accounting":"other_knowledge","virology":"other_knowledge",
}

def read_jsonl(path):
    with open(path,"r",encoding="utf-8") as f:
        for ln, line in enumerate(f,1):
            s=line.strip()
            if s: 
                try:
                    yield json.loads(s)
                except Exception as e:
                    print(f"[warn] bad json {path}:{ln}: {e}")

def norm_diff_by_len(txt, easy_th=80, hard_th=220):
    n = len(txt)
    if n < easy_th: return "easy"
    if n > hard_th: return "hard"
    return "medium"

def conv_boolq(path, out, at_base):
    did = 0
    for ex in read_jsonl(path):
        did += 1
        ds = "boolq"
        cat = "reading_comprehension"
        passage = ex.get("passage","")
        question = ex.get("question","")
        ans = ex.get("answer", "")
        prompt = f"Passage:\n{passage}\n\nQuestion: {question}\nAnswer yes/no."
        out.write(json.dumps({
            "id": f"{ds}-{did}",
            "dataset": ds,
            "category": cat,
            "difficulty": norm_diff_by_len(passage+question),
            "arrival_time": at_base + did,
            "prompt": prompt,
            "meta": {"answer": str(ans).lower() in ["true","1","yes","y","t"]}
        }, ensure_ascii=False)+"\n")
    return did

def conv_gsm8k(path, out, at_base):
    did = 0
    for ex in read_jsonl(path):
        did += 1
        ds = "gsm8k"
        cat = "math_reasoning"
        q = ex.get("question","")
        a = ex.get("answer","")
        prompt = q
        diff = "hard" if len(a) > 200 else ("medium" if len(a) > 80 else "easy")
        out.write(json.dumps({
            "id": f"{ds}-{did}",
            "dataset": ds,
            "category": cat,
            "difficulty": diff,
            "arrival_time": at_base + did,
            "prompt": prompt,
            "meta": {"answer": a}
        }, ensure_ascii=False)+"\n")
    return did

def conv_hellaswag(path, out, at_base):
    did = 0
    for ex in read_jsonl(path):
        did += 1
        ds = "hellaswag"
        cat = "commonsense"
        ctx = ex.get("ctx","") or ex.get("context","")
        ends = ex.get("endings") or ex.get("endings_random") or []
        label = ex.get("label")
        # 构造 MC 提示
        choice_lines = [f"{chr(65+i)}. {t}" for i,t in enumerate(ends)]
        prompt = f"{ctx}\n\nWhich option best continues?\n" + "\n".join(choice_lines)
        gold = None
        try:
            gold = int(label)
        except:
            # 有些版本是字符串索引
            if isinstance(label,str) and label.isdigit(): gold = int(label)
        out.write(json.dumps({
            "id": f"{ds}-{did}",
            "dataset": ds,
            "category": cat,
            "difficulty": norm_diff_by_len(ctx),
            "arrival_time": at_base + did,
            "prompt": prompt,
            "meta": {"choices": ends, "answer_idx": gold}
        }, ensure_ascii=False)+"\n")
    return did

def conv_mmlu(path, out, at_base):
    did = 0
    for ex in read_jsonl(path):
        did += 1
        ds = "mmlu"
        subj = (ex.get("subject") or ex.get("category") or "").strip()
        dom = SUBJ2DOMAIN.get(subj, "general_knowledge")
        q = ex.get("question","")
        # 可能是 list，也可能是 dict/逗号分隔，做点鲁棒
        choices = ex.get("choices")
        if isinstance(choices, str):
            # "A|||B|||C|||D" 或 "A|B|C|D"
            parts = re.split(r"\|\|\||\|", choices)
        elif isinstance(choices, list):
            parts = choices
        else:
            parts = [ex.get("A"), ex.get("B"), ex.get("C"), ex.get("D")]
            parts = [p for p in parts if p not in (None, "")]
        ans = ex.get("answer")
        # 答案可能是字母(A/B/C/D)或索引
        gold_idx = None
        if isinstance(ans, int):
            gold_idx = ans
        elif isinstance(ans, str):
            if ans.isdigit(): gold_idx = int(ans)
            else:
                up = ans.strip().upper()
                if up in ["A","B","C","D"]:
                    gold_idx = ["A","B","C","D"].index(up)
        prompt = q + "\n" + "\n".join([f"{chr(65+i)}. {t}" for i,t in enumerate(parts)])
        out.write(json.dumps({
            "id": f"{ds}-{did}",
            "dataset": ds,
            "category": dom,
            "difficulty": "medium",
            "arrival_time": at_base + did,
            "prompt": prompt,
            "meta": {"choices": parts, "answer_idx": gold_idx, "subject": subj}
        }, ensure_ascii=False)+"\n")
    return did

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data_local", help="包含四个数据集子目录的根路径")
    ap.add_argument("--out", required=True, help="输出统一任务 JSONL")
    args = ap.parse_args()

    writers = {
        "boolq": conv_boolq,
        "gsm8k": conv_gsm8k,
        "hellaswag": conv_hellaswag,
        "mmlu": conv_mmlu,
    }

    total = 0
    with open(args.out, "w", encoding="utf-8") as g:
        at_base = 0.0
        for name, fn in writers.items():
            # 兼容 train/validation/test.jsonl
            for split in ["train","validation","test"]:
                p = os.path.join(args.root, name, f"{split}.jsonl")
                if os.path.isfile(p):
                    print(f"[conv] {name}/{split}")
                    n = fn(p, g, at_base)
                    total += n
                    at_base += n
    print(f"[ok] wrote {args.out}, tasks={total}")

if __name__ == "__main__":
    main()
