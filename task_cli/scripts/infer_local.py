# scripts/infer_local.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, os, json, re
from typing import Dict, List, Any, Tuple, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ========== I/O ==========
def read_jsonl(p: str):
    with open(p, "r", encoding="utf-8") as f:
        for s in f:
            s = s.strip()
            if s:
                yield json.loads(s)

def write_jsonl(p: str, rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ========== 小工具 ==========
def is_chat_capable(tok: AutoTokenizer) -> bool:
    tmpl = getattr(tok, "chat_template", None)
    return tmpl is not None and len(str(tmpl)) > 0

def chunked(it, n):
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) == n:
            yield buf; buf = []
    if buf: yield buf

def looks_like_code(text: str) -> bool:
    if not text: return False
    t = text.strip().lower()
    return any(k in t for k in ["```", "```python", "def ", "class ", "import ", "print(", "#include", "int main("])

# 兼容不同 transformers 版本的 BadWords 处理器
def _import_badwords():
    try:
        from transformers import LogitsProcessorList, BadWordsLogitsProcessor
        return LogitsProcessorList, BadWordsLogitsProcessor
    except Exception:
        pass
    try:
        from transformers.generation.logits_process import LogitsProcessorList, NoBadWordsLogitsProcessor
        return LogitsProcessorList, NoBadWordsLogitsProcessor
    except Exception:
        return None, None

def make_anti_code_processors(tok: AutoTokenizer, enable: bool):
    if not enable: return None
    LogitsProcessorList, BadWordsLogitsProcessor = _import_badwords()
    if LogitsProcessorList is None or BadWordsLogitsProcessor is None:
        return None
    bad = [
        "```", "```python", "Python", "python", "Code", "code", "program", "algorithm",
        "def ", "class ", "import ", "print(", "#include", "public static void", "int main(",
        "Here is a Python program", "Python Code:", "The answer is"
    ]
    ids = []
    for w in bad:
        t = tok.encode(w, add_special_tokens=False)
        if t: ids.append(t)
    return LogitsProcessorList([BadWordsLogitsProcessor(ids)]) if ids else None

# ========== 任务类型判别（数据集无关） ==========
def guess_task_type(ex: Dict[str, Any], override: str = "") -> str:
    if override: return override
    t = (ex.get("task_type") or "").lower()
    if t: return t
    # 样本字段驱动
    options = ex.get("options")
    if isinstance(options, list) and 2 <= len(options) <= 6:  # MCQ
        return "mcq"
    # 布尔问答：标注/提示，或答案是 yes/no
    ans = (ex.get("answer") or ex.get("gold") or "").strip().lower()
    if ex.get("answer_type") == "bool" or ans in {"yes","no","true","false"}:
        return "qa_bool"
    # 数字题：标注/提示
    if ex.get("answer_type") == "number":
        return "math_numeric"
    # 数据集名提示（尽量广义）
    ds = (ex.get("dataset") or "").lower()
    if "boolq" in ds: return "qa_bool"
    if ds in {"hellaswag","mmlu"}: return "mcq"
    if "gsm8k" in ds: return "math_numeric"
    # 代码
    if "humaneval" in ds or "mbpp" in ds or ex.get("expected_format") == "code":
        return "code_gen"
    return "free_text"

# ========== 判别式打分器 ==========
def score_yes_no(model, tok, input_ids, attn_mask, device) -> Tuple[List[str], List[str]]:
    with torch.inference_mode():
        out = model(input_ids=input_ids.to(device), attention_mask=attn_mask.to(device))
        logits = out.logits
    last_idx = attn_mask.sum(dim=1) - 1
    last_logits = logits[torch.arange(logits.size(0), device=device), last_idx]
    cand_yes, cand_no = [], []
    for s in [" yes","Yes","yes"]: 
        ids = tok.encode(s, add_special_tokens=False); 
        if ids: cand_yes.append(ids[0])
    for s in [" no","No","no"]:
        ids = tok.encode(s, add_special_tokens=False); 
        if ids: cand_no.append(ids[0])
    if not cand_yes: cand_yes=[0]
    if not cand_no:  cand_no=[0]
    yes = last_logits[:, cand_yes].max(dim=1).values
    no  = last_logits[:, cand_no].max(dim=1).values
    preds = ["yes" if y>=n else "no" for y,n in zip(yes.tolist(), no.tolist())]
    raw = [f"yes={float(y):.4f} no={float(n):.4f}" for y,n in zip(yes, no)]
    return preds, raw

def score_mcq_abcd(model, tok, input_ids, attn_mask, device, K=4) -> Tuple[List[str], List[str]]:
    with torch.inference_mode():
        out = model(input_ids=input_ids.to(device), attention_mask=attn_mask.to(device))
        logits = out.logits
    last_idx = attn_mask.sum(dim=1) - 1
    last_logits = logits[torch.arange(logits.size(0), device=device), last_idx]
    letters = [chr(65+i) for i in range(K)]  # A,B,C,D,...
    cand = {L: [" "+L, L, f"({L}", f"{L}.", f"{L})"] for L in letters}
    scores = {}
    for L, vs in cand.items():
        ids0 = []
        for v in vs:
            ids = tok.encode(v, add_special_tokens=False)
            if ids: ids0.append(ids[0])
        if not ids0: ids0 = [0]
        scores[L] = last_logits[:, ids0].max(dim=1).values
    stacked = torch.stack([scores[L] for L in letters], dim=1)
    idx = stacked.argmax(dim=1).tolist()
    preds = [letters[i] for i in idx]
    raw = []
    for i in range(stacked.size(0)):
        raw.append(" ".join([f"{letters[j]}={float(stacked[i,j]):.4f}" for j in range(len(letters))]))
    return preds, raw

# ========== 解析器 ==========
def parse_bool(text: str) -> Tuple[str,bool,str]:
    t = (text or "").strip().lower()
    if re.search(r"\byes\b", t): return "yes", True, ""
    if re.search(r"\bno\b",  t): return "no",  True, ""
    if t.startswith("y"): return "yes", True, "fallback"
    if t.startswith("n"): return "no",  True, "fallback"
    return text, False, "expect yes/no"

def parse_mcq(text: str, K: int=4) -> Tuple[str,bool,str]:
    s = (text or "").strip()
    m = re.search(r"\b([A-{}])\b".format(chr(64+K)), s)
    if m: return m.group(1), True, ""
    if s[:1].upper() in {chr(65+i) for i in range(K)}:
        return s[:1].upper(), True, "fallback"
    return s, False, "expect A-{}".format(chr(64+K))

def parse_numeric_tail(text: str) -> Tuple[str,bool,str]:
    t = (text or "").strip()
    t = re.sub(r"```.*?```", "", t, flags=re.S)  # 去代码块
    tail = t[-200:]
    for line in reversed(tail.splitlines()):
        line = line.strip()
        m = re.match(r"^([-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?)$", line)
        if m: return m.group(1).replace(",",""), True, ""
        m = re.match(r"(?i)^Answer:\s*([-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?)$", line)
        if m: return m.group(1).replace(",",""), True, ""
        m = re.match(r"^#{2,}\s*([-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?)$", line)
        if m: return m.group(1).replace(",",""), True, ""
    return t, False, "expect final number on its own line"

# ========== GSM8K 短重试 ==========
def gsm8k_answer_only_retry(model, tok, q_text: str, device: str, eos_id: Optional[int], pad_id: Optional[int], max_new=12) -> str:
    if is_chat_capable(tok):
        msgs = [
            {"role": "system", "content": "Output ONLY the final numeric answer on ONE line. No words."},
            {"role": "user", "content": q_text + "\n\nAnswer: "}
        ]
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
        inp = enc if isinstance(enc, torch.Tensor) else enc["input_ids"]
    else:
        inp = tok(q_text + "\n\nAnswer: ", return_tensors="pt")["input_ids"]
    inp = inp.to(device); attn = torch.ones_like(inp)

    LogitsProcessorList, BadWordsLogitsProcessor = _import_badwords()
    procs = None
    if LogitsProcessorList and BadWordsLogitsProcessor:
        bad = []
        for ch in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ":
            ids = tok.encode(ch, add_special_tokens=False)
            if ids: bad.append(ids)
        procs = LogitsProcessorList([BadWordsLogitsProcessor(bad)]) if bad else None

    with torch.no_grad():
        gen = model.generate(
            input_ids=inp, attention_mask=attn, max_new_tokens=max_new, do_sample=False,
            eos_token_id=eos_id, pad_token_id=pad_id, logits_processor=procs
        )
    cut = int(attn[0].sum().item())
    return tok.decode(gen[0][cut:], skip_special_tokens=True).strip()

# ========== 主流程 ==========
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="bfloat16", choices=["float16","bfloat16","float32"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--local-files-only", action="store_true")
    ap.add_argument("--task-type", default="")          # 全局覆盖
    ap.add_argument("--anti-code", choices=["auto","on","off"], default="auto")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    torch_dtype = {"float16":torch.float16,"bfloat16":torch.bfloat16,"float32":torch.float32}[args.dtype]

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code,
                                        local_files_only=args.local_files_only, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=args.trust_remote_code,
                                                 local_files_only=args.local_files_only, dtype=torch_dtype).to(args.device)
    model.eval()

    eos_id = getattr(tok, "eos_token_id", None)
    pad_id = getattr(tok, "pad_token_id", None)
    if pad_id is None and eos_id is not None: pad_id = eos_id

    rows = list(read_jsonl(args.inp))
    out_rows: List[Dict[str,Any]] = []

    model_lc = str(args.model).lower()
    force_chat = ("qwen" in model_lc) or ("deepseek" in model_lc)
    chat_mode = is_chat_capable(tok) or force_chat

    def build_enc(ex: Dict[str,Any]):
        q = (ex.get("prompt") or ex.get("question") or ex.get("input") or "").strip()
        options = ex.get("options") if isinstance(ex.get("options"), list) else None
        if chat_mode:
            tt = guess_task_type(ex, args.task_type)
            if tt == "qa_bool":
                msgs = [{"role":"system","content":"Answer ONLY 'yes' or 'no'."},
                        {"role":"user","content": q+"\n\nAnswer:"}]
            elif tt == "mcq":
                opts = "\n".join([f"{chr(65+i)}. {o}" for i,o in enumerate(options or [])])
                msgs = [{"role":"system","content":"Reply with ONLY A, B, C, or D."},
                        {"role":"user","content": f"{q}\n\n{opts}\n\nAnswer:"}]
            elif tt == "math_numeric":
                msgs = [{"role":"system","content":"Output final answer on one line as a number (or 'Answer: <number>'). No words."},
                        {"role":"user","content": q}]
            else:
                msgs = [{"role":"system","content":"You are a helpful assistant."},
                        {"role":"user","content": q}]
            enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
            return enc if isinstance(enc, torch.Tensor) else enc["input_ids"]
        else:
            # 纯文本后备
            return tok(q + "\n\nAnswer:", return_tensors="pt")["input_ids"]

    def to_ids1d(x):
        return x.squeeze(0) if x.dim()==2 else x

    # 分批
    for batch in chunked(rows, args.batch_size):
        encs = [build_enc(ex) for ex in batch]
        ids_list = [to_ids1d(e) for e in encs]
        maxlen = max(int(x.size(0)) for x in ids_list)
        pad_val = pad_id if pad_id is not None else (tok.eos_token_id or 0)
        input_ids = torch.full((len(ids_list), maxlen), pad_val, dtype=torch.long)
        attn = torch.zeros_like(input_ids)
        for i, x in enumerate(ids_list):
            L = int(x.size(0)); input_ids[i,:L] = x; attn[i,:L] = 1
        input_ids = input_ids.to(args.device); attn = attn.to(args.device)

        # 任务类型判别
        ttypes = [guess_task_type(ex, args.task_type) for ex in batch]
        # 分类索引
        I_bool = [i for i,t in enumerate(ttypes) if t=="qa_bool"]
        I_mcq  = [i for i,t in enumerate(ttypes) if t=="mcq"]
        I_math = [i for i,t in enumerate(ttypes) if t=="math_numeric"]
        I_code = [i for i,t in enumerate(ttypes) if t=="code_gen"]
        I_free = [i for i,t in enumerate(ttypes) if t not in {"qa_bool","mcq","math_numeric","code_gen"}]

        # 1) Bool 判别
        if I_bool:
            preds, raw = score_yes_no(model, tok, input_ids[I_bool], attn[I_bool], args.device)
            for k,i_src in enumerate(I_bool):
                ex = batch[i_src]
                out_rows.append({"id":ex.get("id"),"dataset":ex.get("dataset"),"pred":preds[k],"raw":raw[k],"ok":True,"reason":"logprob"})

        # 2) MCQ 判别（K 由 options 推断，默认 4）
        if I_mcq:
            Ks = []
            for i in I_mcq:
                opts = batch[i].get("options")
                Ks.append(len(opts) if isinstance(opts,list) and 2 <= len(opts) <= 6 else 4)
            K = max(Ks) if Ks else 4
            preds, raw = score_mcq_abcd(model, tok, input_ids[I_mcq], attn[I_mcq], args.device, K=K)
            for k,i_src in enumerate(I_mcq):
                ex = batch[i_src]
                out_rows.append({"id":ex.get("id"),"dataset":ex.get("dataset"),"pred":preds[k],"raw":raw[k],"ok":True,"reason":"logprob"})

        # 3) 数字题：受限解码 + 解析 + 必要时 answer-only 重试
        if I_math:
            sub_ids = input_ids[I_math]; sub_mask = attn[I_math]
            procs = make_anti_code_processors(tok, enable=True)
            with torch.no_grad():
                gen = model.generate(
                    input_ids=sub_ids, attention_mask=sub_mask,
                    max_new_tokens=min(args.max_new_tokens, 24),
                    do_sample=False, eos_token_id=eos_id, pad_token_id=pad_id,
                    logits_processor=procs, use_cache=True
                )
            for j,i_src in enumerate(I_math):
                ex = batch[i_src]
                cut = int(sub_mask[j].sum().item())
                text = tok.decode(gen[j][cut:], skip_special_tokens=True).strip()
                pred, ok, reason = parse_numeric_tail(text)
                if not ok:
                    q_text = (ex.get("prompt") or ex.get("question") or ex.get("input") or "").strip()
                    try:
                        text2 = gsm8k_answer_only_retry(model, tok, q_text, args.device, eos_id, pad_id, max_new=12)
                        pred2, ok2, reason2 = parse_numeric_tail(text2)
                        if ok2: pred, text, ok, reason = pred2, text2, ok2, "retry_answer_only"
                    except Exception:
                        pass
                out_rows.append({"id":ex.get("id"),"dataset":ex.get("dataset"),"pred":pred,"raw":text,"ok":ok,"reason":reason})

        # 4) 代码生成：放开
        if I_code:
            sub_ids = input_ids[I_code]; sub_mask = attn[I_code]
            with torch.no_grad():
                gen = model.generate(
                    input_ids=sub_ids, attention_mask=sub_mask,
                    max_new_tokens=args.max_new_tokens, do_sample=False,
                    eos_token_id=eos_id, pad_token_id=pad_id, use_cache=True
                )
            for j,i_src in enumerate(I_code):
                ex = batch[i_src]
                cut = int(sub_mask[j].sum().item())
                text = tok.decode(gen[j][cut:], skip_special_tokens=True).strip()
                out_rows.append({"id":ex.get("id"),"dataset":ex.get("dataset"),"pred":text,"raw":text,"ok":True,"reason":"free"})

        # 5) 其它自由文本：不做强解析
        if I_free:
            sub_ids = input_ids[I_free]; sub_mask = attn[I_free]
            procs = make_anti_code_processors(tok, enable=(args.anti_code=="on"))
            with torch.no_grad():
                gen = model.generate(
                    input_ids=sub_ids, attention_mask=sub_mask,
                    max_new_tokens=args.max_new_tokens, do_sample=False,
                    eos_token_id=eos_id, pad_token_id=pad_id, logits_processor=procs, use_cache=True
                )
            for j,i_src in enumerate(I_free):
                ex = batch[i_src]
                cut = int(sub_mask[j].sum().item())
                text = tok.decode(gen[j][cut:], skip_special_tokens=True).strip()
                out_rows.append({"id":ex.get("id"),"dataset":ex.get("dataset"),"pred":text,"raw":text,"ok":True,"reason":"free"})

    write_jsonl(args.out, out_rows)
    print(f"[infer] wrote {args.out}, n={len(out_rows)}")

if __name__ == "__main__":
    main()
