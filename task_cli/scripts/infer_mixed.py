# -*- coding: utf-8 -*-
"""
混合数据集本地推理（HF Transformers）
- 自动识别样本类型：mcq / boolq / gsm8k
- mcq/boolq：对选项做条件对数似然打分（支持多种打分模式）
- gsm8k：批量生成 + PoT 安全算式执行 + 多数表决 + 最终数字抽取

输入：JSONL，每行可来自不同数据集，字段尽量含：
  id, dataset(可选), prompt 或 question, options(可选), gold/answer(可选)

输出：JSONL，每行：
  {id, dataset, task_type, pred, pred_text, gold, correct, score, raw_text}

示例：
  python infer_mixed.py \
    --model /path/to/Qwen2.5-7B-Instruct \
    --input /mnt/Data/test.jsonl \
    --output preds_mixed.jsonl \
    --dtype float16 --batch 8 --trust-remote-code \
    --score-mode avg --math-chat auto --fewshot 6 --pot on
"""
import argparse, json, math, os, re, ast, random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm

try:
    import numpy as np
except Exception:
    np = None

LetterMap = ["A","B","C","D","E","F","G"]

# ================= 工具 =================
def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def pick_dtype(s: str):
    s = (s or "auto").lower()
    if s in ["float16", "fp16"]:
        return torch.float16
    if s in ["bfloat16", "bf16"]:
        return torch.bfloat16
    return None

def ensure_pad_token(tokenizer):
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

def has_chat_template(tok) -> bool:
    return getattr(tok, "chat_template", None) not in (None, "")

# Qwen/DeepSeek 通用兜底聊天格式
def apply_chat_fallback(msgs: List[Dict[str,str]]) -> str:
    def seg(role, content):
        tag = "system" if role=="system" else role
        return f"<|im_start|>{tag}\n{content}\n<|im_end|>\n"
    s = "".join(seg(m["role"], m["content"]) for m in msgs)
    s += "<|im_start|>assistant\n"  # generation prompt
    return s

# ============ few-shot 小样本 ============
FEWSHOT_ZH = [
    ("一只蜗牛每天前进 3 米，晚上下滑 1 米，2 天后共前进多少米？", "4"),
    ("小明买 3 支铅笔每支 2 元，又买 1 本本子 4 元，一共多少钱？", "10"),
    ("一辆车每小时行 60 千米，2.5 小时行驶多少千米？", "150"),
    ("把 3/4 写成小数是多少？", "0.75"),
]
FEWSHOT_EN = [
    ("A snail climbs 3 meters each day and slides 1 meter each night. After 2 days, how many meters has it climbed in total?", "4"),
    ("Alice buys 3 pencils at $2 each and 1 notebook at $4. What is the total cost?", "10"),
    ("A car travels at 60 km per hour. How far in 2.5 hours?", "150"),
    ("Convert 3/4 to a decimal.", "0.75"),
]

# ============ MCQ/BoolQ 聊天包装 ============
def render_mcq_chat(tok, prompt: str, options: List[str], zh=True) -> str:
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    opts_txt = "\n".join([f"{labels[i]}. {options[i]}" for i in range(len(options))])
    instr = "只输出选项字母（例如 A）。" if zh else "Output only the option letter (e.g., A)."
    content = f"{prompt}\n\n选项：\n{opts_txt}\n\n{instr}\n"
    messages = [
        {"role": "system", "content": "你是严谨的助教。"},
        {"role": "user", "content": content},
        {"role": "assistant", "content": "答案："}
    ]
    if has_chat_template(tok):
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return apply_chat_fallback(messages)

# ============ GSM8K 聊天包装（支持 PoT） ============
def render_gsm8k_chat(tokenizer, q: str, fewshot: int, lang: str, pot: bool) -> str:
    shots = (FEWSHOT_EN if lang=="en" else FEWSHOT_ZH)[:max(0, fewshot)]
    if lang=="en":
        sys = "You are a careful math tutor. Think briefly and precisely."
        ask_plain = "Output only the final line as '#### <number>'."
        ask_pot   = "First write a single Python arithmetic expression that computes the answer, then on the final line output '#### <number>'."
        akey = "Answer:"
    else:
        sys = "你是严谨的数学助教。请简洁推理。"
        ask_plain = "最后一行只输出 '#### 数字'。"
        ask_pot   = "先给出一条用于计算的 Python 算式，最后一行输出 '#### 数字'。"
        akey = "答案："
    ask = ask_pot if pot else ask_plain

    msgs = [{"role":"system","content":sys}]
    for s, a in shots:
        msgs += [
            {"role":"user","content": f"Question: {s}\n{ask}"},
            {"role":"assistant","content": f"{akey} #### {a}"}
        ]
    msgs += [
        {"role":"user","content": f"Question: {q}\n{ask}"},
        {"role":"assistant","content": akey}
    ]
    if has_chat_template(tokenizer):
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return apply_chat_fallback(msgs)

def build_gsm8k_prompt_plain(q: str, fewshot: int, lang: str, pot: bool) -> str:
    shots = FEWSHOT_EN if lang=="en" else FEWSHOT_ZH
    fs = "".join([f"Question: {s}\nAnswer: #### {a}\n\n" for s,a in shots[:max(0,fewshot)]])
    if lang=="en":
        tail = ("First write a single Python arithmetic expression that computes the answer, "
                "then on the final line output '#### <number>'.") if pot else "Output only the final line as '#### <number>'."
        return fs + f"Question: {q}\n{tail}\nAnswer:"
    else:
        tail = ("先给出一条用于计算的 Python 算式，最后一行输出 '#### 数字'。"
                if pot else "最后一行只输出 '#### 数字'。")
        return fs + f"问题：{q}\n{tail}\n答案："

# ============ 数字抽取 & PoT 安全求值 ============
def _to_float(x: str) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None

def extract_number(s: str) -> Optional[str]:
    # 归一化符号
    s = s.replace("，", ",").replace("％", "%").replace("−", "-")
    # \boxed{...}
    s = re.sub(r"\\boxed\{([^}]*)\}", r"\1", s)
    # 优先吃 '#### number'
    m = re.search(r"####\s*([+-]?\d[\d,]*(?:\.\d+)?%?)", s)
    if m:
        val = m.group(1)
    else:
        ms = re.findall(r"[+-]?\d[\d,]*(?:\.\d+)?%?", s)
        if ms:
            val = ms[-1]
        else:
            # 分数/带分数兜底
            m1 = re.search(r"([+-]?\d+)\s+(\d+)/(\d+)", s)  # 带分数 k a/b
            if m1:
                k, a, b = int(m1.group(1)), int(m1.group(2)), int(m1.group(3))
                return str(k + a / b)
            m2 = re.search(r"([+-]?\d+)/(\d+)(?!\d)", s)
            if m2:
                a, b = int(m2.group(1)), int(m2.group(2))
                return str(a / b)
            return None
    val = val.replace(",", "")
    if val.endswith("%"):
        f = _to_float(val[:-1])
        return str(f / 100.0) if f is not None else val[:-1]
    return val

def numeric_equal(a: Optional[str], b: Optional[str], tol: float = 0.0) -> Optional[bool]:
    if a is None or b is None:
        return None
    fa, fb = _to_float(a), _to_float(b)
    if fa is not None and fb is not None:
        return (abs(fa - fb) <= tol)
    return (str(a).strip() == str(b).strip())

# ---- PoT 安全求值 ----
def _safe_eval_expr(expr: str):
    import math
    ALLOWED_NODES = {
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Num, ast.Load,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
        ast.USub, ast.UAdd, ast.Constant, ast.Name, ast.Tuple, ast.List,
        ast.Call  # 但后面会禁止调用
    }
    ALLOWED_NAMES = {"pi": math.pi, "e": math.e}
    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if type(node) not in ALLOWED_NODES:
            raise ValueError(f"disallowed: {type(node).__name__}")
        if isinstance(node, ast.Call):
            raise ValueError("call not allowed")
        if isinstance(node, ast.Name) and node.id not in ALLOWED_NAMES:
            raise ValueError(f"name not allowed: {node.id}")
    return eval(compile(tree, "<expr>", "eval"), {"__builtins__": {}}, ALLOWED_NAMES)

def extract_pot_expr(text: str) -> Optional[str]:
    m = re.search(r"```python\s*([\s\S]+?)\s*```", text, re.I)
    if m: return m.group(1).strip()
    m = re.search(r"<calc>([\s\S]+?)</calc>", text, re.I)
    if m: return m.group(1).strip()
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    for ln in reversed(lines):
        if re.fullmatch(r"[0-9\.\s\+\-\*/%//\(\)\^]+", ln):
            return ln.replace("^","**")
    return None

def eval_from_text(text: str) -> Optional[str]:
    try:
        expr = extract_pot_expr(text)
        if not expr: return None
        val = _safe_eval_expr(expr)
        if isinstance(val, float):
            return str(int(val)) if abs(val - int(val)) < 1e-9 else str(val)
        return str(val)
    except Exception:
        return None

# ============ 类型判别（样本级） ============
YES_SET = {"yes","true","是","对","正确"}
NO_SET  = {"no","false","否","不对","错误"}

def norm_str(x: Any) -> str:
    return str(x).strip().lower()

def looks_like_boolq(options: Optional[List[str]], gold: Any) -> bool:
    if not options:
        if isinstance(gold, bool):
            return True
        if isinstance(gold, str) and norm_str(gold) in YES_SET | NO_SET:
            return True
        return False
    if len(options) != 2:
        return False
    a, b = norm_str(options[0]), norm_str(options[1])
    return (a in YES_SET or a in NO_SET) and (b in YES_SET or b in NO_SET)

def bool_gold_to_yn(g: Any) -> Optional[str]:
    if isinstance(g, bool):
        return "Yes" if g else "No"
    s = norm_str(g)
    if s in YES_SET or s in {"y","1"}:
        return "Yes"
    if s in NO_SET or s in {"n","0"}:
        return "No"
    return None

def detect_task_type(r: Dict[str, Any]) -> str:
    ds = norm_str(r.get("dataset", ""))
    options = r.get("options")
    gold = r.get("gold") if "gold" in r else r.get("answer")

    # 1) 优先用 dataset 名字
    if ds.startswith("gsm8k"):
        return "gsm8k"
    if ds.startswith("hellaswag") or ds.startswith("mmlu"):
        return "mcq"
    if ds.startswith("boolq"):
        return "boolq"

    # 2) 用结构推断
    if isinstance(options, list) and len(options) >= 3:
        return "mcq"
    if looks_like_boolq(options, gold):
        return "boolq"

    # 3) 数学题常见信号
    g = r.get("gold") if "gold" in r else r.get("answer")
    g_txt = str(g) if g is not None else ""
    if "####" in g_txt or re.search(r"-?\d+(?:\.\d+)?", g_txt):
        return "gsm8k"

    # 兜底
    return "mcq" if isinstance(options, list) and options else "gsm8k"

# ============ MCQ/BoolQ 打分 ============
@torch.no_grad()
def score_options(
    model, tokenizer,
    prompts: List[str], options_list: List[List[str]],
    batch_size: int, device: torch.device,
    mode: str = "avg", alpha: float = 0.7, target: str = "letter"
) -> List[List[float]]:
    """
    mode:
      - sum: 原始总 logprob
      - avg: 平均每 token logprob（推荐）
      - ppl: 平均负对数似然（取负号后仍然“越大越好”）
      - norm-alpha: sum / (len(option)**alpha)
    """
    ensure_pad_token(tokenizer)
    pairs: List[Tuple[int,int,List[int],List[int]]] = []
    for i, (p, opts) in enumerate(zip(prompts, options_list)):
        p_ids = tokenizer(p, add_special_tokens=False).input_ids
        for j, opt in enumerate(opts):
            if target == "letter":
                letter = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[j]
                o_ids = tokenizer(" " + letter, add_special_tokens=False).input_ids
            else:
                o_ids = tokenizer(" " + opt, add_special_tokens=False).input_ids
            pairs.append((i, j, p_ids, o_ids))

    scores_all = [[] for _ in range(len(prompts))]
    steps = math.ceil(len(pairs) / max(1, batch_size))
    it = range(0, len(pairs), batch_size)
    for k in tqdm(it, total=steps, desc="Scoring (MCQ/BoolQ)"):
        chunk = pairs[k:k+batch_size]
        input_ids, attn, plens, maxlen = [], [], [], 0
        for (_i,_j,p_ids,o_ids) in chunk:
            ids = p_ids + o_ids
            maxlen = max(maxlen, len(ids))
            input_ids.append(ids)
            attn.append([1]*len(ids))
            plens.append(len(p_ids))
        for t in range(len(input_ids)):
            pad = maxlen - len(input_ids[t])
            input_ids[t] += [tokenizer.pad_token_id]*pad
            attn[t] += [0]*pad

        input_ids = torch.tensor(input_ids, device=device)
        attn = torch.tensor(attn, device=device)
        logits = model(input_ids=input_ids, attention_mask=attn).logits  # [B,L,V]
        log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
        tgt = input_ids[:, 1:]
        tgt_lp = log_probs.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)

        for idx, (_i,_j,p_ids,o_ids) in enumerate(chunk):
            start = max(plens[idx]-1, 0)
            end = start + len(o_ids)
            val_sum = tgt_lp[idx, start:end].sum().item() if end > start else float("-inf")
            opt_len = max(1, len(o_ids))

            if mode == "sum":
                s = val_sum
            elif mode == "avg":
                s = val_sum / opt_len
            elif mode == "ppl":
                s = - val_sum / opt_len
            elif mode == "norm-alpha":
                s = val_sum / (opt_len ** alpha)
            else:
                s = val_sum
            while len(scores_all[_i]) < _j + 1:
                scores_all[_i].append(float("-inf"))
            scores_all[_i][_j] = s
    return scores_all

def mcq_pick_letter(scores: List[float]) -> str:
    j = max(range(len(scores)), key=lambda t: scores[t])
    return LetterMap[j]

def gold_to_letter(gold: Any, options: List[str]) -> Optional[str]:
    if isinstance(gold, int) and 0 <= gold < len(options):
        return LetterMap[gold]
    if isinstance(gold, str):
        g = gold.strip()
        if len(g) == 1 and g.upper() in LetterMap[:len(options)]:
            return g.upper()
        try:
            idx = options.index(g)
            return LetterMap[idx]
        except Exception:
            return None
    return None

# ============ GSM8K 生成 ============
@torch.no_grad()
def generate_batch(
    model, tokenizer, texts: List[str], device: torch.device,
    max_new_tokens: int, temperature: float, top_p: float,
    batch_size: int = 4
) -> List[str]:
    ensure_pad_token(tokenizer)
    # decoder-only 批量生成左填充
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    outs: List[str] = []
    steps = math.ceil(len(texts) / max(1, batch_size))
    for k in tqdm(range(0, len(texts), batch_size), total=steps, desc="Generating (GSM8K)"):
        chunk = texts[k:k+batch_size]
        ins = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True).to(device)

        do_sample = (temperature is not None and float(temperature) > 0.0)
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if do_sample:
            gen_kwargs.update({"temperature": float(temperature), "top_p": float(top_p)})

        y = model.generate(**ins, **gen_kwargs)

        for seq in y:
            dec = tokenizer.decode(seq, skip_special_tokens=True)
            # 尝试截取到 assistant 段的答案区
            pos = dec.rfind("答案：")
            if pos < 0:
                pos = dec.rfind("Answer:")
            ans = dec[pos+3:] if pos >= 0 else dec
            outs.append(ans.strip())
    return outs

# ============ 指标 ============
def acc_mcq(pred_letters: List[str], gold_letters: List[Optional[str]]) -> float:
    n = 0; c = 0
    for p,g in zip(pred_letters, gold_letters):
        if g is None: continue
        n += 1; c += int(p == g)
    return c/n if n else 0.0

def acc_exact(preds: List[Optional[str]], golds: List[Optional[str]]) -> float:
    n = 0; c = 0
    for p,g in zip(preds, golds):
        if p is None or g is None: continue
        n += 1; c += int(str(p).strip() == str(g).strip())
    return c/n if n else 0.0

# ============ 主流程 ============
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch", type=int, default=8, help="MCQ/BoolQ 的打分 batch；也作为 GSM8K 生成的 batch")
    ap.add_argument("--dtype", default="auto", choices=["auto","float16","bfloat16","bf16"])
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--score-mode", choices=["sum","avg","ppl","norm-alpha"], default="avg")
    ap.add_argument("--alpha", type=float, default=0.7, help="norm-alpha 的长度幂参数")
    ap.add_argument("--sc-vote", type=int, default=1, help="GSM8K 生成投票次数（>1 启用自洽采样）")
    ap.add_argument("--chat-wrap", choices=["auto","on","off"], default="auto", help="MCQ/BoolQ 是否用 chat_template 包装并按字母打分")
    ap.add_argument("--mcq-target", choices=["letter","text"], default="letter", help="MCQ/BoolQ 按字母还是按全文选项打分（聊天模型建议 letter）")
    ap.add_argument("--math-chat", choices=["auto","on","off"], default="auto", help="GSM8K 是否用 chat_template 包装")
    ap.add_argument("--fewshot", type=int, default=0, help="GSM8K few-shot 个数（0 表示不用）")
    ap.add_argument("--num-tol", type=float, default=0.0, help="GSM8K 数值比对绝对公差（0 表示严格相等）")
    ap.add_argument("--prompt-lang", choices=["zh","en"], default="en", help="GSM8K 提示语言（英文对小模型更稳）")
    ap.add_argument("--keep-raw", action="store_true", help="把生成文本写入 raw_text 便于排错")
    ap.add_argument("--debug-prompt", type=int, default=0, help="打印前 N 条 GSM8K 实际 prompt 到日志")
    ap.add_argument("--pot", choices=["off","on","auto"], default="auto",
                    help="GSM8K Program-of-Thought：模型输出算式，脚本本地求值")

    args = ap.parse_args()

    device = detect_device()
    torch_dtype = pick_dtype(args.dtype)
    print(f"[load] model={args.model} device={device} dtype={args.dtype}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch_dtype, device_map="auto", trust_remote_code=args.trust_remote_code
    )
    ensure_pad_token(tokenizer)
    model.eval()

    data = read_jsonl(args.input)
    n = len(data)
    print(f"[data] loaded {n} samples")

    # —— 判别类型并收集索引 —— #
    kinds = [detect_task_type(r) for r in data]
    idx_mcq   = [i for i,k in enumerate(kinds) if k in ("mcq","boolq")]
    idx_gsm8k = [i for i,k in enumerate(kinds) if k == "gsm8k"]

    out_rows: List[Dict[str, Any]] = [None] * n

    # ==================== MCQ/BoolQ 批处理 ====================
    if idx_mcq:
        prompts, opts_list, gold_letters, metas = [], [], [], []

        for i in idx_mcq:
            r = data[i]
            raw_p = r.get("prompt") or r.get("question") or ""
            tt = kinds[i]

            # 1) 选项
            opts = r.get("options")
            if tt == "boolq":
                if not opts:
                    opts = ["Yes", "No"]
                g_raw = r.get("gold") if "gold" in r else r.get("answer")
                g_norm = bool_gold_to_yn(g_raw)
                g_letter = gold_to_letter(g_norm, opts) if g_norm is not None else None
            else:
                if not isinstance(opts, list) or len(opts) < 2:
                    raise ValueError(f"样本 {r.get('id')} 缺少有效 options（>=2）用于 MCQ 评测")
                g_letter = gold_to_letter(r.get("gold"), opts)

            # 2) 聊天包装
            use_chat = (args.chat_wrap == "on") or (
                args.chat_wrap == "auto" and ("instruct" in args.model.lower()) and has_chat_template(tokenizer)
            )
            if use_chat and args.mcq_target == "letter":
                p = render_mcq_chat(tokenizer, raw_p, opts, zh=True)
            else:
                p = raw_p

            prompts.append(p)
            opts_list.append(opts)
            gold_letters.append(g_letter)
            metas.append((i, r.get("dataset"), tt, opts))

        scores = score_options(
            model, tokenizer, prompts, opts_list,
            batch_size=args.batch, device=device,
            mode=args.score_mode, alpha=args.alpha,
            target=args.mcq_target
        )
        preds_letters = [mcq_pick_letter(sc) for sc in scores]

        for (pl, sc, meta) in zip(preds_letters, scores, metas):
            i, ds, tt, opts = meta
            pred_idx = "ABCDEFG".index(pl)
            pred_text = opts[pred_idx] if 0 <= pred_idx < len(opts) else None
            g_raw = data[i].get("gold")
            if tt == "boolq":
                gold_text = bool_gold_to_yn(g_raw)
                gold_letter_cur = gold_to_letter(gold_text, opts) if gold_text is not None else None
            else:
                gold_text = g_raw
                gold_letter_cur = gold_to_letter(g_raw, opts)

            out_rows[i] = {
                "id": data[i].get("id"),
                "dataset": ds,
                "task_type": tt,
                "pred": pl,
                "pred_text": pred_text,
                "gold": gold_text,
                "correct": (pl == gold_letter_cur) if gold_letter_cur is not None else None,
                "score": max(sc) if sc else None,
                "raw_text": None,
            }

        acc = acc_mcq(preds_letters, gold_letters)
        print(f"[eval] MCQ/BoolQ accuracy = {acc:.4f} (n={len(idx_mcq)})")

    # ==================== GSM8K 批处理 ====================
    if idx_gsm8k:
        prompts, gold_nums, metas = [], [], []

        # 是否启用聊天包装
        use_math_chat = (args.math_chat == "on") or (
            args.math_chat == "auto" and ("instruct" in args.model.lower())
        )
        pot_on = (args.pot == "on") or (args.pot == "auto" and ("instruct" in args.model.lower()))

        for i in idx_gsm8k:
            r = data[i]
            q = r.get("question") or r.get("prompt") or ""
            if use_math_chat:
                p = render_gsm8k_chat(tokenizer, q, args.fewshot, args.prompt_lang, pot_on)
            else:
                p = build_gsm8k_prompt_plain(q, args.fewshot, args.prompt_lang, pot_on)
            prompts.append(p)

            g = r.get("gold") if "gold" in r else r.get("answer")
            gold_nums.append(str(g) if isinstance(g,(int,float)) else extract_number(str(g) if g is not None else ""))
            metas.append((i, r.get("dataset"), "gsm8k"))

        if args.debug_prompt > 0:
            print("=== [DEBUG] first prompts ===")
            for _i in range(min(args.debug_prompt, len(prompts))):
                print(f"[{_i}] >>>\n{prompts[_i]}\n<<<")

        vote = max(1, args.sc_vote)

        if vote == 1:
            gens = generate_batch(
                model, tokenizer, prompts, device,
                max_new_tokens=max(args.max_new, 192),
                temperature=args.temperature, top_p=args.top_p,
                batch_size=args.batch
            )
            gens_all = [gens]
        else:
            gens_all = []
            for v in range(vote):
                seed = 12345 + v
                ctx_devices = [device] if getattr(device, "type", "") == "cuda" else []
                with torch.random.fork_rng(devices=ctx_devices):
                    torch.manual_seed(seed)
                    if getattr(torch.cuda, "manual_seed_all", None) and getattr(device, "type", "") == "cuda":
                        torch.cuda.manual_seed_all(seed)
                    random.seed(seed)
                    if np is not None:
                        np.random.seed(seed)
                    gens_v = generate_batch(
                        model, tokenizer, prompts, device,
                        max_new_tokens=max(args.max_new, 192),
                        temperature=(args.temperature or 0.7),
                        top_p=(args.top_p or 0.95),
                        batch_size=args.batch
                    )
                gens_all.append(gens_v)

        # 聚合（优先 PoT 求值，其次数字抽取）
        from collections import Counter
        preds = []
        for idx in range(len(prompts)):
            cand_texts = [g[idx] for g in gens_all]
            pred_val = None
            if pot_on:
                vals = [eval_from_text(t) for t in cand_texts]
                vals = [v for v in vals if v is not None]
                if vals:
                    best, cmax = Counter(vals).most_common(1)[0]
                    pred_val = best
            if pred_val is None:
                nums = [extract_number(t) for t in cand_texts if extract_number(t) is not None]
                if nums:
                    cnt = Counter(nums).most_common()
                    best, cmax = cnt[0]
                    bests = [x for x,c in cnt if c == cmax]
                    if len(bests) == 1:
                        pred_val = best
                    else:
                        vals = [ _to_float(x) for x in bests if _to_float(x) is not None ]
                        pred_val = str(sorted(vals)[len(vals)//2]) if vals else bests[0]
            preds.append(pred_val)

        # 回填与评估
        corrects = []
        for t, (pred, meta) in enumerate(zip(preds, metas)):
            i, ds, tt = meta
            g_raw = data[i].get("gold") if "gold" in data[i] else data[i].get("answer")
            cor = numeric_equal(pred, extract_number(str(g_raw) if g_raw is not None else ""), tol=args.num_tol)
            row = {
                "id": data[i].get("id"),
                "dataset": ds,
                "task_type": tt,
                "pred": pred,
                "pred_text": pred,
                "gold": g_raw,
                "correct": bool(cor) if cor is not None else None,
                "score": None,
            }
            if args.keep_raw:
                row["raw_text"] = gens_all[0][t] if gens_all and len(gens_all[0])>t else None
            else:
                row["raw_text"] = None
            out_rows[i] = row
            corrects.append(bool(cor) if cor is not None else False)

        acc = (sum(1 for x in corrects if x) / len(corrects)) if corrects else 0.0
        print(f"[eval] GSM8K exact-match = {acc:.4f} (n={len(idx_gsm8k)})")

    # —— 合并写盘（填补 None） —— #
    for i in range(n):
        if out_rows[i] is None:
            r = data[i]
            out_rows[i] = {
                "id": r.get("id"),
                "dataset": r.get("dataset"),
                "task_type": kinds[i],
                "pred": None, "pred_text": None,
                "gold": r.get("gold") if "gold" in r else r.get("answer"),
                "correct": None, "score": None, "raw_text": None,
            }

    write_jsonl(args.output, out_rows)
    print(f"[save] -> {args.output}")

    # —— 统计 —— #
    from collections import defaultdict
    per_ds = defaultdict(lambda: [0,0])     # [n, correct]
    per_tt = defaultdict(lambda: [0,0])
    for r in out_rows:
        n1 = 1
        c1 = int(bool(r.get("correct"))) if r.get("correct") is not None else 0
        ds = r.get("dataset") or "unknown"
        tt = r.get("task_type") or "unknown"
        per_ds[ds][0]+=n1; per_ds[ds][1]+=c1
        per_tt[tt][0]+=n1; per_tt[tt][1]+=c1
    print("[report] per dataset:")
    for ds, (n1,c1) in per_ds.items():
        acc = (c1/n1) if n1 else 0.0
        print(f"  - {ds}: acc={acc:.4f} n={n1}")
    print("[report] per task_type:")
    for tt, (n1,c1) in per_tt.items():
        acc = (c1/n1) if n1 else 0.0
        print(f"  - {tt}: acc={acc:.4f} n={n1}")

if __name__ == "__main__":
    main()
