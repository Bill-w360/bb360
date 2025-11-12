# -*- coding: utf-8 -*-
import os, json, argparse, random, pathlib
from typing import List, Dict, Any
import numpy as np
import torch, torch.nn as nn
from datasets import Dataset
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments, DataCollatorWithPadding, TrainerCallback
import inspect

# --- hotfix: ignore keep_torch_compile for older accelerate ---
try:
    import inspect
    from accelerate import Accelerator
    if "keep_torch_compile" not in inspect.signature(Accelerator.unwrap_model).parameters:
        _orig_unwrap = Accelerator.unwrap_model
        def _compat_unwrap(self, *args, **kwargs):
            kwargs.pop("keep_torch_compile", None)
            return _orig_unwrap(self, *args, **kwargs)
        Accelerator.unwrap_model = _compat_unwrap
        print("[hotfix] Patched Accelerator.unwrap_model to ignore keep_torch_compile")
except Exception as e:
    print("[hotfix] accelerate patch skipped:", e)
# --------------------------------------------------------------


# ========== 小工具 ==========
def seed_all(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def read_any_json(path: str) -> List[Dict[str, Any]]:
    """支持 .json（数组或{'data':[...] }）与 .jsonl"""
    p = pathlib.Path(path)
    rows = []
    if p.suffix.lower() == ".jsonl":
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                if line: rows.append(json.loads(line))
    else:
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], list):
            rows = obj["data"]
        elif isinstance(obj, list):
            rows = obj
        else:
            raise ValueError("JSON 格式不符合：应为数组或含 data 的对象。")
    return rows

def extract_choices(ex: Dict[str, Any]) -> List[str]:
    # 支持：choices(List[str]/List[dict{text:...}] / dict{A:...,B:...})，或 options 同义字段
    ch = ex.get("choices", ex.get("options"))
    if ch is None: return []
    if isinstance(ch, dict):
        return [ch[k] for k in sorted(ch.keys())]
    if isinstance(ch, list):
        out=[]
        for c in ch:
            if isinstance(c, str): out.append(c)
            elif isinstance(c, dict):
                for key in ("text","option","content","value","label"):
                    if key in c and isinstance(c[key], str):
                        out.append(c[key]); break
        return out
    return []

def fmt(q: str, choices: List[str]) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if choices:
        opts = " ".join(f"{letters[i]}. {c}" for i, c in enumerate(choices))
        return f"Q: {q}  Options: {opts}"
    return f"Q: {q}"

def load_mapping(mp_path: str, super_id_path: str):
    with open(mp_path,"r",encoding="utf-8") as f: s2sup = json.load(f)
    with open(super_id_path,"r",encoding="utf-8") as f: sup2id = json.load(f)
    super_names = [name for name,_id in sorted(sup2id.items(), key=lambda kv: kv[1])]
    return s2sup, sup2id, super_names

class LogToJSONLCallback(TrainerCallback):
    """将 Trainer 的 on_log 输出追加写入 JSONL 文件，方便后续分析曲线。"""
    def __init__(self, log_path: str):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        # 重新训练时先清空旧日志
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write("")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        rec = dict(logs)
        # 补充 step / epoch 信息（有些版本里 epoch 会是 None）
        rec["step"] = int(state.global_step)
        if state.epoch is not None:
            rec["epoch"] = float(state.epoch)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ========== 模型 ==========
class MTLClassifier(nn.Module):
    """编码器骨干 + 两个线性头（57学科，8上位域）"""
    def __init__(self, model_name: str, num_sub: int, num_sup: int,
                 alpha: float=0.7, dropout: float=0.1, local_files_only: bool=False):
        super().__init__()

        # 为了兼容 Trainer._issue_warnings_after_load 里的访问
        self._keys_to_ignore_on_save = None
        self._keys_to_ignore_on_load_unexpected = set()
        self._keys_to_ignore_on_load_missing = set()

        self.model_name = model_name
        self.alpha = alpha
        self.backbone = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
        h = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.head_sub = nn.Linear(h, num_sub)
        self.head_sup = nn.Linear(h, num_sup)
        self.ce_sub = nn.CrossEntropyLoss()
        self.ce_sup = nn.CrossEntropyLoss()

    def forward(self, input_ids=None, attention_mask=None, labels_sub=None, labels_sup=None):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # === 改成与推理一致的均值池化 ===
        last = out.last_hidden_state                            # [B, L, H]
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).to(last.dtype)  # [B, L, 1]
            h = (last * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        else:
            # 理论上不会走到这里，兜底用 CLS
            h = last[:, 0, :]
        # ==========================
        h = self.dropout(h)
        logit_sub = self.head_sub(h)
        logit_sup = self.head_sup(h)
        loss=None
        if labels_sub is not None and labels_sup is not None:
            loss = self.alpha * self.ce_sub(logit_sub, labels_sub) + (1-self.alpha) * self.ce_sup(logit_sup, labels_sup)
        return {"loss": loss, "logits": torch.cat([logit_sub, logit_sup], dim=-1),
                "logits_sub": logit_sub, "logits_sup": logit_sup}

    def save_all(self, outdir: str, meta: Dict[str, Any]):
        os.makedirs(outdir, exist_ok=True)
        self.backbone.save_pretrained(outdir)
        torch.save({"meta": meta,
                    "head_sub": self.head_sub.state_dict(),
                    "head_sup": self.head_sup.state_dict(),
                    "model_name": self.model_name,
                    "alpha": self.alpha},
                   os.path.join(outdir, "mtl_heads.pt"))

class MTLTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        ysub = inputs.pop("labels_sub")
        ysup = inputs.pop("labels_sup")
        outputs = model(**inputs, labels_sub=ysub, labels_sup=ysup)
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss
    def save_model(self, output_dir: str=None, _internal_call: bool=False):
        outdir = output_dir or self.args.output_dir
        self.model.save_all(outdir, meta=self.args.meta_save)

# ========== 数据 ==========
def build_texts_and_labels(rows, subjects, s2sup, sup2id):
    X, ysub, ysup, skipped = [], [], [], 0
    for r in rows:
        subj = r.get("subject")
        if subj not in subjects:
            skipped += 1; continue
        q = r.get("question") or r.get("prompt") or r.get("input") or ""
        ch = extract_choices(r)
        X.append(fmt(str(q), [str(x) for x in ch]))
        ysub.append(subjects.index(subj))
        ysup.append(sup2id[s2sup[subj]])
    return X, np.array(ysub, np.int64), np.array(ysup, np.int64), skipped

def make_ds(tokenizer, texts, ysub, ysup, max_len):
    enc = tokenizer(texts, truncation=True, max_length=max_len)
    enc["labels_sub"] = ysub
    enc["labels_sup"] = ysup
    return Dataset.from_dict(enc)

# ========== 主程序 ==========
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-json", required=True)           # e.g., validation.json
    ap.add_argument("--eval-json", default=None)             # 可选：单独验证集，没有就不评估
    ap.add_argument("--mapping", default="data/mappings/subject_to_super.json")
    ap.add_argument("--super2id", default="data/mappings/super_to_id.json")
    ap.add_argument("--pretrained", default="roberta-base")
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--outdir", default="outputs/mmlu_mtl_fromjson")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--offline", action="store_true", help="完全离线：不访问网络，只用本地模型文件")
    args = ap.parse_args()

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    seed_all(42)
    s2sup, sup2id, super_names = load_mapping(args.mapping, args.super2id)

    train_rows = read_any_json(args.train_json)
    eval_rows  = read_any_json(args.eval_json) if args.eval_json else []

    # 统一标签空间（来自 train + eval）
    subjects = sorted(set([r.get("subject") for r in (train_rows + eval_rows) if r.get("subject")]))
    miss = [s for s in subjects if s not in s2sup]
    if miss:
        raise ValueError(f"subject_to_super.json 缺少学科映射：{miss}")

    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir,"label_subjects.json"),"w",encoding="utf-8") as f:
        json.dump(subjects, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.outdir,"super_names.json"),"w",encoding="utf-8") as f:
        json.dump(super_names, f, ensure_ascii=False, indent=2)

    tok = AutoTokenizer.from_pretrained(args.pretrained, use_fast=True, local_files_only=args.offline)

    Xtr, ytr_sub, ytr_sup, skip_tr = build_texts_and_labels(train_rows, subjects, s2sup, sup2id)
    if not Xtr: raise ValueError("训练集为空，请检查 train-json 字段/subject。")
    dtr = make_ds(tok, Xtr, ytr_sub, ytr_sup, args.max_length)
    has_eval = False
    if eval_rows:
        Xev, yev_sub, yev_sup, skip_ev = build_texts_and_labels(eval_rows, subjects, s2sup, sup2id)
        if Xev and len(yev_sub) and len(yev_sup):
            dev = make_ds(tok, Xev, yev_sub, yev_sup, args.max_length)
            has_eval = True
            print(f"[load] train={len(dtr)} (skipped={skip_tr})  eval={len(dev)} (skipped={skip_ev})")
        else:
            dev = None
            print(f"[load] train={len(dtr)} (skipped={skip_tr})  eval=0 (skip)")
    else:
        dev = None
        print(f"[load] train={len(dtr)} (skipped={skip_tr})")

    model = MTLClassifier(args.pretrained, num_sub=len(subjects), num_sup=len(super_names), alpha=args.alpha)
    collator = DataCollatorWithPadding(tokenizer=tok)

    def build_training_args(args, has_eval):
        full_kwargs = {
            "output_dir": args.outdir,
            "num_train_epochs": args.epochs,
            "per_device_train_batch_size": args.batch_size,
            "per_device_eval_batch_size": args.batch_size,
            "gradient_accumulation_steps": args.grad_accum,
            "learning_rate": args.lr,
            "logging_steps": 50,
            "evaluation_strategy": "epoch" if has_eval else "no",
            "save_strategy": "epoch",
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.06,
            "fp16": getattr(args, "fp16", False),
            "bf16": getattr(args, "bf16", False),
            "load_best_model_at_end": False,
            "metric_for_best_model": None,
            "greater_is_better": True,
            "label_names": ["labels_sub","labels_sup"],
            "report_to": []
        }
        full_kwargs = {k: v for k, v in full_kwargs.items() if v is not None}
        sig = inspect.signature(TrainingArguments.__init__)
        supported = set(sig.parameters.keys())
        safe_kwargs = {k: v for k, v in full_kwargs.items() if k in supported}
        return TrainingArguments(**safe_kwargs)

    targs = build_training_args(args, has_eval)

    # 保存元信息到权重文件 —— 增加 subjects / super_names，防止日后乱序
    targs.meta_save = {
        "num_subjects": len(subjects),
        "num_supers": len(super_names),
        "subjects": subjects,
        "super_names": super_names,
    }

    def compute_metrics(p):
        n_sub = len(subjects)

        preds = p.predictions
        # 兼容多种返回形式：np.ndarray 或 tuple(...)
        if isinstance(preds, (list, tuple)):
            # 常见两种情况：
            # 1) (logits,)  => 第一项就是我们要的
            # 2) (logits_sub, logits_sup) => 直接拿来用
            if len(preds) == 2 and preds[0].ndim == 2 and preds[1].ndim == 2:
                logits_sub = preds[0]
                logits_sup = preds[1]
            else:
                logits = preds[0]
                logits_sub = logits[:, :n_sub]
                logits_sup = logits[:, n_sub:]
        else:
            logits = preds
            logits_sub = logits[:, :n_sub]
            logits_sup = logits[:, n_sub:]

        yhat_sub = logits_sub.argmax(-1)
        yhat_sup = logits_sup.argmax(-1)

        # label_ids 也可能是 tuple
        if isinstance(p.label_ids, (list, tuple)) and len(p.label_ids) == 2:
            ytrue_sub, ytrue_sup = p.label_ids
        else:
            ytrue_sub, ytrue_sup = p.label_ids, None

        acc_sub = float((yhat_sub == ytrue_sub).mean())
        acc_sup = float((yhat_sup == ytrue_sup).mean()) if ytrue_sup is not None else 0.0
        return {"acc_sub": acc_sub, "acc_sup": acc_sup}

    # === 挂上 JSONL 日志回调 ===
    log_path = os.path.join(args.outdir, "training_log.jsonl")
    log_cb = LogToJSONLCallback(log_path)

    trainer = MTLTrainer(
        model=model, args=targs,
        train_dataset=dtr, eval_dataset=dev,
        tokenizer=tok, data_collator=collator,
        compute_metrics=compute_metrics if has_eval else None,
        callbacks=[log_cb]
    )
    trainer.train()
    tok.save_pretrained(args.outdir)
    trainer.save_model(args.outdir)
    print(f"[save] 模型与标签已保存到: {args.outdir}")
    print(f"[log] 训练过程日志已写入: {log_path}")

if __name__ == "__main__":
    main()
