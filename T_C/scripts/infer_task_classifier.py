# -*- coding: utf-8 -*-
import os, json, argparse, pathlib
from typing import List, Dict, Any
import numpy as np
import torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoConfig

# ====== 工具 ======
def read_any_json(path: str) -> List[Dict[str, Any]]:
    p = pathlib.Path(path)
    rows=[]
    if p.suffix.lower()==".jsonl":
        with open(p,"r",encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                if line: rows.append(json.loads(line))
    else:
        with open(p,"r",encoding="utf-8") as f:
            obj=json.load(f)
        if isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], list):
            rows=obj["data"]
        elif isinstance(obj, list):
            rows=obj
        else:
            raise ValueError("JSON 格式不符合：应为数组或含 data 的对象。")
    return rows

def extract_choices(ex: Dict[str, Any]) -> List[str]:
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
    letters="ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if choices:
        opts=" ".join(f"{letters[i]}. {c}" for i,c in enumerate(choices))
        return f"Q: {q}  Options: {opts}"
    return f"Q: {q}"

def softmax_np(x):
    x = x - x.max(axis=-1, keepdims=True)
    ex = np.exp(x)
    return ex / ex.sum(axis=-1, keepdims=True)

def load_mapping(mp_path: str, super_id_path: str):
    with open(mp_path,"r",encoding="utf-8") as f: s2sup = json.load(f)
    with open(super_id_path,"r",encoding="utf-8") as f: sup2id = json.load(f)
    super_names = [name for name,_id in sorted(sup2id.items(), key=lambda kv: kv[1])]
    return s2sup, sup2id, super_names

# ====== 推理主程序 ======
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="训练输出目录（含 tokenizer / model / mtl_heads.pt）")
    ap.add_argument("--test-json", required=True, help="MMLU 的 test.json 或 test.jsonl")
    ap.add_argument("--mapping", default="data/mappings/subject_to_super.json")
    ap.add_argument("--super2id", default="data/mappings/super_to_id.json")
    ap.add_argument("--max-length", type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--pad-to-multiple-of", type=int, default=8)
    ap.add_argument("--out", default=None, help="预测输出路径；默认写到 model-dir 下 pred_test_1.jsonl")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--offline", action="store_true", help="完全离线：不访问网络，只用本地模型文件")
    args = ap.parse_args()

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # 上位域映射（用于计算 acc_sup & 概率）
    s2sup, sup2id, super_names = load_mapping(args.mapping, args.super2id)

    # 加载 tokenizer（训练时已 save_pretrained 到 model-dir）
    tok = AutoTokenizer.from_pretrained(
        args.model_dir, use_fast=True, local_files_only=args.offline
    )

    # 加载骨干（禁 pooler，更稳更快）
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float32

    cfg = AutoConfig.from_pretrained(args.model_dir, local_files_only=args.offline)
    cfg.add_pooling_layer = False

    backbone = AutoModel.from_pretrained(
        args.model_dir,
        config=cfg,
        local_files_only=args.offline,
        use_safetensors=True,
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
    ).to(device).eval()

    # 组装 subject 分类头并加载权重
    ckpt = torch.load(os.path.join(args.model_dir, "mtl_heads.pt"), map_location="cpu")
    meta = ckpt.get("meta", {})

    # 优先从 meta 中拿 subjects，保证与训练一致；否则退回到 label_subjects.json
    if "subjects" in meta:
        subjects = meta["subjects"]
    else:
        with open(os.path.join(args.model_dir,"label_subjects.json"),"r",encoding="utf-8") as f:
            subjects = json.load(f)

    n_sub = meta.get("num_subjects", len(subjects))
    assert n_sub == len(subjects), "meta.num_subjects 与 subjects 长度不一致"

    # 这里只用 head_sub，super 由 subject_to_super.json 映射，不再用 head_sup 输出
    class HeadSub(nn.Module):
        def __init__(self, hidden, n_sub):
            super().__init__()
            self.head_sub = nn.Linear(hidden, n_sub)
        def forward(self, h):
            return self.head_sub(h)

    head_sub = HeadSub(backbone.config.hidden_size, n_sub)
    head_sub.head_sub.load_state_dict(ckpt["head_sub"])

    model_dtype = next(backbone.parameters()).dtype
    head_sub.to(device=device, dtype=model_dtype).eval()
    torch.set_grad_enabled(False)

    # == 预先构建 "上位域 -> 对应学科 index 列表" ==
    num_sup = len(super_names)
    subidx_by_sup = {i: [] for i in range(num_sup)}
    for si, subj in enumerate(subjects):
        sup_name = s2sup[subj]                # e.g. "math_reasoning"
        sup_id = sup2id[sup_name]             # 0..7
        subidx_by_sup[sup_id].append(si)
    # 把 dict 变成 list，方便按顺序访问
    subidx_lists = [subidx_by_sup[i] for i in range(num_sup)]

    # 读入测试集并格式化文本
    rows = read_any_json(args.test_json)
    if not rows: raise ValueError("测试集为空。")
    texts = [fmt(str(r.get("question") or r.get("prompt") or r.get("input") or ""),
                 extract_choices(r)) for r in rows]

    # 批处理推理（均值池化 + 推理模式）
    B = args.batch_size
    MAXLEN = args.max_length
    PADMULT = args.pad_to_multiple_of

    def prob_batches(text_batch: List[str]):
        enc = tok(
            text_batch, return_tensors="pt", truncation=True, max_length=MAXLEN,
            padding="longest", pad_to_multiple_of=PADMULT
        ).to(device)
        with torch.inference_mode():
            out = backbone(**enc)
            # 均值池化（与训练一致）
            mask = enc["attention_mask"].unsqueeze(-1).to(out.last_hidden_state.dtype)
            h = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp_min(1.0)
            h = h.to(next(head_sub.parameters()).dtype)
            logit_sub = head_sub(h)
        ps = softmax_np(logit_sub.detach().cpu().numpy())
        return ps

    probs_sub_all = []
    for i in range(0, len(texts), B):
        ps = prob_batches(texts[i:i+B])
        probs_sub_all.append(ps)

    prob_sub = np.concatenate(probs_sub_all, 0)        # [N, 57]
    yhat_sub = prob_sub.argmax(-1)                     # [N]

    # 由学科预测映射出上位域预测（不再用 head_sup）
    pred_sup_names = [s2sup[subjects[int(idx)]] for idx in yhat_sub]

    # 保存预测（super 概率 = 对应学科概率之和，并给出 subject top-k）
    out_path = args.out or os.path.join(args.model_dir, "pred_test_1.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            subj_name = subjects[int(yhat_sub[i])]
            sup_name  = pred_sup_names[i]
            # 计算本样本的上位域概率分布
            prob_super = {}
            for sup_id, sup_n in enumerate(super_names):
                idxs = subidx_lists[sup_id]
                if idxs:
                    prob_super[sup_n] = float(prob_sub[i, idxs].sum())
                else:
                    prob_super[sup_n] = 0.0
            # 计算 top-k 学科预测（按 subject 概率排序）
            k = min(max(1, args.topk), len(subjects))
            topk_idx = np.argsort(-prob_sub[i])[:k]
            topk_subjects = [
                {"subject": subjects[int(j)], "prob": float(prob_sub[i, j])}
                for j in topk_idx
            ]
            rec = {
                **r,
                "_pred_subject": subj_name,
                "_pred_super": sup_name,
                "_prob_super": prob_super,
                "_topk_subjects": topk_subjects,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[save] 预测写入 {out_path}")

    # 如 test.json 含真值 subject，则计算准确率
    gt_idx, gt_sub = [], []
    for i, r in enumerate(rows):
        subj = r.get("subject")
        if subj in subjects:
            gt_idx.append(i)
            gt_sub.append(subjects.index(subj))

    if gt_idx:
        gt_idx = np.array(gt_idx, dtype=np.int64)
        gt_sub = np.array(gt_sub, dtype=np.int64)
        # 57 学科准确率：直接比较 subject index
        acc_sub = float((yhat_sub[gt_idx] == gt_sub).mean())

        # 8 上位域：用 subject_to_super 映射 GT 和预测
        gt_sup_idx = np.array(
            [sup2id[s2sup[subjects[s]]] for s in gt_sub],
            dtype=np.int64
        )
        yhat_sup_idx = np.array(
            [sup2id[s2sup[subjects[int(s)]]] for s in yhat_sub[gt_idx]],
            dtype=np.int64
        )
        acc_sup = float((yhat_sup_idx == gt_sup_idx).mean())

        metrics = {"test/acc_sub": acc_sub, "test/acc_sup": acc_sup, "num_eval": int(len(gt_idx))}
        mpath = os.path.join(args.model_dir, "metrics_test.json")
        with open(mpath, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"[test] 57学科 acc={acc_sub:.4f} | 8上位域 acc={acc_sup:.4f}  (评估样本数={len(gt_idx)})")
    else:
        print("[test] 未发现真值 subject，跳过准确率计算。")

if __name__ == "__main__":
    main()
