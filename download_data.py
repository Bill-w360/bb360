#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
批量下载评测数据集（优先 ModelScope，失败回退 HuggingFace 国内镜像）。
支持多 config 数据集（MMLU、GSM8K等），并统一保存为 datasets 的 Arrow 目录。
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple

# ====== 环境与缓存路径（按需改） ======
os.environ.setdefault("MODELSCOPE_CACHE", "/mnt/Data/yangyongbiao/.cache/modelscope")
os.environ.setdefault("MODELSCOPE_DATASETS_CACHE", "/mnt/Data/yangyongbiao/.cache/modelscope/datasets")
os.environ.setdefault("MODELSCOPE_HUB_MIRROR", "https://modelscope.cn")

os.environ.setdefault("HF_HOME", "/mnt/Data/yangyongbiao/.cache/huggingface")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")  # HuggingFace 国内镜像
os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")

TARGET_ROOT = Path("/mnt/Data/yangyongbiao/datasets_eval").expanduser()
TARGET_ROOT.mkdir(parents=True, exist_ok=True)

# ====== 数据集配置 ======
# 每个条目：(modelscope_name, hf_name, configs, split)
# - configs: None 表示无子配置；list 表示要循环下载的子配置
# - split: 默认 split；也可以在 special_splits 里对某些 config 单独覆盖
DATASETS: List[Tuple[str, str, Optional[List[str]], str]] = [
    ("modelscope/boolq",     "google/boolq",           None,                  "validation"),
    ("modelscope/hellaswag", "Rowan/hellaswag",        None,                  "validation"),
    ("modelscope/super_glue","super_glue",             ["rte","wic","cb","copa","wsc","multirc","record"], "validation"),
    ("modelscope/mmlu",      "cais/mmlu",              ["all","nutrition"],   "validation"),
    ("modelscope/gsm8k",     "gsm8k",                  ["main"],              "test"),
]

# 对某些 (hf_name, config) 指定特殊 split（若不同于上面的默认 split）
SPECIAL_SPLITS = {
    # 例：("super_glue", "record"): "test",
}

def _safe_name(*parts: str) -> str:
    return "__".join(p for p in parts if p).replace("/", "__")

def _save_hf_dataset(ds, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out_dir))

def _to_hf_if_needed(ds):
    # ModelScope 的 MsDataset 支持 to_hf_dataset()
    if hasattr(ds, "to_hf_dataset"):
        try:
            return ds.to_hf_dataset()
        except Exception:
            pass
    return ds

def _try_modelscope(ms_name: str, subset: Optional[str], split: str):
    # 避免因 modelscope 依赖缺失而崩溃：缺什么库，外层会 fallback 到 HF
    try:
        from modelscope import MsDataset  # 可能触发 ImportError 或 addict 未安装
    except Exception as e:
        raise RuntimeError(f"import modelscope 失败: {e}")
    # MsDataset.load 支持 subset_name / split
    kwargs = {"split": split}
    if subset:
        kwargs["subset_name"] = subset
    return MsDataset.load(ms_name, **kwargs)

def _try_huggingface(hf_name: str, config: Optional[str], split: str):
    from datasets import load_dataset
    if config:
        return load_dataset(hf_name, config, split=split)
    return load_dataset(hf_name, split=split)

def _download_one(ms_name: str, hf_name: str,
                  config: Optional[str], default_split: str):
    # 目标落地路径
    # 命名：<源名>__<config>__<split>
    base = ms_name or hf_name
    split = SPECIAL_SPLITS.get((hf_name, config), default_split)
    tag = _safe_name(base, config or "", split)
    out_dir = TARGET_ROOT / tag

    if out_dir.exists():
        print(f"✅ 已存在，跳过：{out_dir}")
        return

    # 先 ModelScope
    if ms_name:
        try:
            print(f"[ModelScope] 尝试: {ms_name} | config={config} | split={split}")
            ds_ms = _try_modelscope(ms_name, config, split)
            hf_ds = _to_hf_if_needed(ds_ms)
            print(f"   -> 成功(ModelScope)，样本≈ {len(hf_ds)}")
            _save_hf_dataset(hf_ds, out_dir)
            print(f"💾 保存到：{out_dir}")
            return
        except Exception as e:
            print(f"⚠️  ModelScope 失败: {e}")

    # 再 HuggingFace（走国内镜像）
    if hf_name:
        try:
            print(f"[HuggingFace] 尝试: {hf_name} | config={config} | split={split}")
            ds_hf = _try_huggingface(hf_name, config, split)
            print(f"   -> 成功(HF)，样本≈ {len(ds_hf)}")
            _save_hf_dataset(ds_hf, out_dir)
            print(f"💾 保存到：{out_dir}")
            return
        except Exception as e:
            print(f"❌  HuggingFace 失败: {e}")

    print(f"⛔ 放弃：{base} | config={config} | split={split}")

def main():
    print("=== 批量下载评测数据集（优先 ModelScope，回退 HuggingFace 镜像）===")
    print(f"[INFO] MODELSCOPE_CACHE = {os.environ.get('MODELSCOPE_CACHE')}")
    print(f"[INFO] HF_HOME         = {os.environ.get('HF_HOME')}")
    print(f"[INFO] HF_ENDPOINT     = {os.environ.get('HF_ENDPOINT')}")
    print(f"[INFO] 保存目录         = {TARGET_ROOT}")

    for ms_name, hf_name, configs, split in DATASETS:
        title = ms_name or hf_name
        if not configs:  # 无子配置
            print(f"\n=== 🚀 开始：{title} (split={split}) ===")
            _download_one(ms_name, hf_name, None, split)
        else:
            print(f"\n=== 🚀 开始：{title}（多 config）===")
            for cfg in configs:
                print(f"\n—> 子配置：{cfg} (默认 split={split})")
                _download_one(ms_name, hf_name, cfg, split)

    print("\n🎉 全部完成。再次运行可续传/跳过已完成部分。")

if __name__ == "__main__":
    main()
