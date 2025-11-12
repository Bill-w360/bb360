from __future__ import annotations
import os, json
from typing import Dict, Any, Optional

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

def parse_param_to_billion(param_str: str) -> float:
    if not param_str:
        return 0.0
    s = str(param_str).upper().strip()
    try:
        if s.endswith("B"):
            return float(s[:-1])
        if s.endswith("M"):
            return float(s[:-1]) / 1000.0
        val = float(s)
        return val / 1e9 if val > 1e6 else val
    except Exception:
        return 0.0

def build_default_benchmark_mapping() -> Dict[str, str]:
    return {
        # 数学推理
        "GSM8K": "math_reasoning",
        "MATH": "math_reasoning",
        "SVAMP": "math_reasoning",
        # 代码能力
        "HumanEval": "coding",
        "MBPP": "coding",
        "CodeXGLUE": "coding",
        # 常识推理
        "ARC": "commonsense_reasoning",
        "HellaSwag": "commonsense_reasoning",
        "Winogrande": "commonsense_reasoning",
        # 知识问答
        "MMLU": "knowledge_qa",
        "C-Eval": "knowledge_qa",
        "AGIEval": "knowledge_qa",
        # 阅读理解
        "SQuAD": "reading_comprehension",
        "DROP": "reading_comprehension",
        "RACE": "reading_comprehension",
        # 中文能力
        "CMMLU": "chinese_capability",
        "Gaokao": "chinese_capability",
        # 安全与对齐
        "TruthfulQA": "safety_alignment",
        "BBQ": "safety_alignment",
        # 多语言/跨域
        "XCOPA": "multilingual",
        "XStoryCloze": "multilingual",
        # 创意写作
        "CreativeWriting": "creative_writing",
    }
