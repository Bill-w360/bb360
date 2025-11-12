# -*- coding: utf-8 -*-
# 文件：multiscale_llm_eval/cli.py
# 功能（仅分类）：
#   从 scores.csv + 可选 meta/config 生成：
#     - 能力评估与聚类
#     - 导出 classification_results.csv
#     - 可选可视化到输出目录
#
# 用法示例：
#   python -m multiscale_llm_eval.cli classify \
#       --format csv \
#       --input out/scores.csv \
#       --meta meta.csv \
#       --config config.yaml \
#       --clusters 5 \
#       --output-dir out \
#       --skip-plots 0

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional

import yaml  # pip install pyyaml

# 依赖你的 evaluator 实现
from .evaluator import MultiScaleLLMEvaluator, ModelMeta


# ----------------------------
# 工具函数
# ----------------------------
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _load_csv_as_list(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        return list(rdr)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _parse_config_yaml(config_path: Optional[str]) -> Dict[str, Any]:
    if not config_path:
        return {}
    p = Path(config_path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def _load_meta_csv(meta_path: Optional[str]) -> Dict[str, ModelMeta]:
    """
    读取 meta.csv -> {model: ModelMeta(...)}
    允许（可选）列：parameters,family,release_date,tps,vram_gb 等
    """
    meta: Dict[str, ModelMeta] = {}
    if not meta_path:
        return meta
    p = Path(meta_path)
    if not p.exists():
        return meta

    rows = _load_csv_as_list(p)
    for r in rows:
        name = r.get("model") or r.get("Model") or r.get("name")
        if not name:
            continue
        m = ModelMeta(
            parameters=r.get("parameters", "") or r.get("params", ""),
            family=r.get("family", ""),
            release_date=r.get("release_date", "")
        )
        # 可选：运行侧元信息放入 extras（若 evaluator 支持）
        extras = {}
        for k in ("tps", "vram_gb"):
            if k in r and r[k] != "":
                extras[k] = _safe_float(r[k], 0.0)
        if extras and hasattr(m, "extras"):
            # type: ignore[attr-defined]
            getattr(m, "extras").update(extras)
        meta[name] = m
    return meta


def _ingest_scores_csv(scores_csv: str) -> Dict[str, Dict[str, float]]:
    """
    输入（标准列）：
      model,dataset,metric,value
    输出：
      {model: {dataset: value(浮点)}}
    注：
      - 若同一 (model,dataset) 多条，保留最后一条
      - metric 列目前不在此处聚合，交给 evaluator 内部做映射与融合
    """
    rows = _load_csv_as_list(Path(scores_csv))
    res: Dict[str, Dict[str, float]] = {}

    def lc(s: str) -> str:
        return s.lower() if isinstance(s, str) else s

    for r in rows:
        keys = {lc(k): k for k in r.keys()}
        if "model" not in keys or "dataset" not in keys or "value" not in keys:
            # 非法行，跳过
            continue
        m = r[keys["model"]]
        d = r[keys["dataset"]]
        v = _safe_float(r[keys["value"]], 0.0)
        if not m or not d:
            continue
        res.setdefault(m, {})[d] = v
    return res


# ----------------------------
# 分类主流程
# ----------------------------
def cmd_classify(args: argparse.Namespace) -> None:
    # 0) 准备输出目录
    out_dir = Path(args.output_dir).resolve()
    _ensure_dir(out_dir)

    # 1) 读取配置与元信息
    cfg = _parse_config_yaml(args.config)
    meta_map = _load_meta_csv(args.meta)

    # 2) 初始化评估器（阈值/映射从 config 中读取，可设默认）
    score_thresholds = cfg.get("score_thresholds", {"expert": 75, "competent": 50, "developing": 30})
    benchmark_mapping = cfg.get("benchmark_mapping", {})  # 例如把 hellaswag → commonsense_reasoning

    # 每次分类前清理旧状态，避免历史模型“串场”
    state_file = out_dir / "multiscale_llm_evaluation.json"
    if state_file.exists():
        try:
            state_file.unlink()
        except Exception:
            pass

    evaluator = MultiScaleLLMEvaluator(
        score_thresholds=score_thresholds,
        save_file=str(state_file),
        output_dir=str(out_dir),
        benchmark_mapping=benchmark_mapping,
    )

    # 3) 注册模型元信息（若 evaluator 支持）
    for mname, meta in meta_map.items():
        try:
            evaluator.register_model(mname, meta)
        except Exception:
            # 某些实现可能不强制注册；忽略异常
            pass

    # 4) 读取分数并注入
    model_scores = _ingest_scores_csv(args.input)
    for mname, scores in model_scores.items():
        evaluator.update_model_scores(mname, scores)

    # 5) 生成自我评估与聚类
    evaluator.generate_self_assessment()

    # 簇数不超过模型数（至少 2 簇，若模型数>=2）
    try:
        n_models = len(model_scores)
        k = int(args.clusters)
        if n_models >= 2:
            k = max(2, min(k, n_models))
        else:
            k = 2
        evaluator.cluster_models(n_clusters=k)
    except Exception:
        # 若聚类失败（如样本过少），忽略
        pass

    # 6) 导出分类结果 CSV
    out_csv = out_dir / "classification_results.csv"
    evaluator.export_classification_csv(str(out_csv))

    # 7) 可视化（可选）
    if not bool(int(args.skip_plots)):
        try:
            evaluator.create_all_visualizations()
        except Exception:
            # 绘图失败不影响主流程
            pass

    print(f"[classify] 导出完成：{out_csv}")


# ----------------------------
# CLI 入口
# ----------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Multiscale-LLM-Eval :: classify only")
    sp = p.add_subparsers(dest="command", required=True)

    sp_c = sp.add_parser("classify", help="从 scores.csv 生成能力评估与聚类结果（仅分类，无路由）")
    sp_c.add_argument("--format", required=True, choices=["csv"], help="当前仅支持 csv")
    sp_c.add_argument("--input", required=True, help="scores.csv（列：model,dataset,metric,value）")
    sp_c.add_argument("--meta", required=False, help="可选 meta.csv（列：model,parameters,family,release_date,tps,vram_gb...）")
    sp_c.add_argument("--config", required=False, help="可选 config.yaml（benchmark_mapping、score_thresholds 等）")
    sp_c.add_argument("--clusters", required=False, default=5, help="聚类簇数（自动裁剪到不超过模型数）")
    sp_c.add_argument("--output-dir", required=True, help="输出目录，如 out")
    sp_c.add_argument("--skip-plots", required=False, default=0, help="是否跳过可视化(0/1)")
    sp_c.set_defaults(func=cmd_classify)

    return p


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
