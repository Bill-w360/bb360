# -*- coding: utf-8 -*-
# 文件：multiscale_llm_eval/evaluator.py
# 说明：
#   - 提供 MultiScaleLLMEvaluator / ModelMeta
#   - 用于把 lm-eval 等评测汇总得到的 (model,dataset,metric,value) 聚合为“能力维度”并聚类
#
# 典型用法见：multiscale_llm_eval/cli.py 的 classify 子命令

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd

# 可选依赖（聚类/可视化）
try:
    from sklearn.cluster import KMeans
except Exception:
    KMeans = None  # 允许无 sklearn 的环境仅导出 CSV

# 画图非强制
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


# -------------------------
# 数据结构
# -------------------------
@dataclass
class ModelMeta:
    parameters: str = ""
    family: str = ""
    release_date: str = ""
    extras: Dict[str, float] = field(default_factory=dict)  # tps, vram_gb 等可选


# -------------------------
# 工具函数
# -------------------------
def _to_100_scale(x: float) -> float:
    """把分数统一到 0..100 区间：若 <=1 认为是 0..1 的比例，乘以 100。"""
    if x is None:
        return 0.0
    try:
        v = float(x)
    except Exception:
        return 0.0
    if math.isnan(v) or math.isinf(v):
        return 0.0
    if v <= 1.0:
        return max(0.0, min(100.0, v * 100.0))
    return max(0.0, min(100.0, v))


def _safe_mean(values: List[float]) -> float:
    vals = [float(x) for x in values if x is not None]
    if not vals:
        return 0.0
    return float(np.mean(vals))


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# -------------------------
# 评估器
# -------------------------
class MultiScaleLLMEvaluator:
    """
    把各数据集分数映射为“能力维度”，并对多个模型做聚类与导出。
    关键入口：
      - register_model(name, meta)
      - update_model_scores(model, {dataset: value})
      - generate_self_assessment()
      - cluster_models(n_clusters)
      - export_classification_csv(path)
      - create_all_visualizations()
    """

    def __init__(
        self,
        score_thresholds: Optional[Dict[str, float]] = None,
        save_file: Optional[str] = None,
        output_dir: Optional[str] = None,
        benchmark_mapping: Optional[Dict[str, str]] = None,
        agg_fn: str = "mean",  # 维度聚合函数：mean / max / median
    ) -> None:
        self.score_thresholds = score_thresholds or {"expert": 75, "competent": 50, "developing": 30}
        self.save_file = Path(save_file) if save_file else None
        self.output_dir = Path(output_dir) if output_dir else None
        self.agg_fn = agg_fn

        # “数据集 → 能力维度”的默认映射；会被 config.yaml 的 benchmark_mapping 覆盖/扩展
        default_mapping = {
            # 常见小基准
            "gsm8k": "math_reasoning",
            "hellaswag": "commonsense",
            "boolq": "reading",
            # MMLU 子域：给个粗归并（若 CSV 已做 group，可直接用它们的 group 名）
            "mmlu.humanities": "humanities",
            "mmlu.social_sciences": "social_sciences",
            "mmlu.stem": "stem",
            "mmlu.other": "other",
            # 有些人把 mmlu 直接作为整体
            "mmlu": "mmlu_overall",
        }
        # 用户自定义覆盖
        self.benchmark_mapping: Dict[str, str] = {**default_mapping, **(benchmark_mapping or {})}

        # 内部存储
        self.models_meta: Dict[str, ModelMeta] = {}
        # 原始：model -> dataset -> score(0..100)
        self.models_raw_scores: Dict[str, Dict[str, float]] = {}
        # 聚合：model -> dimension -> score(0..100)
        self.models_dim_scores: Dict[str, Dict[str, float]] = {}
        # 输出表（聚类后）
        self.classification_table: Optional[pd.DataFrame] = None

    # -------------------------
    # 数据注册与更新
    # -------------------------
    def register_model(self, name: str, meta: ModelMeta) -> None:
        self.models_meta[name] = meta

    def update_model_scores(self, model: str, dataset_scores: Dict[str, float]) -> None:
        """
        dataset_scores: {dataset_name: score}，分数可为 0..1 或 0..100
        """
        normed = {k: _to_100_scale(v) for k, v in dataset_scores.items()}
        self.models_raw_scores.setdefault(model, {}).update(normed)

    # -------------------------
    # 生成能力维度与整体评估
    # -------------------------
    def _map_dataset_to_dim(self, dataset: str) -> str:
        """
        按映射表找维度；若没有，就用 dataset 自身作为一个维度（不丢信息）。
        同时为 MMLU 这类“分组”留后门：若 dataset 形如 'mmlu - humanities'，做一次规整。
        """
        key = dataset.strip().lower()

        # 尝试从映射表命中
        if key in self.benchmark_mapping:
            return self.benchmark_mapping[key]

        # 兼容 "mmlu - humanities" / "mmlu_humanities" / "mmlu:humanities"
        if key.startswith("mmlu"):
            for sep in [" - ", "_", ":", "/"]:
                if sep in key:
                    sub = key.split(sep, 1)[1].strip()
                    # mmlu 子域映射
                    if sub in ("humanities", "social_sciences", "stem", "other"):
                        return sub
                    return f"mmlu.{sub}"

        # 未配置就直接把数据集名当维度
        return key

    def _aggregate_to_dimensions(self) -> None:
        """
        把 self.models_raw_scores 聚合到维度空间；默认同一维度多个数据集取均值。
        """
        dim_scores: Dict[str, Dict[str, List[float]]] = {}
        # 汇总同一维度的多个数据集
        for model, ds_scores in self.models_raw_scores.items():
            for ds, v in ds_scores.items():
                dim = self._map_dataset_to_dim(ds)
                dim_scores.setdefault(model, {}).setdefault(dim, []).append(float(v))

        # 聚合（mean/max/median）
        self.models_dim_scores = {}
        for model, dim_map in dim_scores.items():
            self.models_dim_scores[model] = {}
            for dim, vals in dim_map.items():
                if not vals:
                    self.models_dim_scores[model][dim] = 0.0
                    continue
                if self.agg_fn == "max":
                    self.models_dim_scores[model][dim] = float(np.max(vals))
                elif self.agg_fn == "median":
                    self.models_dim_scores[model][dim] = float(np.median(vals))
                else:
                    self.models_dim_scores[model][dim] = _safe_mean(vals)

    def _label_tier(self, score: float) -> str:
        """
        根据 score_thresholds 打标签。顺序：expert > competent > developing > novice
        """
        s = float(score)
        thr = self.score_thresholds
        if s >= thr.get("expert", 75):
            return "expert"
        if s >= thr.get("competent", 50):
            return "competent"
        if s >= thr.get("developing", 30):
            return "developing"
        return "novice"

    def generate_self_assessment(self) -> None:
        """
        完成从数据集到维度聚合；同时计算 overall_score / tier。
        """
        self._aggregate_to_dimensions()

        # 生成 DataFrame：包含 meta、各维度、overall_score、tier
        all_dims: List[str] = sorted({
            d for m in self.models_dim_scores.values() for d in m.keys()
        })

        rows: List[Dict[str, Any]] = []
        for model, dmap in self.models_dim_scores.items():
            row: Dict[str, Any] = {"model": model}
            # 注入 meta（若有）
            meta = self.models_meta.get(model)
            if meta:
                row.update({
                    "parameters": meta.parameters,
                    "family": meta.family,
                    "release_date": meta.release_date,
                })
                # 展平 extras
                for k, v in (meta.extras or {}).items():
                    row[k] = v
            # 能力维度
            for dim in all_dims:
                row[dim] = float(dmap.get(dim, 0.0))
            # overall（各维度均值）
            dim_vals = [float(dmap.get(dim, 0.0)) for dim in all_dims]
            overall = _safe_mean(dim_vals)
            row["overall_score"] = overall
            row["tier"] = self._label_tier(overall)
            rows.append(row)

        self.classification_table = pd.DataFrame(rows)
        # 保留三位小数
        self.classification_table = self.classification_table.round(3)


    # -------------------------
    # 聚类
    # -------------------------
    def cluster_models(self, n_clusters: int = 5, random_state: int = 0) -> None:
        if self.classification_table is None or self.classification_table.empty:
            return
        if KMeans is None:
            # 环境无 sklearn：跳过聚类
            self.classification_table["cluster"] = -1
            self.classification_table["cluster_name"] = "unclustered"
            return

        # 选取“能力维度列”：数值列里剔除 meta/标签列
        df = self.classification_table.copy()
        exclude_cols = {
            "model", "parameters", "family", "release_date", "tier",
            "tps", "vram_gb", "overall_score", "cluster", "cluster_name"
        }
        feature_cols = [c for c in df.columns if c not in exclude_cols and np.issubdtype(df[c].dtype, np.number)]

        if len(df) < 2 or len(feature_cols) == 0:
            # 样本或特征不足
            self.classification_table["cluster"] = -1
            self.classification_table["cluster_name"] = "unclustered"
            return

        # 聚类数裁剪到 [2, 样本数]
        k = int(n_clusters)
        k = max(2, min(k, len(df)))

        X = df[feature_cols].to_numpy(dtype=float)
        # 若完全相同向量，KMeans 可能报错；做个微扰
        if np.allclose(X.std(axis=0), 0.0):
            X = X + 1e-6 * np.random.RandomState(42).randn(*X.shape)

        km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        labels = km.fit_predict(X)

        self.classification_table["cluster"] = labels
        # 也许你想给簇起个“强/中/弱”名字；简单用中心的 overall 排序命名
        centers = []
        for c in range(k):
            idx = (labels == c)
            centers.append((c, float(df.loc[idx, "overall_score"].mean() if idx.any() else 0.0)))
        # overall 高的簇叫 A 类
        centers.sort(key=lambda t: t[1], reverse=True)
        rank_name = {cid: f"C{rank+1}" for rank, (cid, _) in enumerate(centers)}
        self.classification_table["cluster_name"] = self.classification_table["cluster"].map(rank_name)

    # -------------------------
    # 导出
    # -------------------------
    def export_classification_csv(self, path: str) -> None:
        if self.classification_table is None:
            # 空表也写出表头，避免下游崩
            pd.DataFrame(columns=["model", "overall_score", "tier", "cluster", "cluster_name"]).to_csv(
                path, index=False, encoding="utf-8"
            )
            return

        

        out = self.classification_table.copy()
        out = out.round(3)
        # 统一列顺序：meta → scores → overall → tier/cluster
        meta_cols = [c for c in ["model", "parameters", "family", "release_date", "tps", "vram_gb"] if c in out.columns]
        other_cols = [c for c in out.columns if c not in meta_cols + ["overall_score", "tier", "cluster", "cluster_name"]]
        ordered = meta_cols + other_cols + ["overall_score", "tier", "cluster", "cluster_name"]
        out = out.reindex(columns=ordered)
        _ensure_dir(Path(path).parent)
        out.to_csv(path, index=False, encoding="utf-8")

        # 同步保存一个 JSON 状态（可选）
        if self.save_file:
            try:
                state = {
                    "score_thresholds": self.score_thresholds,
                    "benchmark_mapping": self.benchmark_mapping,
                    "models_meta": {k: asdict(v) for k, v in self.models_meta.items()},
                    "models_raw_scores": self.models_raw_scores,
                    "models_dim_scores": self.models_dim_scores,
                    "table": out.to_dict(orient="records"),
                }
                self.save_state(state)
            except Exception:
                pass

    def save_state(self, state: Dict[str, Any]) -> None:
        if not self.save_file:
            return
        _ensure_dir(self.save_file.parent)
        with self.save_file.open("w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    # -------------------------
    # 可视化（非强制）
    # -------------------------
    def create_all_visualizations(self) -> None:
        if not HAS_MPL or self.classification_table is None or self.classification_table.empty:
            return
        if not self.output_dir:
            return
        out_dir = self.output_dir / "figs"
        _ensure_dir(out_dir)

        # 1) 能力维度热力图（模型×维度）
        try:
            self._plot_heatmap(out_dir / "heatmap.png")
        except Exception:
            pass

        # 2) 每模型雷达图（维度分布）
        try:
            self._plot_radars(out_dir)
        except Exception:
            pass

        # 3) overall_score 排序柱状图
        try:
            self._plot_overall_bars(out_dir / "overall_bars.png")
        except Exception:
            pass

    def _feature_columns(self) -> List[str]:
        df = self.classification_table
        assert df is not None
        exclude = {"model", "parameters", "family", "release_date", "tps", "vram_gb", "overall_score", "tier", "cluster", "cluster_name"}
        return [c for c in df.columns if c not in exclude and np.issubdtype(df[c].dtype, np.number)]

    def _plot_heatmap(self, save_path: Path) -> None:
        df = self.classification_table
        assert df is not None
        feats = self._feature_columns()
        if not feats:
            return
        mat = df.set_index("model")[feats]
        plt.figure(figsize=(max(6, 0.4 * len(feats)), max(4, 0.4 * len(mat))))
        plt.imshow(mat.values, aspect="auto")
        plt.colorbar(label="score (0-100)")
        plt.xticks(range(len(feats)), feats, rotation=45, ha="right")
        plt.yticks(range(len(mat.index)), mat.index)
        plt.title("Model × Dimension Heatmap")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def _plot_radars(self, out_dir: Path) -> None:
        df = self.classification_table
        assert df is not None
        feats = self._feature_columns()
        if not feats:
            return

        for _, row in df.iterrows():
            model = row["model"]
            vals = [float(row.get(f, 0.0)) for f in feats]
            # 闭合雷达
            labels = feats + [feats[0]]
            values = vals + [vals[0]]

            angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
            angles += angles[:1]

            fig = plt.figure(figsize=(5, 5))
            ax = plt.subplot(111, polar=True)
            ax.plot(angles, values, linewidth=2)
            ax.fill(angles, values, alpha=0.15)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(feats, fontsize=8)
            ax.set_yticklabels([])
            ax.set_title(f"Radar — {model}", y=1.08)
            plt.tight_layout()
            _ensure_dir(out_dir)
            plt.savefig(out_dir / f"radar_{str(model).replace('/', '_')}.png")
            plt.close()

    def _plot_overall_bars(self, save_path: Path) -> None:
        df = self.classification_table
        assert df is not None
        data = df[["model", "overall_score"]].sort_values("overall_score", ascending=False)
        plt.figure(figsize=(max(6, 0.4 * len(data))))
        plt.bar(data["model"], data["overall_score"])
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("overall_score (0-100)")
        plt.title("Overall Score")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
