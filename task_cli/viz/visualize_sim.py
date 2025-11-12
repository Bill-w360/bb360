#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_sim.py
----------------
可视化 sim_run.py 的输出结果：
- 支持读取汇总 JSON（--summary）与分配明细 JSONL（--assign，来自 --dump-assign）。
- 生成每模型接单量、利用率图，以及等待/逗留时间分布图；同时输出一个可读的统计文本与 CSV。

使用示例：
  python visualize_sim.py --summary sim_result.json --assign assign.jsonl --outdir sim_plots --show

注意：仅使用 matplotlib，不依赖 seaborn。
"""

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Iterable

import matplotlib.pyplot as plt

def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                # 跳过损坏行
                continue
    return rows

def ensure_outdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def safe_quantiles(xs: List[float], qs: Iterable[float]) -> Dict[float, float]:
    """返回给定分位点的字典。"""
    if not xs:
        return {q: float("nan") for q in qs}
    xs_sorted = sorted(xs)
    out: Dict[float, float] = {}
    n = len(xs_sorted)
    for q in qs:
        if q <= 0:
            out[q] = xs_sorted[0]
        elif q >= 1:
            out[q] = xs_sorted[-1]
        else:
            # 采用线性插值分位数
            pos = q * (n - 1)
            lo = int(math.floor(pos))
            hi = int(math.ceil(pos))
            if lo == hi:
                out[q] = float(xs_sorted[lo])
            else:
                w = pos - lo
                out[q] = float(xs_sorted[lo]) * (1 - w) + float(xs_sorted[hi]) * w
    return out

def jain_index(vals: List[float]) -> float:
    if not vals:
        return float("nan")
    s = sum(vals)
    s2 = sum(v * v for v in vals)
    n = len(vals)
    if s2 <= 0:
        return float("nan")
    return (s * s) / (n * s2)

def build_stats_from_assign(assign_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    从 --dump-assign 的 JSONL 里重建：
      - 每模型服务次数 served
      - 每模型忙碌时间 busy_time（尽量从 service_time，退化为 finish-start）
      - t_end（所有完成时间的最大值）
      - 等待时间/逗留时间数组
    兼容字段：arrival_time/start/finish/service_time/wait/soj/chosen/(maybe "model"/"chosen_model").
    """
    per_model_served: Dict[str, int] = {}
    per_model_busy: Dict[str, float] = {}
    waits: List[float] = []
    sojs: List[float] = []
    t_end = 0.0

    for r in assign_rows:
        model = r.get("chosen") or r.get("chosen_model") or r.get("model") or r.get("model_name")
        if not model:
            # 如果连模型都没有，就跳过
            continue
        per_model_served[model] = per_model_served.get(model, 0) + 1

        arrival = r.get("arrival_time")
        start = r.get("start")
        finish = r.get("finish")

        service_time = r.get("service_time")
        if service_time is None and start is not None and finish is not None:
            try:
                service_time = float(finish) - float(start)
            except Exception:
                service_time = None
        if service_time is not None:
            per_model_busy[model] = per_model_busy.get(model, 0.0) + float(service_time)

        wait = r.get("wait")
        if wait is None and arrival is not None and start is not None:
            try:
                wait = float(start) - float(arrival)
            except Exception:
                wait = None
        if wait is not None and float(wait) >= 0:
            waits.append(float(wait))

        soj = r.get("soj")
        if soj is None and arrival is not None and finish is not None:
            try:
                soj = float(finish) - float(arrival)
            except Exception:
                soj = None
        if soj is not None and float(soj) >= 0:
            sojs.append(float(soj))

        if finish is not None:
            try:
                t_end = max(t_end, float(finish))
            except Exception:
                pass

    # 计算利用率（需要 t_end>0）
    per_model_util: Dict[str, float] = {}
    if t_end > 0:
        for m, busy in per_model_busy.items():
            per_model_util[m] = busy / t_end

    return {
        "served": per_model_served,
        "busy_time": per_model_busy,
        "util": per_model_util,
        "t_end": t_end,
        "waits": waits,
        "sojs": sojs,
    }

def extract_model_stats_from_summary(summary: Dict[str, Any]) -> Tuple[Dict[str, int], Dict[str, float], Dict[str, float], float]:
    # NOTE: 兼容多种输出结构：
    # 1) 扁平三键：per_model_served / per_model_busy / per_model_util
    # 2) 嵌套单键：model_stats / per_model / models -> {model: {served,busy_time/util}}
    # 3) 字段名别名：busy/busy_time、util/utilization

    """
    从汇总 JSON（sim_run.py 的 --out）里提取模型级指标。
    适配多种可能字段命名：model_stats / per_model / models。
    期望每模型包含 served/busy_time/util（能取到啥就取啥）。
    返回：served_map, busy_map, util_map, t_end
    """
    # 先尝试扁平键
    served: Dict[str, int] = {}
    busy: Dict[str, float] = {}
    util: Dict[str, float] = {}
    t_end = 0.0

    if isinstance(summary.get("per_model_served"), dict):
        for m, v in summary["per_model_served"].items():
            try:
                served[m] = int(v)
            except Exception:
                pass
    if isinstance(summary.get("per_model_busy"), dict):
        for m, v in summary["per_model_busy"].items():
            try:
                busy[m] = float(v)
            except Exception:
                pass
    # 支持 per_model_util 或 per_model_utilization
    util_block = summary.get("per_model_util") or summary.get("per_model_utilization")
    if isinstance(util_block, dict):
        for m, v in util_block.items():
            try:
                util[m] = float(v)
            except Exception:
                pass

    # 若扁平键没取到，再查嵌套块
    if not (served or busy or util):
        candidates = []
        for key in ("model_stats", "per_model", "models"):
            if key in summary and isinstance(summary[key], dict):
                candidates.append(summary[key])
        stats_block: Optional[Dict[str, Any]] = candidates[0] if candidates else None

        if stats_block:
            for m, info in stats_block.items():
                if isinstance(info, dict):
                    if "served" in info:
                        try:
                            served[m] = int(info["served"])
                        except Exception:
                            pass
                    # busy 或 busy_time
                    busy_val = info.get("busy_time", info.get("busy"))
                    if busy_val is not None:
                        try:
                            busy[m] = float(busy_val)
                        except Exception:
                            pass
                    # util 或 utilization
                    util_val = info.get("util", info.get("utilization"))
                    if util_val is not None:
                        try:
                            util[m] = float(util_val)
                        except Exception:
                            pass

    # t_end
    if "t_end" in summary:
        try:
            t_end = float(summary["t_end"])
        except Exception:
            t_end = 0.0

    return served, busy, util, t_end

def merge_stats(pref: Dict[str, Any], suf: Dict[str, Any]) -> Dict[str, Any]:
    """
    把 summary 与 assign 推断出的统计合并：
    - 如果某指标在 summary 缺失，就用 assign 的；
    - 如果两者都有，优先 summary（认为更权威）。
    """
    out: Dict[str, Any] = {}
    for k in ("served", "busy_time", "util"):
        d: Dict[str, Any] = {}
        if k in pref and isinstance(pref[k], dict):
            d.update(pref[k])
        if k in suf and isinstance(suf[k], dict):
            for m, v in suf[k].items():
                if m not in d:
                    d[m] = v
        out[k] = d

    out["t_end"] = pref.get("t_end") or suf.get("t_end") or 0.0
    out["waits"] = pref.get("waits") or suf.get("waits") or []
    out["sojs"] = pref.get("sojs") or suf.get("sojs") or []
    return out

def plot_bar(values: Dict[str, float], title: str, ylabel: str, out_png: Path, show: bool, add_labels: bool=True, palette: str = 'tab20', add_legend: bool=True, legend_loc: str='upper right', show_xticks: bool=True, legend_outside: bool=False, label_inside: bool=False, figsize=(6.0,4.0), ylim_max=None) -> None:
    labels = list(values.keys())
    data = [values[k] for k in labels]
    fig = plt.figure(figsize=figsize)

    # 颜色映射
    try:
        cmap = plt.get_cmap(palette)
    except Exception:
        cmap = plt.get_cmap('tab20')
    n = max(1, len(labels))
    colors = [cmap(i / max(1, n - 1)) for i in range(n)]

    bars = plt.bar(range(len(labels)), data, color=colors)
    if show_xticks:
        plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    else:
        plt.xticks([])
    plt.title(title)
    plt.ylabel(ylabel)

    # 图例：右上角标注颜色对应模型
    if add_legend and len(labels) > 0:
        from matplotlib.patches import Patch
        handles = [Patch(facecolor=colors[i], edgecolor='none', label=labels[i]) for i in range(len(labels))]
        if legend_outside:
            plt.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, frameon=False, fontsize=9)
        else:
            plt.legend(handles=handles, loc=legend_loc, frameon=False, fontsize=9)

    # 给柱子顶部加数值标注
    if add_labels and len(bars) > 0:
        import math
        ymax = max(data) if data else 0.0
        if ymax <= 0:
            ymax = 1.0
        # 若用户指定了 ylim_max，就用更大的那个来设置上限
        if ylim_max is not None:
            ymax_eff = max(ymax, float(ylim_max))
        else:
            ymax_eff = ymax
        plt.ylim(0, ymax_eff * 1.10)
        offset = 0.01 * ymax_eff
        for i, b in enumerate(bars):
            v = data[i]
            if isinstance(v, (int, float)) and math.isfinite(v):
                txt = f"{int(round(v))}" if abs(v - round(v)) < 1e-6 else f"{v:.2f}"
                y = b.get_height() - offset if label_inside else b.get_height() + offset
                va = 'top' if label_inside else 'bottom'
                plt.text(
                    b.get_x() + b.get_width()/2.0,
                    y,
                    txt,
                    ha='center',
                    va=va,
                    fontsize=9,
                    rotation=0,
                    clip_on=False,
                )

    plt.tight_layout()
    fig.savefig(out_png, dpi=150)
    if show:
        plt.show()
    plt.close(fig)

def plot_hist(xs: List[float], bins: int, title: str, xlabel: str, out_png: Path, show: bool) -> None:
    fig = plt.figure()
    plt.hist(xs, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    fig.savefig(out_png, dpi=150)
    if show:
        plt.show()
    plt.close(fig)

def plot_ecdf(xs: List[float], title: str, xlabel: str, out_png: Path, show: bool) -> None:
    if not xs:
        return
    xs_sorted = sorted(xs)
    n = len(xs_sorted)
    ys = [(i + 1) / n for i in range(n)]
    fig = plt.figure()
    plt.plot(xs_sorted, ys)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("ECDF")
    plt.tight_layout()
    fig.savefig(out_png, dpi=150)
    if show:
        plt.show()
    plt.close(fig)

def save_text_summary(out_path: Path, merged: Dict[str, Any]) -> None:
    served = merged.get("served", {})
    busy = merged.get("busy_time", {})
    util = merged.get("util", {})
    waits = merged.get("waits", [])
    sojs = merged.get("sojs", [])
    t_end = float(merged.get("t_end") or 0.0)

    served_list = list(served.values())
    busy_list = list(busy.values())

    lines: List[str] = []
    lines.append("# Simulation Summary")
    lines.append(f"t_end: {t_end:.6f}")
    if served:
        lines.append(f"Models: {len(served)}")
        lines.append(f"Total served: {sum(served_list)}")
        lines.append(f"Jain index (served): {jain_index([float(x) for x in served_list]):.4f}")
    if busy:
        lines.append(f"Jain index (busy_time): {jain_index([float(x) for x in busy_list]):.4f}")

    def fmt_quantiles(xs: List[float], name: str) -> None:
        qs = [0.5, 0.9, 0.95, 0.99]
        qv = safe_quantiles(xs, qs)
        lines.append(f"{name} (n={len(xs)}): " + ", ".join([f"p{int(q*100)}={qv[q]:.6f}" for q in qs]))

    if waits:
        fmt_quantiles(waits, "wait")
    if sojs:
        fmt_quantiles(sojs, "sojourn")

    # Per-model block
    lines.append("")
    lines.append("## Per-Model")
    for m in sorted(served.keys() | busy.keys() | util.keys()):
        s = served.get(m, 0)
        b = busy.get(m, 0.0)
        u = util.get(m, float("nan"))
        lines.append(f"- {m}: served={s}, busy_time={b:.6f}, util={u:.4f}")

    out_path.write_text("\n".join(lines), encoding="utf-8")

def save_csv_per_model(out_path: Path, merged: Dict[str, Any]) -> None:
    served = merged.get("served", {})
    busy = merged.get("busy_time", {})
    util = merged.get("util", {})
    with out_path.open("w", encoding="utf-8") as f:
        f.write("model,served,busy_time,util\n")
        for m in sorted(served.keys() | busy.keys() | util.keys()):
            s = served.get(m, 0)
            b = busy.get(m, 0.0)
            u = util.get(m, float("nan"))
            f.write(f"{m},{s},{b},{u}\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=str, help="sim_run.py 输出的汇总 JSON 路径", default=None)
    ap.add_argument("--assign", type=str, help="--dump-assign 生成的 JSONL 路径（可选）", default=None)
    ap.add_argument("--outdir", type=str, default="sim_plots", help="输出目录（图像/统计文本/CSV）")
    ap.add_argument("--bins", type=int, default=50, help="直方图的分箱数")
    ap.add_argument("--show", action="store_true", help="生成后在本地弹出图窗显示")
    ap.add_argument("--no-bar-labels", action="store_true", help="不在柱状图顶部标注数值")
    ap.add_argument("--palette", type=str, default="tab20", help="柱状图调色板（如 tab20 / Set3 / Accent / hsv 等）")
    ap.add_argument("--no-xtick-labels", action="store_true", help="隐藏底部 x 轴的模型名称（仅用右上角图例标注）")
    ap.add_argument("--legend-loc", type=str, default="upper right", help="图例位置（如 upper right/upper left/lower right/...）")
    ap.add_argument("--legend-outside", action="store_true", help="把图例放在图外右侧，避免与标签遮挡")
    ap.add_argument("--label-inside", action="store_true", help="把柱顶的数字标注放到柱子内部顶部，避免与图例遮挡")
    ap.add_argument("--figw", type=float, default=6.0, help="图宽（英寸）")
    ap.add_argument("--figh", type=float, default=4.0, help="图高（英寸，调小可让柱子看起来更短）")
    ap.add_argument("--ylim-max", type=float, default=None, help="y 轴上限（设得更大，柱子看起来会更短；默认自适应）")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    ensure_outdir(outdir)

    # 1) 读取 summary（若有）
    summary_block: Dict[str, Any] = {}
    if args.summary:
        sp = Path(args.summary)
        if sp.exists():
            try:
                summary_block = read_json(sp)
            except Exception as e:
                print(f"[warn] failed to read summary: {e}")

    served_s, busy_s, util_s, t_end_s = extract_model_stats_from_summary(summary_block)
    pref = {"served": served_s, "busy_time": busy_s, "util": util_s, "t_end": t_end_s}

    # 2) 读取 assign（若有）并构造统计
    suf = {}
    assign_rows: List[Dict[str, Any]] = []
    if args.assign:
        apath = Path(args.assign)
        if apath.exists():
            assign_rows = read_jsonl(apath)
            suf = build_stats_from_assign(assign_rows)
        else:
            print(f"[warn] assign file not found: {apath}")

    # 3) 合并两侧统计
    merged = merge_stats(pref, suf)

    # 4) 画图
    if merged.get("served"):
        plot_bar(
            {m: float(v) for m, v in merged["served"].items()},
            title="Served count per model",
            ylabel="#tasks",
            out_png=outdir / "served_per_model.png",
            show=args.show,
            add_labels=(not args.no_bar_labels), palette=args.palette, add_legend=True, legend_loc=args.legend_loc, show_xticks=(not args.no_xtick_labels), legend_outside=args.legend_outside, label_inside=args.label_inside, figsize=(args.figw, args.figh), ylim_max=args.ylim_max,
        )
    if merged.get("util"):
        plot_bar(
            {m: float(v) for m, v in merged["util"].items()},
            title="Utilization per model",
            ylabel="utilization (busy_time / t_end)",
            out_png=outdir / "util_per_model.png",
            show=args.show,
            add_labels=(not args.no_bar_labels), palette=args.palette, add_legend=True, legend_loc=args.legend_loc, show_xticks=(not args.no_xtick_labels), legend_outside=args.legend_outside, label_inside=args.label_inside, figsize=(args.figw, args.figh), ylim_max=args.ylim_max,
        )
    waits = merged.get("waits", [])
    if waits:
        plot_hist(
            waits,
            bins=args.bins,
            title="Wait time distribution",
            xlabel="wait time",
            out_png=outdir / "wait_hist.png",
            show=args.show,
        )
        plot_ecdf(
            waits,
            title="Wait time ECDF",
            xlabel="wait time",
            out_png=outdir / "wait_ecdf.png",
            show=args.show,
        )
    sojs = merged.get("sojs", [])
    if sojs:
        plot_hist(
            sojs,
            bins=args.bins,
            title="Sojourn time distribution",
            xlabel="sojourn time",
            out_png=outdir / "sojourn_hist.png",
            show=args.show,
        )

    # 5) 文本与 CSV
    save_text_summary(outdir / "summary.txt", merged)
    save_csv_per_model(outdir / "per_model.csv", merged)

    print(f"[done] outputs are in: {outdir.resolve()}")
    print(" - served_per_model.png, util_per_model.png")
    if waits:
        print(" - wait_hist.png, wait_ecdf.png")
    if sojs:
        print(" - sojourn_hist.png")
    print(" - summary.txt, per_model.csv")

if __name__ == "__main__":
    main()
