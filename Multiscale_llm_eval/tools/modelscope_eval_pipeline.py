# -*- coding: utf-8 -*-
# ModelScope 下载(可选) → lm-eval(一次性评测多个任务) → 仅准确率汇总 →（可选）全局聚合/分类
# 依赖：
#   pip install modelscope "huggingface_hub[hf_transfer]" transformers datasets accelerate sentencepiece
#   pip install lm-eval==0.4.3
#   tools/lm_eval_to_scores.py 必须存在

from __future__ import annotations
import argparse, csv, json, os, re, shlex, subprocess, sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

# ---------------------------
# 控制台与小工具
# ---------------------------
def log(msg: str): print(msg, flush=True)
def run(cmd: List[str], cwd: Path|None=None, env: dict|None=None) -> int:
    log(f"[run] {' '.join(shlex.quote(str(x)) for x in cmd)}")
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env)
    return proc.returncode
def safe_name(s: str) -> str: return re.sub(r"[^\w.\-]", "_", s)
def is_instruct_or_chat(model_id: str) -> bool: return bool(re.search(r"(instruct|chat)", model_id, re.I))

def json_default(obj):
    try:
        import numpy as _np
        if isinstance(obj, _np.dtype): return str(obj)
        if isinstance(obj, _np.generic): return obj.item()
        if isinstance(obj, _np.ndarray): return obj.tolist()
    except Exception: pass
    try:
        import torch as _torch
        if isinstance(obj, _torch.dtype): return str(obj)
        if isinstance(obj, _torch.device): return str(obj)
        if isinstance(obj, _torch.Tensor):
            return {"__torch_tensor__": True, "shape": list(obj.shape), "dtype": str(obj.dtype)}
    except Exception: pass
    try: return str(obj)
    except Exception: pass
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

def parse_kv(s: str) -> Dict[str, Any]:
    if not s: return {}
    out: Dict[str, Any] = {}
    for seg in s.split(","):
        seg = seg.strip()
        if not seg: continue
        if "=" not in seg: raise ValueError(f"--gen-kwargs 项格式错误：{seg}（应为 key=value）")
        k, v = seg.split("=", 1); k, v = k.strip(), v.strip()
        low = v.lower()
        if low in ("true","false"): out[k] = (low=="true")
        else:
            try:
                out[k] = int(v) if re.fullmatch(r"[+-]?\d+", v) else float(v)
            except Exception:
                out[k] = v
    return out

# ---------------------------
# ModelScope 下载
# ---------------------------
def ms_snapshot_download(ms_id: str, cache_dir: Path) -> Path:
    from modelscope.hub.snapshot_download import snapshot_download
    path = snapshot_download(ms_id, cache_dir=str(cache_dir))
    return Path(path)

# ---------------------------
# 一次性评测多个任务（进度条与日志“一步到位”收敛）
# ---------------------------
def eval_one_model(
    model_id: str,
    local_path: Path,
    tasks: List[str],
    results_dir: Path,
    device: str,
    dtype: str,
    batch_size: str,
    limit: int|None,
    global_gen_kwargs: Dict[str, Any],
    per_task_gen: Dict[str, Dict[str, Any]],
) -> Path|None:
    """
    对该模型一次性评测所有任务；禁用内部进度条、降低噪声日志；清理无效/冲突生成参数。
    """
    from lm_eval import evaluator

    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = results_dir / "results.json"

    model_args_str = ",".join([
        f"pretrained={str(local_path)}",
        f"dtype={dtype}",
        f"device={device}",
        "trust_remote_code=true",
    ])
    apply_tmpl = is_instruct_or_chat(model_id)

    if per_task_gen:
        log("[warn] 一次性评测不支持 per-task 生成参数 (--task-gen-kwargs)。将忽略该参数。")

    # ——生成参数：先拷贝
    gk = dict(global_gen_kwargs)

    # ——一步到位：收敛进度条与日志——
    # 彻底关闭 tqdm 刷屏（lm-eval 与 transformers 内部）
    os.environ["TQDM_DISABLE"] = "1"
    os.environ.setdefault("TQDM_MININTERVAL", "5")  # 给备选路径留面子
    os.environ.setdefault("LM_EVAL_LOG_LEVEL", "ERROR")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    import logging as _logging
    _logging.getLogger("lm_eval").setLevel(_logging.ERROR)
    _logging.getLogger("transformers").setLevel(_logging.ERROR)

    # ——避免冲突与无效提示——
    # 1) 不要同时给 max_length 和 max_new_tokens（保留后者）
    gk.pop("max_length", None)

    # 2) 判别/选择题任务（loglikelihood 路径）会忽略抽样参数；清理它们以免提示
    MC_TASKS = {"hellaswag", "boolq", "mmlu"}
    if any(t.split(":")[0].lower() in MC_TASKS for t in tasks):
        for k in ("temperature", "top_p", "top_k", "do_sample", "top_k_sampling"):
            gk.pop(k, None)

    # 3) 生成类任务（如 GSM8K）给足 tokens（一次性评测无法 per-task 覆写）
    if any(t.lower() in ("gsm8k","svamp","math","asdiv","mawps") for t in tasks):
        gk.setdefault("max_new_tokens", 256)

    log(f"[lm-eval] model_args={model_args_str}")
    log(f"[lm-eval] tasks={tasks}  batch_size={batch_size}  limit={limit}  apply_chat_template={apply_tmpl}")
    log(f"[lm-eval] gen_kwargs={gk}")

    # 可选预热（internlm 家族）
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if any(k in model_id.lower() for k in ["internlm2", "internlm"]):
            tok = AutoTokenizer.from_pretrained(str(local_path), trust_remote_code=True)
            mdl = AutoModelForCausalLM.from_pretrained(
                str(local_path),
                trust_remote_code=True,
                device_map={"": device},
            )
            if hasattr(mdl.config, "use_cache"): mdl.config.use_cache = True
            if hasattr(mdl, "generation_config"): mdl.generation_config.use_cache = True
            del mdl, tok
    except Exception as e:
        log(f"[warmup] 跳过可选预热：{e}")

    # ——一次性评测多个任务；压低 lm-eval 自身 verbosity——
    res = evaluator.simple_evaluate(
        model="hf",
        model_args=model_args_str,
        tasks=tasks,
        batch_size=batch_size,
        limit=limit if (limit and limit>0) else None,
        apply_chat_template=apply_tmpl,
        fewshot_as_multiturn=apply_tmpl,
        gen_kwargs=gk if gk else None,
        verbosity="ERROR",  # 原为 "INFO"
    )

    with out_file.open("w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2, default=json_default)
    size = out_file.stat().st_size if out_file.exists() else 0
    if size > 64:
        log(f"[ok] results.json 写入成功：{out_file}  ({size} bytes)")
        return out_file
    log("[err] results.json 未成功生成或内容过小。")
    return None

# ---------------------------
# 将 results.json 转成临时 CSV（复用你的 tools/lm_eval_to_scores.py）
# ---------------------------
def convert_results_to_tmp_csv(results_json: Path, model_id: str, tmp_csv: Path, project_root: Path) -> bool:
    tmp_csv.parent.mkdir(parents=True, exist_ok=True)
    conv = [sys.executable, str(project_root / "tools" / "lm_eval_to_scores.py"),
            str(results_json), model_id, str(tmp_csv)]
    code = run(conv)
    if code != 0 or (not tmp_csv.exists()):
        return False
    try:
        with tmp_csv.open("r", encoding="utf-8") as f:
            lines = [ln for ln in f if ln.strip()]
        return len(lines) >= 2  # header + 至少一行
    except Exception:
        return False

# ---------------------------
# 仅保留“准确率”写入目标 CSV（签名对齐 main 的调用）
# ---------------------------
def append_scores(tmp_csv: Path, dest_csv: Path, model_id_forced: str) -> int:
    """
    从临时 CSV 里筛出 “准确率类指标”，但每个 dataset 只保留 1 行：
      - 默认优先级：acc > accuracy > acc_norm > exact_match > em > strict_match > flexible_extract > correct > pass@1
      - 针对特定任务（如 gsm8k）覆写优先级：exact_match > strict_match > flexible_extract > acc > accuracy > acc_norm
      - 忽略统计量：*_stderr/_std/_se/_var
      - 兼容百分号数值；自动写表头
    """
    def write_header_if_needed(path: Path):
        if not path.exists() or path.stat().st_size == 0:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8", newline="") as f:
                csv.writer(f).writerow(["model","dataset","metric","value"])

    def find_idx(header:list[str], cands:list[str]) -> int|None:
        mp = {header[i].strip().lower(): i for i in range(len(header))}
        for c in cands:
            if c in mp: return mp[c]
        for k, i in mp.items():
            if any(c in k for c in cands):
                return i
        return None

    def clean_metric(m: str) -> str:
        x = str(m).strip().lower()
        # 1) 先去掉括号备注
        x = re.sub(r"\(.*?\)", "", x)
        # 2) **关键修正**：逗号后通常是分组/聚合标签（如 ",none", ",macro"），只取逗号前的真正指标名
        if "," in x:
            x = x.split(",", 1)[0]
        # 3) 统一空白/连字符
        x = x.replace(" ", "").replace("-", "_")
        # 4) 去掉路径式/层级式后缀（很少见，但保留兼容）
        for sep in [":", "/", "\\", ">"]:
            if sep in x:
                x = x.split(sep, 1)[0]   # 注意这里也取“前半段”
        # 5) 去掉统计量后缀
        x = re.sub(r"_(stderr|std|se|var)$", "", x)
        x = re.sub(r"(stderr|std|se|var)$", "", x)
        return x


    # 识别哪些是“像准确率的名字”
    ACC_ALIASES = {
        "acc", "acc_norm", "accuracy", "normalized_accuracy",
        "exact_match", "em", "match", "correct",
        "strict_match", "flexible_extract",
        "pass@1", "pass_1", "pass1",
    }
    def is_accuracy_like(name: str) -> bool:
        x = clean_metric(name)
        if x in ACC_ALIASES:
            return True
        if ("acc" in x) or ("accuracy" in x):
            return True
        if x.startswith("pass") and ("1" in x):
            return True
        return False

    # 默认优先级 & 特定数据集优先级
    DEFAULT_ORDER = ["acc","accuracy","acc_norm","exact_match","em","strict_match","flexible_extract","correct","pass@1","pass_1","pass1"]
    DATASET_ORDER = {
        # GSM8K 更关注“是否完全答对”，优先 exact_match
        "gsm8k": ["flexible_extract", "exact_match", "strict_match", "acc", "accuracy", "acc_norm"],
    }
    def metric_rank(ds: str, m: str) -> int:
        m = clean_metric(m)
        order = None
        # dataset 名可能带子集，取主名匹配（例如 "mmlu/stem" → "mmlu"）
        ds_main = ds.split("/")[0].lower() if ds else ""
        if ds_main in DATASET_ORDER:
            order = DATASET_ORDER[ds_main]
        else:
            order = DEFAULT_ORDER
        try:
            return order.index(m)
        except ValueError:
            # 未在表中：仍允许包含 acc/accuracy 的排在靠后
            if ("acc" in m) or ("accuracy" in m): return len(order) - 1
            return 10_000

    def to_float(v: str) -> float | None:
        s = str(v).strip()
        try:
            if s.endswith("%"):
                return float(s[:-1])/100.0  # 若想统一到 0~1，换成 /100.0
            return float(s)
        except Exception:
            return None

    if not tmp_csv.exists():
        log(f"[warn] 临时CSV不存在：{tmp_csv}")
        return 0

    # 1) 读入并筛选“准确率类”候选
    with tmp_csv.open("r", encoding="utf-8") as f_in:
        rdr = csv.reader(f_in)
        header = next(rdr, None)
        if not header:
            log(f"[warn] 临时CSV为空：{tmp_csv}")
            return 0

        metidx = find_idx(header, ["metric", "metric_name", "name", "key", "measure"])
        vidx   = find_idx(header, ["value", "val", "score", "mean"])
        didx   = find_idx(header, ["dataset", "task", "bench", "benchmark", "suite"])
        subidx = find_idx(header, ["subset", "category", "subtask", "domain"])

        if (metidx is None) or (vidx is None):
            log(f"[warn] 临时CSV缺少 metric/value 列：{header}")
            return 0

        # 候选池：dataset -> [(metric_name, value)]
        pool: Dict[str, List[Tuple[str, float]]] = {}

        for row in rdr:
            if not row or max(metidx, vidx) >= len(row):
                continue
            mraw = row[metidx]
            m = clean_metric(mraw)
            if m in {"", "none", "nan"}:
                continue
            if not is_accuracy_like(m):
                continue

            # dataset 名
            ds = ""
            if didx is not None and didx < len(row):
                ds = (row[didx] or "").strip()
            if (not ds) and subidx is not None and subidx < len(row):
                ds = (row[subidx] or "").strip()
            if not ds:
                ds = "unknown_task"

            # 数值
            v = to_float(row[vidx])
            if v is None:
                continue

            pool.setdefault(ds, []).append((m, v))

    if not pool:
        return 0

    # 2) 对每个 dataset，按优先级挑 1 个；若同名多值取“最大值”；若存在非零值则优先非零
    write_header_if_needed(dest_csv)
    written = 0
    with dest_csv.open("a", encoding="utf-8", newline="") as f_out:
        w = csv.writer(f_out)
        for ds, items in pool.items():
            # 先按 (rank, -value) 排序；这样优先级高、数值大的在前
            items_sorted = sorted(items, key=lambda kv: (metric_rank(ds, kv[0]), -kv[1]))
            # 如果前面是 0 且后面有非 0，同优先级时会被 -value 推到前头；不同优先级时仍按优先级
            best_metric, best_value = items_sorted[0]
            w.writerow([model_id_forced, ds, "acc", str(best_value)])
            written += 1

    return written


# ---------------------------
#（可选）全局分类
# ---------------------------
def run_classify(scores_csv: Path, meta_csv: Path, config_yaml: Path, clusters: int, out_dir: Path, skip_plots: int=0):
    state_file = out_dir / "multiscale_llm_evaluation.json"
    if state_file.exists():
        try: state_file.unlink()
        except Exception: pass

    try:
        n_models = len({r.split(",")[0] for r in scores_csv.read_text(encoding="utf-8").splitlines()[1:] if r.strip()})
        k = max(2, min(int(clusters), n_models)) if n_models >= 2 else 2
    except Exception:
        k = clusters

    cmd = [
        sys.executable, "-m", "multiscale_llm_eval.cli", "classify",
        "--format", "csv",
        "--input", str(scores_csv),
        "--meta", str(meta_csv) if meta_csv.exists() else "",
        "--config", str(config_yaml) if config_yaml.exists() else "",
        "--clusters", str(k),
        "--output-dir", str(out_dir),
        "--skip-plots", str(skip_plots),
    ]
    cmd = [c for c in cmd if c != ""]
    code = run(cmd)
    if code == 0:
        log(f"[classify] 完成：{out_dir / 'classification_results.csv'}")
    else:
        log("[classify] 失败，请检查 scores.csv / config.yaml / meta.csv")

# ---------------------------
# 主流程
# ---------------------------
def main():
    ap = argparse.ArgumentParser("ModelScope+lm-eval（一次性多任务）→ 仅准确率 →（可选）全局聚合/分类")
    ap.add_argument("--models", nargs="+", required=True, help="ModelScope 模型ID或本地路径，例：qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--tasks", default="gsm8k,hellaswag,boolq,mmlu", help="任务逗号分隔")
    ap.add_argument("--cache", default="./_modelscope", help="ModelScope 缓存目录")
    ap.add_argument("--results-dir", default="./results_modelscope", help="每模型评测输出根目录")
    ap.add_argument("--out-dir", default="./out", help="（聚合模式）最终输出目录")

    ap.add_argument("--device", default="cuda", choices=["cuda","cpu"])
    ap.add_argument("--dtype", default="float16", choices=["float16","bfloat16","float32"])
    ap.add_argument("--batch-size", default="1")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--clusters", type=int, default=5)
    ap.add_argument("--meta", default="./meta.csv")
    ap.add_argument("--config", default="./config.yaml")
    ap.add_argument("--skip-plots", type=int, default=0)

    ap.add_argument("--hf-cache", type=str, default=None, help="HF datasets 缓存目录")
    ap.add_argument("--offline", type=int, default=0, help="1=严格离线")
    ap.add_argument("--skip-download", type=int, default=0, help="1=不从 ModelScope 拉，直接用本地")
    ap.add_argument("--gen-kwargs", type=str, default="", help="全局生成参数，如: max_new_tokens=64,temperature=0.0")
    ap.add_argument("--task-gen-kwargs", type=str, default="", help="按任务覆写（一次性评测会忽略）")
    ap.add_argument("--no-aggregate", type=int, default=0, help="1=剔除聚合：每个模型各自输出 scores_acc.csv，不产出全局 scores.csv / 不分类")

    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    cache_dir    = (project_root / args.cache).resolve()
    results_root = (project_root / args.results_dir).resolve()
    out_dir      = (project_root / args.out_dir).resolve()
    meta_csv     = (project_root / args.meta).resolve()
    config_yaml  = (project_root / args.config).resolve()

    # 环境与离线
    os.environ.setdefault("HUGGINGFACE_HUB_HTTP_TIMEOUT", "60")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("MS_HUB_DISABLE_TQDM", "1")
    if args.hf_cache:
        os.environ["HF_DATASETS_CACHE"] = str(Path(args.hf_cache).resolve())
        log(f"[env] HF_DATASETS_CACHE = {os.environ['HF_DATASETS_CACHE']}")
    if int(args.offline) == 1:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        log("[env] 严格离线：HF_DATASETS_OFFLINE/TRANSFORMERS_OFFLINE/HF_HUB_OFFLINE = 1")

    cache_dir.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    global_gen = parse_kv(args.gen_kwargs)
    try:
        per_task_gen = json.loads(args.task_gen_kwargs) if args.task_gen_kwargs else {}
    except Exception as e:
        log(f"[warn] --task-gen-kwargs 不是合法JSON，将忽略。错误：{e}")
        per_task_gen = {}

    failures: List[Tuple[str, str]] = []
    appended_total = 0

    # 若启用聚合模式，先初始化全局 scores.csv
    global_scores_csv = out_dir / "scores.csv"
    if int(args.no_aggregate) == 0:
        global_scores_csv.parent.mkdir(parents=True, exist_ok=True)
        with global_scores_csv.open("w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(["model","dataset","metric","value"])

    for ms_id in args.models:
        # 解析本地/下载
        if (Path(ms_id).exists() and Path(ms_id).is_dir()):
            local = Path(ms_id).resolve()
            log(f"[local] 使用本地模型目录：{local}")
        else:
            if int(args.skip_download) == 1:
                maybe_local = cache_dir / ms_id.replace("/", os.sep)
                if maybe_local.exists():
                    local = maybe_local
                    log(f"[skip] 跳过下载，使用本地：{local}")
                else:
                    log(f"[warn] 未找到本地：{maybe_local}，尝试下载。")
                    try:
                        local = ms_snapshot_download(ms_id, cache_dir=cache_dir)
                        log(f"[download] {ms_id} -> {local}")
                    except Exception as e:
                        log(f"[error] 下载失败：{ms_id} | {e}")
                        failures.append((ms_id, "download"))
                        continue
            else:
                log(f"[download] {ms_id}")
                try:
                    local = ms_snapshot_download(ms_id, cache_dir=cache_dir)
                    log(f"           -> {local}")
                except Exception as e:
                    log(f"[error] 下载失败：{ms_id} | {e}")
                    failures.append((ms_id, "download"))
                    continue

        # 一次性评测
        log(f"[eval] {ms_id}")
        res_dir = results_root / safe_name(ms_id)
        res_json = eval_one_model(
            model_id=ms_id,
            local_path=local,
            tasks=tasks,
            results_dir=res_dir,
            device=args.device,
            dtype=args.dtype,
            batch_size=args.batch_size,
            limit=None if (args.limit is None or args.limit<=0) else args.limit,
            global_gen_kwargs=global_gen,
            per_task_gen=per_task_gen,
        )
        if not res_json:
            log(f"[warn] 评测未生成 results.json：{ms_id}")
            failures.append((ms_id, "eval"))
            continue

        # 转换为临时CSV
        tmp_csv = res_dir / "_tmp_scores.csv"
        if not convert_results_to_tmp_csv(res_json, ms_id, tmp_csv, project_root):
            log(f"[warn] 转换临时CSV失败：{ms_id}")
            failures.append((ms_id, "convert"))
            continue

        if int(args.no_aggregate) == 1:
            # ——每模型单独导出——
            per_model_csv = res_dir / "scores_acc.csv"
            n = append_scores(tmp_csv, per_model_csv, model_id_forced=ms_id)
            log(f"[model-csv] {ms_id} → {per_model_csv}（写入{n}行，仅准确率）")
            appended_total += n
            try: tmp_csv.unlink(missing_ok=True)
            except Exception: pass
        else:
            # ——聚合到全局——
            n = append_scores(tmp_csv, global_scores_csv, model_id_forced=ms_id)
            log(f"[append] {ms_id} 仅准确率 {n} 行已追加到 {global_scores_csv}")
            appended_total += n
            try: tmp_csv.unlink(missing_ok=True)
            except Exception: pass

    # 汇总/分类
    if int(args.no_aggregate) == 1:
        if appended_total > 0:
            log(f"[summary] 已为各模型分别生成仅准确率的 scores_acc.csv（共写入 {appended_total} 行）。不做全局分类。")
        else:
            log("[summary] 未产生有效分数（准确率）。")
    else:
        if appended_total > 0:
            log(f"[summary] 已汇总 {appended_total} 条仅准确率分数 → {global_scores_csv}")
            run_classify(global_scores_csv, meta_csv, config_yaml, args.clusters, out_dir, args.skip_plots)
        else:
            log("[summary] 未产生有效分数（准确率），跳过分类。")

    if failures:
        log("[summary] 失败清单：")
        for mid, stage in failures:
            log(f"  - {mid} @ {stage}")

if __name__ == "__main__":
    main()
