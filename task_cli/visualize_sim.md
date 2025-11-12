# 可视化脚本使用说明

**文件**：`visualize_sim.py`  
**用途**：对 `sim_run.py` 产生的结果进行可视化与统计汇总。

## 输入
- `--summary`：`sim_run.py` 运行时通过 `--out` 保存的汇总 JSON（可选）。
- `--assign`：`sim_run.py` 运行时通过 `--dump-assign` 生成的明细 JSONL（可选）。
  - 若只给了 `--assign`，脚本会从明细重建每模型的接单量、忙碌时长、利用率等；
  - 若两者都给，默认以 `--summary` 为主、`--assign` 为补充。

## 输出
在 `--outdir`（默认 `sim_plots/`）下生成：
- `served_per_model.png`：每个模型的接单数量柱状图；
- `util_per_model.png`：每个模型的利用率柱状图（busy_time / t_end）；
- `wait_hist.png` 与 `wait_ecdf.png`：等待时间直方图与 ECDF（若明细含有可恢复的等待时间）；
- `sojourn_hist.png`：逗留时间直方图（若可恢复）；
- `summary.txt`：整体统计（Jain 指数、分位数等）；
- `per_model.csv`：每模型统计表。

## 运行示例
```bash
# 仅使用汇总 JSON
python viz/visualize_sim.py --summary sim_result.json --outdir viz/viz_out

# 仅使用分配明细（来自 --dump-assign）
python viz/visualize_sim.py --assign data/assign_from_sim_1.jsonl --outdir viz/viz_out

# 两者都用，并弹出图窗
python viz/visualize_sim.py --summary sim_result.json --assign data/assign_from_sim_1.jsonl --outdir  viz/viz_out 
```

## 备注
- 若你的明细 JSONL 没有显式的 `wait/soj/service_time` 字段，脚本会尝试用
  `start/finish/arrival_time` 推回去（若字段缺失，某些图可能为空）。
- 该脚本不依赖 seaborn，仅使用 matplotlib；每个图单独一张图片，便于汇报文档引用。
- 若你在远程服务器上运行且没有显示环境，不要加 `--show`，直接取生成的 PNG/CSV。

祝实验顺利！
