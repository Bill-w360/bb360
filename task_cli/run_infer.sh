#!/usr/bin/env bash
set -euo pipefail

# 一键 Ctrl-C 结束所有子进程
trap 'echo; echo "[main] stopping..."; kill 0' INT TERM

# ===== 环境（按需修改） =====
export HF_HOME=/mnt/Data/yangyongbiao/.cache/huggingface
export HF_DATASETS_CACHE=/mnt/Data/yangyongbiao/.cache/huggingface/datasets_eval
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1                       # 让 python 实时输出
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128


QWEN_15B_DIR="../Multiscale_llm_eval/_modelscope/qwen/Qwen2.5-1.5B-Instruct"
QWEN_3B_DIR="../Multiscale_llm_eval/_modelscope/qwen/Qwen2.5-3B-Instruct"
DEEPSEEK_DIR="../Multiscale_llm_eval/_modelscope/deepseek-ai/DeepSeek-Coder-1.3B-Instruct"

# 公共参数（你的 infer_local 已支持）
COMMON="--dtype float16 --batch-size 16 --max-new-tokens 128 --trust-remote-code --local-files-only --device cuda:0"

# 带前缀运行的封装：每行输出自动加 [TAG]
run_tag () {
  local TAG="$1"; shift
  ( set -o pipefail; stdbuf -oL -eL "$@" 2>&1 | sed -u "s/^/[$TAG] /" ) &
}

# ===== 三进程并行（每进程只看见一张卡）=====
# GPU0: Qwen2.5-1.5B
CUDA_VISIBLE_DEVICES=1 run_tag QWEN15B \
  python -u scripts/infer_local.py \
    --model "$QWEN_15B_DIR" \
    --in work/batches/batch_qwen_Qwen2.5-1.5B-Instruct.jsonl \
    --out work/pred_qwen15b.jsonl \
    $COMMON

# GPU1: Qwen2.5-3B
CUDA_VISIBLE_DEVICES=3 run_tag QWEN3B \
  python -u scripts/infer_local.py \
    --model "$QWEN_3B_DIR" \
    --in work/batches/batch_qwen_Qwen2.5-3B-Instruct.jsonl \
    --out work/pred_qwen3b.jsonl \
    $COMMON

# GPU2: DeepSeek-Coder-1.3B
CUDA_VISIBLE_DEVICES=1 run_tag DEEPSEEK13B \
  python -u scripts/infer_local.py \
    --model "$DEEPSEEK_DIR" \
    --in work/batches/batch_deepseek-ai_DeepSeek-Coder-1.3B-Instruct.jsonl \
    --out work/pred_deepseek13b.jsonl \
    $COMMON

wait
echo "[main] all done."
