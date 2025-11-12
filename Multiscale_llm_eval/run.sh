export CUDA_VISIBLE_DEVICES="3"


# 设置环境变量加速下载
export HF_HOME=./
export HF_DATASETS_CACHE=./datasets
export HF_HUB_OFFLINE=0
export HF_DATASETS_OFFLINE=0


# —— Hugging Face 国内镜像 + 缓存根目录 ——
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/mnt/Data/yangyongbiao/.cache/huggingface

# —— ModelScope 镜像 + 缓存根目录 ——
export MODELSCOPE_HUB_MIRROR=https://modelscope.cn
export MODELSCOPE_CACHE=/mnt/Data/yangyongbiao/.cache/modelscope

# --models qwen/Qwen2.5-3B-Instruct deepseek-ai/DeepSeek-Coder-1.3B-Instruct qwen/Qwen2.5-1.5B-Instruct \
#  --tasks gsm8k,hellaswag,boolq,mmlu \

python tools/modelscope_eval_pipeline.py \
  --models deepseek-ai/DeepSeek-vl-1.3B-Chat \
  --tasks gsm8k,hellaswag,boolq,mmlu \
  --device cuda \
  --dtype float16 \
  --batch-size 1 \
  --skip-plots 1 \
  --no-aggregate 0 \
  --hf-cache "$HF_HOME/datasets_eval" \
  --gen-kwargs max_new_tokens=256,temperature=0.0 \
  --task-gen-kwargs '{"gsm8k":{"max_new_tokens":512}}' \
  --results-dir ./eval_results \
  --out-dir ./eval_out_1
   