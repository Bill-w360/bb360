export CUDA_VISIBLE_DEVICES="3"
# --data data/tasks_boolq.jsonl data/tasks_gsm8k.jsonl data/tasks_mmlu.jsonl data/tasks_hellaswag.jsonl \

## 训练标签匹配器
# python -m router.train_label_matcher \
#   --data data/assign_history_synth.jsonl \
#   --dim 5 \
#   --weights checkpoints/label_matcher_bilinear.json \
#   --lr 0.05 --epochs 8 --l2 1e-4  

# # 模拟真实场景下的标签匹配

# python extras/ds_to_tasks.py --root data_local_1 --out data/tasks_unified.jsonl
python scripts/sim_run.py \
  --models models.yaml \
  --tasks data/tasks_unified.jsonl \
  --domain math_reasoning:0,commonsense:1,reading_comprehension:2,general_knowledge:3,humanities:4,social_science:5,stem:6,other_knowledge:7 \
  --k 0.75 --rho-fair 2.0 --lambda-util 0.5 --bp-mode linear --beta 0.5 --gamma 0.0  \
  --service default:expo:1.8 model:qwen2.5-1.5b:const:1.2 pair:llama3-8b,math_reasoning:normal:1.4:0.3 \
  --dump-assign data/assign_from_sim_1.jsonl \
  --out sim_result.json

# 仿真可视化
# python viz/viz_sim.py \
#   --summary sim_result.json \
#   --assign data/assign_from_sim_1.jsonl \
#   --outdir viz_out


# # 合并修补

# python tools/fix_assign.py \
#   --tasks data/tasks_unified.jsonl \
#   --assign data/assign_from_sim_1.jsonl \
#   --out data/assign_fixed.jsonl


# ## 按模型测拆批

# python extras/split_assign_by_model.py \
#   --in data/assign_fixed.jsonl \
#   --out-dir work/batches/

# export HF_HUB_OFFLINE=1
# MODELS=(
#   "../Multiscale_llm_eval/_modelscope/qwen/Qwen2.5-1.5B-Instruct"
#   "../Multiscale_llm_eval/_modelscope/qwen/Qwen2.5-3B-Instruct"
#   "../Multiscale_llm_eval/_modelscope/deepseek-ai/DeepSeek-Coder-1.3B-Instruct"
# )
# DATASETS=(
#   "./work/batches/batch_qwen_Qwen2.5-1.5B-Instruct.jsonl"
#   "./work/batches/batch_qwen_Qwen2.5-3B-Instruct.jsonl"
#   "./work/batches/batch_deepseek-ai_DeepSeek-Coder-1.3B-Instruct.jsonl"
# )
# GPUS=(1 1 1)

#   # --output out/Qwen2.5-3B-Instruct__t_q3_chatletter.jsonl \
#   # --dtype float16 --batch 8 --trust-remote-code \
#   # --chat-wrap auto --mcq-target letter \
#   # --score-mode norm-alpha --alpha 0.7

# mkdir -p out logs
# for i in "${!MODELS[@]}"; do
#   CUDA_VISIBLE_DEVICES="${GPUS[$i]}" TQDM_DISABLE=1 \
#   python scripts/infer_mixed.py \
#     --model "${MODELS[$i]}" \
#     --input "${DATASETS[$i]}" \
#     --output "out/$(basename "${MODELS[$i]}")__$(basename "${DATASETS[$i]}" .jsonl).jsonl" \
#     --dtype float16 --batch 4 --trust-remote-code \
#     --score-mode avg \
#     --chat-wrap auto --mcq-target letter \
#     --math-chat auto --prompt-lang en --fewshot 4 --pot on \
#     --sc-vote 5 --temperature 0.7 --top-p 0.95 \
#     --num-tol 1e-9 --keep-raw --debug-prompt 0 \
#     > "logs/$(basename "${MODELS[$i]}")__$(basename "${DATASETS[$i]}" .jsonl).log" 2>&1 &
# done
# wait



# # 例：三台模型各跑一遍

# # 进程1（GPU0）  Qwen2.5-1.5B-Instruct
# CUDA_VISIBLE_DEVICES=0 python scripts/infer_local.py \
#   --model "../Multiscale_llm_eval/_modelscope/qwen/Qwen2.5-1.5B-Instruct" \
#   --in work/batches/batch_qwen_Qwen2.5-1.5B-Instruct.jsonl \
#   --out work/pred_qwen15b.jsonl \
#   --device cuda:0 --dtype float16 --batch-size 1 --max-new-tokens 256 \
#   --trust-remote-code --local-files-only \
#   > logs/qwen15b_gpu0.log 2>&1 &

# # 进程2（GPU1） Qwen2.5-3B-Instruct
# CUDA_VISIBLE_DEVICES=2 python scripts/infer_local.py \
#   --model "../Multiscale_llm_eval/_modelscope/qwen/Qwen2.5-3B-Instruct" \
#   --in work/batches/batch_qwen_Qwen2.5-3B-Instruct.jsonl \
#   --out work/pred_qwen3b.jsonl \
#   --device cuda:0 --dtype float16 --batch-size 1 --max-new-tokens 256 \
#   --trust-remote-code --local-files-only \
#   > logs/qwen3b_gpu1.log 2>&1 &

# # 进程3（GPU2） DeepSeek-Coder-1.3B-Instruct
# CUDA_VISIBLE_DEVICES=3 python scripts/infer_local.py \
#   --model "../Multiscale_llm_eval/_modelscope/deepseek-ai/DeepSeek-Coder-1.3B-Instruct" \
#   --in work/batches/batch_deepseek-ai_DeepSeek-Coder-1.3B-Instruct.jsonl\
#   --out work/pred_deepseek13b.jsonl \
#   --device cuda:0 --dtype float16 --batch-size 1 --max-new-tokens 256 \
#   --trust-remote-code --local-files-only \
#   > logs/deepseek13b_gpu3.log 2>&1 &

# # 统一评测
# export CUDA_VISIBLE_DEVICES="3"
# python extras/eval_offline.py \
#   --gold data/tasks_unified.jsonl \
#   --pred work/pred_*.jsonl \
#   --out eval_report.json


# # 跑全链路的
# python scripts/route_and_eval.py \
#   --models models.yaml \
#   --tasks tasks_unified.jsonl \
#   --domain math_reasoning:0,commonsense:1,reading_comprehension:2,general_knowledge:3,humanities:4,social_science:5,stem:6,other_knowledge:7 \
#   --k 0.25 --shadow log --alpha 1.0 \
#   --out results.jsonl --report report.txt
