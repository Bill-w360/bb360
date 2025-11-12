export CUDA_VISIBLE_DEVICES="1"

# train
# export HF_HUB_OFFLINE=1
# python scripts/train_task_classifier.py \
#   --train-json ./data/mmlu_out/mmlu_out_train.jsonl \
#   --eval-json ./data/mmlu_out/mmlu_out_validation.jsonl \
#   --mapping ./data/mappings/subject_to_super.json \
#   --super2id ./data/mappings/super_to_id.json \
#   --pretrained ../models/roberta-base \
#   --alpha 0.95 \
#   --max-length 384 \
#   --epochs 8 \
#   --batch-size 16 \
#   --grad-accum 2 \
#   --lr 2e-5 \
#   --outdir ./outputs/mmlu_mtl_fromjson \
#   --offline



# infer

# python scripts/infer_task_classifier.py \
#   --model-dir ./outputs/mmlu_mtl_fromjson \
#   --test-json ./data/mmlu_out/mmlu_out_test.jsonl \
#   --mapping ./data/mappings/subject_to_super.json \
#   --super2id ./data/mappings/super_to_id.json \
#   --max-length 384 \
#   --batch-size 64 \
#   --offline

# mmlu-pro
# # 1) 规范化（保留 subject 字段）
# python prep_mmlu_pro_jsonl.py \
#   --infile data/mmlu-pro/mmlu_pro_test.jsonl \
#   --out data/mmlu-pro/test_val.jsonl \
#   --keep-gold

# # 2) 自动造 8 类“银标”（无需手写映射）
# python auto_sup_labeler.py \
#   --in data/mmlu-pro/test_val.jsonl \
#   --out data/mmlu-pro/test_val.auto.jsonl

# # 3) 你的分类器推理
# python scripts/infer_task_classifier.py \
#   --model-dir outputs/mmlu_mtl_fromjson \
#   --test-json data/mmlu-pro/test_val.jsonl \
#   --out outputs/results/mmlu_pro/preds_test.jsonl --offline

# # 4) 严格逐行对齐打分（用自动银标做参考）
# python score_sup_accuracy_gold.py \
#   --pred outputs/results/mmlu_pro/preds_test.jsonl \
#   --jsonl data/mmlu-pro/test_val.auto.jsonl

# python audit_label_gap.py \
#   --pred outputs/results/mmlu_pro/preds_test.jsonl \
#   --jsonl data/mmlu-pro/test_val.auto.jsonl \
#   --silver-field label_sup \
#   --out outputs/results/audit_mmlu_pro



# # 测试boolq、gsm8k、hellaswag
# # BoolQ → 目标上位域 reading_comprehension
# #测
# python scripts/infer_task_classifier.py \
#   --model-dir outputs/mmlu_mtl_fromjson \
#   --test-json ./data/boolq/validation.jsonl \
#   --out preds_boolq.jsonl --offline
# #评
# python score_sup_accuracy.py \
#   --pred preds_boolq.jsonl --target reading_comprehension --reject 0.5

# # GSM8K → math_reasoning
# #测
# python scripts/infer_task_classifier.py \
#   --model-dir outputs/mmlu_mtl_fromjson \
#   --test-json ./data/gsm8k/test.jsonl \
#   --out preds_gsm8k.jsonl --offline
# #评
# python score_sup_accuracy.py \
#   --pred preds_gsm8k.jsonl --target math_reasoning --reject 0.5

# # HellaSwag → commonsense
# # #测
# python scripts/infer_task_classifier.py \
#   --model-dir outputs/mmlu_mtl_fromjson \
#   --test-json ./data/hellaswag/test.jsonl \
#   --out preds_hellaswag.jsonl --offline
# #评
# python score_sup_accuracy.py \
#   --pred preds_hellaswag.jsonl --target commonsense --reject 0.5

# 批量跑
# python run_sup_eval_local.py \
#   --data-root data \
#   --model-dir outputs/mmlu_mtl_fromjson \
#   --out-dir eval_out \
#   --tau 0.5


# python fix_heads_by_mapping.py \
#   --model_dir outputs/mmlu_mtl_fromjson \
#   --pred_file outputs/mmlu_mtl_fromjson/pred_test_1.jsonl \
#   --pred_remapped outputs/mmlu_mtl_fromjson/pred_test_1_remapped.jsonl
