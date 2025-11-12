# diagnose_preds.py
import json, collections, numpy as np, os

PRED = "outputs/mmlu_mtl_fromjson/pred_test_1.jsonl"
LABELS = "outputs/mmlu_mtl_fromjson/label_subjects.json"
TRAIN_JSONL = "/mnt/Data/yangyongbiao/MMLU/validation.jsonl"  # 你训练用的文件路径

# 载入 labels
labels = json.load(open(LABELS, "r", encoding="utf-8"))
idx = {s:i for i,s in enumerate(labels)}

# 统计预测 top-level domain 分布 & per-subject accuracy
cnt_super = collections.Counter()
cnt_sub = collections.Counter()
correct_sub = collections.Counter()
confusion = {}

total_with_gt = 0

with open(PRED, "r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        pred_s = r["_pred_subject"]
        pred_sup = r["_pred_super"]
        cnt_super[pred_sup] += 1
        cnt_sub[pred_s] += 1

        gt = r.get("subject")
        if gt:
            total_with_gt += 1
            if gt == pred_s:
                correct_sub[gt] += 1
            # confusion
            confusion.setdefault(gt, collections.Counter())[pred_s] += 1

print("=== Top predicted super domains ===")
for k,v in cnt_super.most_common(20):
    print(k, v)
print()
print("=== Top predicted subjects (most frequent preds) ===")
for k,v in cnt_sub.most_common(20):
    print(k, v)
print()
print("=== Overall GT count (with subject) ===", total_with_gt)
print("=== Per-subject accuracy (sample top 10 by GT count) ===")
# compute gt counts in dataset (from pred file)
gt_counts = collections.Counter()
with open(PRED,"r",encoding="utf-8") as f:
    for line in f:
        r=json.loads(line)
        if r.get("subject"):
            gt_counts[r["subject"]]+=1

most_common_gt = gt_counts.most_common(10)
for s,c in most_common_gt:
    acc = correct_sub[s]/c if c>0 else 0.0
    print(s, "GT_count=", c, "acc=", acc)

# Print top confusion pairs
print()
print("=== Top confusion pairs (GT -> Pred) overall ===")
pairs = []
for gt,ctr in confusion.items():
    for pred,cc in ctr.items():
        pairs.append((cc, gt, pred))
pairs.sort(reverse=True)
for cc,gt,pred in pairs[:30]:
    print(f"{gt} -> {pred} : {cc}")
