# -*- coding: utf-8 -*-
import json
import collections
import numpy as np
import matplotlib.pyplot as plt

# ====== 路径按你的工程结构设置 ======
PRED_PATH = "outputs/mmlu_mtl_fromjson/pred_test_1.jsonl"
SUBJECT_TO_SUPER_PATH = "data/mappings/subject_to_super.json"

# ====== 1. 读取 subject -> super 映射 ======
with open(SUBJECT_TO_SUPER_PATH, "r", encoding="utf-8") as f:
    subject_to_super = json.load(f)

# ====== 2. 统计每个上位域的总样本数 / 正确数 / 预测分布 / 混淆 ======
total_by_sup = collections.Counter()     # 每个 GT super 的样本数
correct_by_sup = collections.Counter()   # 每个 GT super 中预测正确的数量
pred_dist = collections.Counter()        # 每个 predicted super 的数量

# 为了画 8x8 混淆矩阵，先统计所有出现的 super 名称
all_sup_names = set()

# 第一遍先扫一遍，把所有 super 收集起来
with open(PRED_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        subj = rec.get("subject")
        pred_sup = rec.get("_pred_super")
        if subj is None or pred_sup is None:
            continue
        gt_sup = subject_to_super.get(subj)
        if gt_sup is None:
            continue
        all_sup_names.add(gt_sup)
        all_sup_names.add(pred_sup)

# 统一一个顺序，方便画图
super_names = sorted(all_sup_names)
sup_to_idx = {name: i for i, name in enumerate(super_names)}
num_sup = len(super_names)

# 初始化混淆矩阵：rows=GT, cols=Pred
conf_mat = np.zeros((num_sup, num_sup), dtype=np.int64)

# 第二遍正式统计
with open(PRED_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        subj = rec.get("subject")
        pred_sup = rec.get("_pred_super")
        if subj is None or pred_sup is None:
            continue
        gt_sup = subject_to_super.get(subj)
        if gt_sup is None:
            continue

        # 总数 + 正确数
        total_by_sup[gt_sup] += 1
        if pred_sup == gt_sup:
            correct_by_sup[gt_sup] += 1

        # 预测分布（只看预测，不管 GT）
        pred_dist[pred_sup] += 1

        # 混淆矩阵
        i = sup_to_idx[gt_sup]
        j = sup_to_idx[pred_sup]
        conf_mat[i, j] += 1

# ====== 3. 计算每个上位域的准确率 ======
acc_by_sup = []
for name in super_names:
    tot = total_by_sup.get(name, 0)
    cor = correct_by_sup.get(name, 0)
    acc = cor / tot if tot > 0 else 0.0
    acc_by_sup.append(acc)

print("各上位域准确率：")
for name, acc in zip(super_names, acc_by_sup):
    print(f"{name:20s}  acc={acc:.4f}  (n={total_by_sup.get(name,0)})")

# ====== 4. 画图：每个上位域的准确率 ======
plt.figure(figsize=(8, 4))
x = np.arange(len(super_names))
plt.bar(x, acc_by_sup)
plt.xticks(x, super_names, rotation=30, ha="right")
plt.ylim(0, 1.0)
plt.ylabel("Accuracy")
plt.title("Per-super-domain accuracy (Eight categories)")
plt.tight_layout()
plt.grid(axis="y", linestyle="--", alpha=0.4)

plt.savefig("acc_by_super.png", dpi=200)
print("保存图像: acc_by_super.png")

# ====== 5. 画图：预测分布（每个 super 被预测了多少次） ======
pred_counts = [pred_dist.get(name, 0) for name in super_names]

plt.figure(figsize=(8, 4))
x = np.arange(len(super_names))
plt.bar(x, pred_counts)
plt.xticks(x, super_names, rotation=30, ha="right")
plt.ylabel("Count")
plt.title("Predicted super-domain distribution")
plt.tight_layout()
plt.grid(axis="y", linestyle="--", alpha=0.4)

plt.savefig("pred_dist_super.png", dpi=200)
print("保存图像: pred_dist_super.png")

# ====== 6. （可选）画 8x8 混淆矩阵热力图 ======
# 转成归一化（每行除以行总数），更容易看模式
row_sums = conf_mat.sum(axis=1, keepdims=True).clip(min=1)
conf_norm = conf_mat / row_sums

plt.figure(figsize=(6, 5))
plt.imshow(conf_norm, interpolation="nearest")
plt.colorbar(label="Row-normalized freq")

plt.xticks(range(num_sup), super_names, rotation=30, ha="right")
plt.yticks(range(num_sup), super_names)
plt.xlabel("Predicted super")
plt.ylabel("Ground-truth super")
plt.title("Super-domain confusion matrix (row-normalized)")

plt.tight_layout()
plt.savefig("confusion_super.png", dpi=200)
print("保存图像: confusion_super.png")
