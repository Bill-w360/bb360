import json
import matplotlib.pyplot as plt

# 1. 读入日志文件（路径按你的实际情况改）
log_path = "outputs/mmlu_mtl_fromjson/training_log.jsonl"

epochs = []
acc_sup = []

with open(log_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)

        # 只关心有 eval_acc_sup 的记录（即每个 epoch 评估那一条）
        if "eval_acc_sup" in rec:
            # epoch 在我们自定义 callback 里已经塞进去了
            ep = rec.get("epoch")
            acc = rec["eval_acc_sup"]
            if ep is not None:
                epochs.append(ep)
                acc_sup.append(acc)

print("epochs:", epochs)
print("eval_acc_sup:", acc_sup)

# 2. 画图
plt.figure(figsize=(6, 4))
plt.plot(epochs, acc_sup, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Eval Accuracy (8 domains, acc_sup)")
plt.title("Epoch vs 8-domain Accuracy")
plt.grid(True)
plt.tight_layout()

# 如果在有图形界面的环境：
plt.show()

# 如果是纯服务器（没有图形界面），可以改用保存到文件：
plt.savefig("epoch_vs_acc_sup.png", dpi=200)
