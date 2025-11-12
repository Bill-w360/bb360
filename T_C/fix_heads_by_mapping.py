# fix_heads_by_mapping_simple.py (如果还没建，就建这个脚本)
import os, json, collections, torch, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--pred_file", default="outputs/mmlu_mtl_fromjson/pred_test_1.jsonl")
ap.add_argument("--pred_remapped", default="outputs/mmlu_mtl_fromjson/pred_test_1_remapped.jsonl")
ap.add_argument("--model_dir", default="outputs/mmlu_mtl_fromjson")
ap.add_argument("--in_heads", default=None)   # 默认 model_dir/mtl_heads.pt
ap.add_argument("--out_heads", default=None)  # 默认 model_dir/mtl_heads_fixed.pt
args = ap.parse_args()

model_dir = args.model_dir

# 1) mapping: pred_name -> most_common_gt
mapping = {}
if os.path.exists(args.pred_remapped):
    c = {}
    with open(args.pred_remapped, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            pred = r["_pred_subject"]
            new = r.get("_pred_subject_remapped")
            if new:
                c.setdefault(pred, collections.Counter())[new] += 1
    for pred, ctr in c.items():
        mapping[pred] = ctr.most_common(1)[0][0]
else:
    c = {}
    with open(args.pred_file, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            pred = r["_pred_subject"]; gt = r.get("subject")
            if gt:
                c.setdefault(pred, collections.Counter())[gt] += 1
    for pred, ctr in c.items():
        mapping[pred] = ctr.most_common(1)[0][0]

print("Mapping sample (pred -> most_common_gt):")
for k in list(mapping.keys())[:20]:
    print(k, "->", mapping[k])

# 2) labels
label_path = os.path.join(model_dir, "label_subjects.json")
labels = json.load(open(label_path, "r", encoding="utf-8"))
label_idx = {name:i for i,name in enumerate(labels)}

# 3) heads
ckpt_path = args.in_heads or os.path.join(model_dir, "mtl_heads.pt")
print("Loading heads from:", ckpt_path)
ckpt = torch.load(ckpt_path, map_location="cpu")
W = ckpt["head_sub"]["weight"]
b = ckpt["head_sub"]["bias"]
num_sub, hidden = W.shape
print("num_sub =", num_sub, "hidden =", hidden)

old_name_to_idx = {name:i for i,name in enumerate(labels)}
perm = [-1]*num_sub
used_old = set()

for pred_name, gt_name in mapping.items():
    if pred_name not in old_name_to_idx or gt_name not in label_idx:
        continue
    old_idx = old_name_to_idx[pred_name]
    new_idx = label_idx[gt_name]
    if perm[new_idx] == -1:
        perm[new_idx] = old_idx
        used_old.add(old_idx)

remaining_old = [i for i in range(num_sub) if i not in used_old]
ri = 0
for i in range(num_sub):
    if perm[i] == -1:
        perm[i] = remaining_old[ri]
        ri += 1

assert sorted(perm) == list(range(num_sub)), "perm is not a permutation!"

print("perm[:20] =", perm[:20])

W_new = W[perm, :].clone()
b_new = b[perm].clone()
ckpt["head_sub"] = {"weight": W_new, "bias": b_new}

out_path = args.out_heads or os.path.join(model_dir, "mtl_heads_fixed.pt")
torch.save(ckpt, out_path)
print("Saved permuted heads to", out_path)
