# remap_preds.py
import json, collections, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--pred", default="outputs/mmlu_mtl_fromjson/pred_test_1.jsonl")
ap.add_argument("--out", default="outputs/mmlu_mtl_fromjson/pred_test_1_remapped.jsonl")
args = ap.parse_args()

pred2gt = {}
with open(args.pred, "r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        pred = r["_pred_subject"]
        gt = r.get("subject")
        if gt is None:
            continue
        pred2gt.setdefault(pred, collections.Counter())[gt] += 1

mapping = {p: ctr.most_common(1)[0][0] for p, ctr in pred2gt.items()}

print("Mapping (pred -> most_common_gt) sample:")
for k in list(mapping.keys())[:20]:
    print(k, "->", mapping[k])

total = 0
correct_before = 0
correct_after = 0

with open(args.pred, "r", encoding="utf-8") as f, \
     open(args.out, "w", encoding="utf-8") as fo:
    for line in f:
        r = json.loads(line)
        total += 1
        gt = r.get("subject")
        pred = r["_pred_subject"]
        if gt is not None:
            if pred == gt:
                correct_before += 1
            if pred in mapping:
                newpred = mapping[pred]
                r["_pred_subject_remapped"] = newpred
                if newpred == gt:
                    correct_after += 1
        fo.write(json.dumps(r, ensure_ascii=False) + "\n")

print("Total rows:", total)
print("Correct before:", correct_before, "=> acc_before:", correct_before/total)
print("Correct after:", correct_after, "=> acc_after:", correct_after/total)
print("Remapped preds saved to", args.out)
