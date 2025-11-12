import json
print(json.load(open("outputs/mmlu_mtl_fromjson/label_subjects.json"))[:20])
# 如果你还有训练时的 subjects 文件（train_outdir），也对比：
print(json.load(open("<train_outdir>/label_subjects.json"))[:20])
