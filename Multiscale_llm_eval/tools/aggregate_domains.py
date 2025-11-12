import pandas as pd

# 读取原始 classification_results.csv
df = pd.read_csv("eval_out/classification_results.csv")

# 定义映射规则（每个正则关键字对应目标领域）
mapping = {
    "math_reasoning": ["math", "algebra", "geometry", "statistics"],
    "commonsense": ["commonsense", "hellaswag"],
    "reading_comprehension": ["reading", "boolq", "super_glue"],
    "general_knowledge": ["mmlu_overall", "general_knowledge"],
    "humanities": ["humanities", "history", "philosophy", "law", "jurisprudence", "religions"],
    "social_science": ["social", "sociology", "politics", "economics", "business", "management", "marketing"],
    "stem": ["stem", "physics", "chemistry", "biology", "engineering", "computer", "machine_learning"],
    "other_knowledge": ["other", "miscellaneous", "moral", "nutrition", "public_relations"],
}

# 提取模型基础信息列
base_cols = ["model", "parameters", "family", "release_date"]
score_cols = [c for c in df.columns if c not in base_cols + ["tier", "cluster", "cluster_name", "overall_score"]]

# 初始化新DataFrame
agg_data = df[base_cols].copy()

# 按映射表计算每个领域的平均分
for domain, keywords in mapping.items():
    domain_cols = [c for c in score_cols if any(k in c.lower() for k in keywords)]
    if domain_cols:
        agg_data[domain] = df[domain_cols].mean(axis=1)
    else:
        agg_data[domain] = 0.0

# 重新计算 overall
agg_data["overall_score"] = agg_data[[d for d in mapping.keys()]].mean(axis=1).round(3)

# 输出结果
agg_data = agg_data.round(3)
agg_data.to_csv("./eval_out/domain_aggregated.csv", index=False, encoding="utf-8")

print("✅ 聚合完成：./eval_out/domain_aggregated.csv")
print("包含领域：", list(mapping.keys()))
