
import pandas as pd

# 读取 CSV 文件
df_1 = pd.read_csv('./outputs/results/audit_mmlu_pro/confusion_pred_vs_gold.csv')
df_2 = pd.read_csv('./outputs/results/audit_mmlu_pro/confusion_pred_vs_silver.csv')
df_3 = pd.read_csv('./outputs/results/audit_mmlu_pro/confusion_silver_vs_gold.csv')
df_4 = pd.read_csv('./outputs/results/audit_mmlu_pro/disagreements.csv')

# 转换为 JSON 并保存
df_1.to_json('./outputs/results/audit_mmlu_pro/confusion_pred_vs_gold.json', orient='records', indent=2)
df_2.to_json('./outputs/results/audit_mmlu_pro/confusion_pred_vs_silver.json', orient='records', indent=2)
df_3.to_json('./outputs/results/audit_mmlu_pro/confusion_silver_vs_gold.json', orient='records', indent=2)
df_4.to_json('./outputs/results/audit_mmlu_pro/disagreements.json', orient='records', indent=2)

print("转换完成！")