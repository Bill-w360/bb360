
import pandas as pd

# 读取 CSV 文件
df = pd.read_csv('./eval_out/domain_aggregated.csv')

# 转换为 JSON 并保存
df.to_json('./eval_out/domain_aggregated.json', orient='records', indent=2)

print("转换完成！")