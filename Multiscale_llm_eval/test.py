from modelscope import MsDataset
import os

# 设置缓存路径
os.environ["MODELSCOPE_CACHE"] = "/mnt/Data/yangyongbiao/.cache/modelscope"
os.environ["MODELSCOPE_HUB_MIRROR"] = "https://modelscope.cn"

# 想要下载的数据集列表
datasets = [
    "modelscope/boolq",
    "modelscope/gsm8k",
    "modelscope/hellaswag",
    "modelscope/super_glue",
    "modelscope/mmlu"
]

for name in datasets:
    print(f"\n🚀 正在下载数据集：{name}")
    try:
        ds = MsDataset.load(name, split='validation')
        print(f"✅ 已成功下载 {name}, 样本数: {len(ds)}")
    except Exception as e:
        print(f"❌ 下载失败 {name}: {e}")
