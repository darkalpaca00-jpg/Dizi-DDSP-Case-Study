import os
import glob
import random
import shutil

# 配置路径
AUDIO_DIR = "/Volumes/logic资源/DiziData/synthetic_audio_dataset"
FEATURE_DIR = "/Volumes/logic资源/DiziData/synthetic_features"
OUTPUT_BASE = "/Volumes/logic资源/DiziData/dataset_split"

# 获取所有音频文件名前缀
wav_files = glob.glob(os.path.join(AUDIO_DIR, "*.wav"))
file_prefixes = [os.path.basename(f).replace(".wav", "") for f in wav_files]

# 随机打乱
random.seed(42) # 固定随机种子，确保结果可复现
random.shuffle(file_prefixes)

# 计算划分数量 (8:1:1)
total = len(file_prefixes)
train_num = int(total * 0.8)
val_num = int(total * 0.1)

train_files = file_prefixes[:train_num]
val_files = file_prefixes[train_num:train_num+val_num]
test_files = file_prefixes[train_num+val_num:]

print(f"📊 数据集规划完成：训练集 {len(train_files)} 个，验证集 {len(val_files)} 个，测试集 {len(test_files)} 个")

# 创建划分目录并生成清单
os.makedirs(OUTPUT_BASE, exist_ok=True)
for split_name, files in [("train", train_files), ("val", val_files), ("test", test_files)]:
    with open(os.path.join(OUTPUT_BASE, f"{split_name}_files.txt"), "w") as f:
        for file in files:
            f.write(f"{file}\n")

print(f"🎉 划分清单已成功写入外接硬盘！存放于: {OUTPUT_BASE}")
