import os
import glob
import numpy as np
import librosa
from tqdm import tqdm

# 1. 配置路径（利用之前建好的软链接）
AUDIO_DIR = "./synthetic_audio_dataset"
# 特征直接存在外接硬盘中
FEATURE_DIR = "/Volumes/logic资源/DiziData/synthetic_features"
os.makedirs(FEATURE_DIR, exist_ok=True)

# 2. 扫描所有渲染好的 wav 文件
wav_files = glob.glob(os.path.join(AUDIO_DIR, "*.wav"))
print(f"🎬 开始提取特征，共 {len(wav_files)} 个音频文件...")

# 3. 循环处理
for wav_path in tqdm(wav_files):
    file_name = os.path.basename(wav_path).replace(".wav", "")
    
    try:
        # 加载音频，保持 48000Hz 采样率
        y, sr = librosa.load(wav_path, sr=48000)
        
        # --- 提取基频 f0 (音高) ---
        # 限制在 300Hz 到 1700Hz 之间
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, 
            fmin=300, 
            fmax=1700, 
            sr=sr, 
            frame_length=2048, 
            hop_length=240 # 5ms 一个点
        )
        f0 = np.nan_to_num(f0)
        
        # --- 提取响度 (Loudness) ---
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=240)
        loudness = librosa.amplitude_to_db(rms, ref=np.max)
        loudness = loudness.squeeze()
        
        # --- 对齐长度并保存 ---
        min_len = min(len(f0), len(loudness))
        f0 = f0[:min_len]
        loudness = loudness[:min_len]
        
        features = np.stack([f0, loudness], axis=-1)
        
        # 保存到外接硬盘
        output_path = os.path.join(FEATURE_DIR, f"{file_name}_features.npy")
        np.save(output_path, features)
        
    except Exception as e:
        print(f"❌ 文件 {file_name} 提取失败，原因: {e}")

print(f"🎉 所有人声/乐器特征提取完毕！全部保存在: {FEATURE_DIR}")
