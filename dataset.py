import os
import torch
from torch.utils.data import Dataset
import numpy as np
import soundfile as sf

class DiziDataset(Dataset):
    def __init__(self, split="train", base_dir="/Volumes/logic资源/DiziData"):
        self.base_dir = base_dir
        self.audio_dir = os.path.join(base_dir, "synthetic_audio_dataset")
        self.feature_dir = os.path.join(base_dir, "synthetic_features")
        
        self.max_frames = 2000  
        self.max_samples = self.max_frames * 240  
        
        split_file = os.path.join(base_dir, "dataset_split", f"{split}_files.txt")
        with open(split_file, "r") as f:
            self.file_list = [line.strip() for line in f.readlines() if line.strip()]
            
    def __len__(self):
        return len(self.file_list)
        
    def __getitem__(self, idx):
        file_prefix = self.file_list[idx]
        
        audio_path = os.path.join(self.audio_dir, f"{file_prefix}.wav")
        audio, sr = sf.read(audio_path)
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=-1)
            
        feature_path = os.path.join(self.feature_dir, f"{file_prefix}_features.npy")
        features = np.load(feature_path)
        f0 = features[:, 0]
        loudness = features[:, 1]
        
        # 【核心修复】将 dB (-80 到 0) 严格缩放到 (0 到 1) 的安全正数区间
        loudness = np.clip(loudness, -80.0, 0.0)
        loudness = (loudness + 80.0) / 80.0
        
        if len(audio) >= self.max_samples:
            audio = audio[:self.max_samples]
        else:
            audio = np.pad(audio, (0, self.max_samples - len(audio)))
            
        if len(f0) >= self.max_frames:
            f0 = f0[:self.max_frames]
            loudness = loudness[:self.max_frames]
        else:
            f0 = np.pad(f0, (0, self.max_frames - len(f0)))
            loudness = np.pad(loudness, (0, self.max_frames - len(loudness)))
            
        return {
            "audio": torch.FloatTensor(audio),
            "f0": torch.FloatTensor(f0),
            "loudness": torch.FloatTensor(loudness)
        }
