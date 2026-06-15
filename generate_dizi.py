import torch
import numpy as np
import soundfile as sf
import os
from dataset import DiziDataset
from model import DiziDDSP
from train import synthesize_noise  # 复用训练集的高性能噪声引擎

device = torch.device("cpu")
OUTPUT_DIR = "/Volumes/logic资源/DiziData/ai_generated_audio"
os.makedirs(OUTPUT_DIR, exist_ok=True)

test_dataset = DiziDataset(split="test")
model = DiziDDSP().to(device)
model.load_state_dict(torch.load("./checkpoints/dizi_ddsp_epoch_5.pt", map_location=device))
model.eval()

sample = test_dataset[0]
f0 = sample["f0"].unsqueeze(0).to(device)
loudness = sample["loudness"].unsqueeze(0).to(device)

with torch.no_grad():
    amp_pred, noise_param = model(f0, loudness)
    
    # 1. 基础信号插值与高精度谐波合成
    f0_up = torch.nn.functional.interpolate(f0.unsqueeze(1), size=480000, mode='linear', align_corners=False).squeeze(1)
    amp_up = torch.nn.functional.interpolate(amp_pred.permute(0, 2, 1), size=480000, mode='linear', align_corners=False).permute(0, 2, 1)
    
    harmonics = torch.arange(1, 61, device=device).float().unsqueeze(0).unsqueeze(0)
    phases = 2 * np.pi * torch.cumsum(f0_up.unsqueeze(-1) * harmonics, dim=1) / 48000
    harmonic_audio = torch.sum(amp_up * torch.sin(phases), dim=-1)
    
    # 2. 动态注入真实的呼吸气流声
    noise_audio = synthesize_noise(noise_param, target_size=480000)
    
    # 3. 结合
    audio_pred = (harmonic_audio + noise_audio).squeeze(0).cpu().numpy()

# 4. 纯物理无损后处理
audio_pred = audio_pred - np.mean(audio_pred)
max_val = np.max(np.abs(audio_pred))
if max_val > 1e-5:
    audio_pred = audio_pred / max_val
audio_pred = audio_pred * 0.8  # 锁定在安全的 -2dB 峰值

output_path = os.path.join(OUTPUT_DIR, "ai_dizi_sota.wav")
sf.write(output_path, audio_pred, 48000)

print(f"🎉 物理约束 SOTA 音频渲染完成！去 Logic Pro 里看真正的、布满锯齿和微动态的艺术品波形吧！")