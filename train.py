import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import DiziDataset
from model import DiziDDSP
from tqdm import tqdm
import os
import numpy as np

device = torch.device("cpu")
print("🚀 SOTA Training Pipeline Started with Physical Constraints...")

class MultiScaleSpectralLoss(nn.Module):
    def __init__(self, fft_sizes=[2048, 1024, 512], hop_sizes=[512, 256, 128], win_lengths=[2048, 1024, 512]):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        
    def forward(self, y_true, y_pred):
        loss = 0.0
        for n_fft, hop, win in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            window = torch.hann_window(win, device=y_true.device)
            stft_true = torch.stft(y_true, n_fft=n_fft, hop_length=hop, win_length=win, window=window, return_complex=True).abs()
            stft_pred = torch.stft(y_pred, n_fft=n_fft, hop_length=hop, win_length=win, window=window, return_complex=True).abs()
            
            lin_loss = torch.mean(torch.abs(stft_true - stft_pred))
            log_loss = torch.mean(torch.abs(torch.log(stft_true + 1e-5) - torch.log(stft_pred + 1e-5)))
            loss += lin_loss + 1.0 * log_loss
        return loss

def synthesize_noise(noise_param, target_size=480000):
    batch, frames, bands = noise_param.shape
    
    # Interpolate 65 bands to 241 bins for ISTFT with n_fft=480, hop_length=240
    noise_param_interp = torch.nn.functional.interpolate(
        noise_param, size=241, mode='linear', align_corners=False
    ) # [batch, frames, 241]
    
    n_fft = 480
    hop_length = target_size // frames  # 480000 // 2000 = 240
    
    uniform = torch.rand(batch, frames, 241, device=noise_param.device)
    phases = 2 * np.pi * uniform
    
    real = noise_param_interp * torch.cos(phases)
    imag = noise_param_interp * torch.sin(phases)
    spec = torch.complex(real, imag).permute(0, 2, 1)  # [batch, 241, frames]
    
    window = torch.hann_window(n_fft, device=noise_param.device)
    noise_audio = torch.istft(spec, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window=window, length=target_size)
    return noise_audio

train_dataset = DiziDataset(split="train")
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

model = DiziDDSP().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
spectral_loss_fn = MultiScaleSpectralLoss().to(device)

os.makedirs("./checkpoints", exist_ok=True)
epochs = 5

for epoch in range(epochs):
    model.train()
    total_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]")
    
    for batch in loop:
        f0 = batch["f0"].to(device)
        loudness = batch["loudness"].to(device)
        audio_true = batch["audio"].to(device)
        
        amp_pred, noise_param = model(f0, loudness)
        
        # A. Harmonic Synthesis
        f0_up = torch.nn.functional.interpolate(f0.unsqueeze(1), size=480000, mode='linear', align_corners=False).squeeze(1)
        amp_up = torch.nn.functional.interpolate(amp_pred.permute(0, 2, 1), size=480000, mode='linear', align_corners=False).permute(0, 2, 1)
        harmonics = torch.arange(1, 61, device=device).float().unsqueeze(0).unsqueeze(0)
        phases = 2 * np.pi * torch.cumsum(f0_up.unsqueeze(-1) * harmonics, dim=1) / 48000
        harmonic_audio = torch.sum(amp_up * torch.sin(phases), dim=-1)
        
        # B. Noise Synthesis
        noise_audio = synthesize_noise(noise_param, target_size=480000)
        
        # C. Combine Audio
        audio_pred = harmonic_audio + noise_audio
        
        loss = spectral_loss_fn(audio_true, audio_pred)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
        
    print(f"📈 Epoch {epoch+1} Complete! MSSL Loss: {total_loss/len(train_loader):.4f}")
    torch.save(model.state_dict(), f"./checkpoints/dizi_ddsp_epoch_{epoch+1}.pt")