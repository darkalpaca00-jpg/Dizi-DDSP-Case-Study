import torch
import torch.nn as nn

class DiziDDSP(nn.Module):
    def __init__(self, n_harmonics=60, n_noise_bands=65):
        super().__init__()
        self.n_harmonics = n_harmonics
        self.n_noise_bands = n_noise_bands
        
        # Upgrade to LayerNorm to support [Batch, Time, Feature] structure
        self.mlp_f0 = nn.Sequential(nn.Linear(1, 64), nn.LayerNorm(64), nn.ReLU())
        self.mlp_loudness = nn.Sequential(nn.Linear(1, 64), nn.LayerNorm(64), nn.ReLU())
        
        self.rnn = nn.GRU(128, 256, batch_first=True, bidirectional=True)
        
        # Physical constraints
        self.global_harmonic_amp = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid())
        self.harmonic_distribution = nn.Sequential(nn.Linear(512, n_harmonics), nn.Softmax(dim=-1))
        
        self.noise_out = nn.Sequential(nn.Linear(512, n_noise_bands), nn.Sigmoid())

    def forward(self, f0, loudness):
        # f0, loudness: [batch, frames]
        feat_f0 = self.mlp_f0(f0.unsqueeze(-1))
        feat_lou = self.mlp_loudness(loudness.unsqueeze(-1))
        x = torch.cat([feat_f0, feat_lou], dim=-1)
        
        x, _ = self.rnn(x) # [batch, frames, 512]
        
        global_amp = self.global_harmonic_amp(x) * loudness.unsqueeze(-1)
        dist = self.harmonic_distribution(x)
        amplitudes = global_amp * dist * 2.0  # 2.0 is energy compensation factor
        
        noise_param = self.noise_out(x) * loudness.unsqueeze(-1) * 0.1
        
        return amplitudes, noise_param