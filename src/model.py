"""
Model architectures for cyclone prediction.
- CycloneLSTM: baseline LSTM with multi-task heads
- HybridCycloneModel: LSTM + ConvLSTM + Transformer fusion
Extracted from notebooks/04_train.py and notebooks/06_train_hybrid.py.
"""

import torch
import torch.nn as nn


# ─── LSTM Baseline ───────────────────────────────────────────────────────────

class CycloneLSTM(nn.Module):
    """
    Baseline LSTM: shared encoder + 4 task-specific output heads.
    Input  : (batch, lookback=8, features=17)
    Outputs: track (6), wind (3), ri (1), landfall (1)
    """

    def __init__(self, input_size: int = 17, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)

        self.track_head = nn.Sequential(
            nn.Linear(hidden_size, 64), nn.ReLU(), nn.Linear(64, 6)
        )
        self.wind_head = nn.Sequential(
            nn.Linear(hidden_size, 32), nn.ReLU(), nn.Linear(32, 3)
        )
        self.ri_head = nn.Sequential(
            nn.Linear(hidden_size, 32), nn.ReLU(), nn.Linear(32, 1)
        )
        self.landfall_head = nn.Sequential(
            nn.Linear(hidden_size, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        feat = self.dropout(out[:, -1])
        return (
            self.track_head(feat),
            self.wind_head(feat),
            self.ri_head(feat),
            self.landfall_head(feat),
        )


# ─── Hybrid Model Components ────────────────────────────────────────────────

class TrackEncoder(nn.Module):
    """LSTM encoder for IBTrACS track + feature sequence."""

    def __init__(self, input_size: int = 17, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.dropout(out[:, -1])


class ConvLSTMCell(nn.Module):
    """Single ConvLSTM cell for spatial-temporal processing."""

    def __init__(self, in_channels: int, hidden_channels: int,
                 kernel_size: int = 3):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=kernel_size // 2,
        )

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, g, o = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class ERA5Encoder(nn.Module):
    """
    ConvLSTM encoder for ERA5 spatial patches.
    Input : (B, T=4, C=8, H=20, W=20)
    Output: (B, 64) flattened feature vector
    """

    def __init__(self, in_channels: int = 8, hidden_channels: int = 32,
                 dropout: float = 0.2):
        super().__init__()
        self.hidden_channels = hidden_channels

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((5, 5)),
        )
        self.clstm = ConvLSTMCell(hidden_channels, hidden_channels, kernel_size=3)
        self.flatten = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_channels * 5 * 5, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        hc = self.hidden_channels
        h = torch.zeros(B, hc, 5, 5, device=x.device)
        c = torch.zeros(B, hc, 5, 5, device=x.device)
        for t in range(T):
            feat = self.cnn(x[:, t])
            h, c = self.clstm(feat, h, c)
        return self.flatten(h)


class HybridCycloneModel(nn.Module):
    """
    Full hybrid model:
        TrackEncoder (LSTM)    -> 128-dim
        ERA5Encoder (ConvLSTM) -> 64-dim
        Concat                 -> 192-dim
        Transformer            -> 192-dim
        4 output heads
    """

    def __init__(self, track_input: int = 17, track_hidden: int = 128,
                 track_layers: int = 2, era5_channels: int = 8,
                 era5_hidden: int = 32, fusion_dim: int = 192,
                 n_heads: int = 4, n_layers: int = 2,
                 dropout: float = 0.2):
        super().__init__()

        self.track_enc = TrackEncoder(track_input, track_hidden,
                                      track_layers, dropout)
        self.era5_enc = ERA5Encoder(era5_channels, era5_hidden, dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=fusion_dim,
            nhead=n_heads,
            dim_feedforward=fusion_dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)

        self.track_head = nn.Sequential(
            nn.Linear(fusion_dim, 96), nn.ReLU(), nn.Linear(96, 6)
        )
        self.wind_head = nn.Sequential(
            nn.Linear(fusion_dim, 48), nn.ReLU(), nn.Linear(48, 3)
        )
        self.ri_head = nn.Sequential(
            nn.Linear(fusion_dim, 48), nn.ReLU(), nn.Linear(48, 1)
        )
        self.landfall_head = nn.Sequential(
            nn.Linear(fusion_dim, 48), nn.ReLU(), nn.Linear(48, 1)
        )

    def forward(self, track, era5):
        track_feat = self.track_enc(track)
        era5_feat  = self.era5_enc(era5)
        fused = torch.cat([track_feat, era5_feat], dim=1)
        fused = fused.unsqueeze(1)
        fused = self.transformer(fused).squeeze(1)
        fused = self.dropout(fused)
        return (
            self.track_head(fused),
            self.wind_head(fused),
            self.ri_head(fused),
            self.landfall_head(fused),
        )
