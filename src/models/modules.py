from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class TemporalEncoder(nn.Module):
    def __init__(self, in_dim: int, d_model: int = 128, nhead: int = 4, num_layers: int = 2, dropout: float = 0.15):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.pe = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out_dim = d_model * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        h = self.pe(h)
        h = self.encoder(h)
        mean_pool = h.mean(dim=1)
        max_pool, _ = h.max(dim=1)
        return torch.cat([mean_pool, max_pool], dim=-1)


class SowingHead(nn.Module):
    def __init__(self, enc_dim: int, hidden: int = 256, sow_min: float = 200.0, sow_max: float = 420.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(enc_dim, hidden), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(hidden, 1),
        )
        self.sigmoid = nn.Sigmoid()
        self.sow_min = sow_min
        self.sow_max = sow_max

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        p = self.sigmoid(self.net(e))
        return self.sow_min + p * (self.sow_max - self.sow_min)


class SeasonLengthHead(nn.Module):
    def __init__(self, enc_dim: int, hidden: int = 256, sl_min: float = 30.0, sl_max: float = 380.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(enc_dim, hidden), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(hidden, 1),
        )
        self.sigmoid = nn.Sigmoid()
        self.sl_min = sl_min
        self.sl_max = sl_max

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        p = self.sigmoid(self.net(e))
        return self.sl_min + p * (self.sl_max - self.sl_min)


class TeacherARDecoder(nn.Module):
    def __init__(self, enc_dim: int, hidden: int = 256, n_stages: int = 10):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(enc_dim + 1, hidden), nn.ReLU(), nn.Dropout(0.10),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, 1),
            ) for _ in range(n_stages)
        ])
        self.act = nn.Sigmoid()

    def forward(
        self,
        enc_feat: torch.Tensor,
        y_teacher: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 1.0,
    ) -> torch.Tensor:
        bsz = enc_feat.size(0)
        prev = torch.zeros(bsz, 1, device=enc_feat.device)
        outs = []
        for i, block in enumerate(self.blocks):
            if (y_teacher is not None) and (i > 0):
                if teacher_forcing_ratio >= 1.0:
                    prev_in = y_teacher[:, i - 1:i]
                elif teacher_forcing_ratio <= 0.0:
                    prev_in = prev
                else:
                    mask = (torch.rand(bsz, 1, device=enc_feat.device) < teacher_forcing_ratio).float()
                    prev_in = mask * y_teacher[:, i - 1:i] + (1.0 - mask) * prev
            else:
                prev_in = prev
            yi = self.act(block(torch.cat([enc_feat, prev_in], dim=-1)))
            outs.append(yi)
            prev = yi
        return torch.cat(outs, dim=-1)


class NonLinearActivation(nn.Module):
    def __init__(self, n_stages: int = 10, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_stages, hidden), nn.ReLU(),
            nn.Linear(hidden, n_stages), nn.Sigmoid(),
        )

    def forward(self, y_teacher_phase: torch.Tensor) -> torch.Tensor:
        return self.net(y_teacher_phase)


class StudentParallelDecoder(nn.Module):
    def __init__(self, enc_dim: int, hidden: int = 256, n_stages: int = 10):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(enc_dim, hidden), nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(hidden, n_stages),
        )
        self.act = nn.Sigmoid()

    def forward(self, enc_feat: torch.Tensor) -> torch.Tensor:
        return self.act(self.mlp(enc_feat))
