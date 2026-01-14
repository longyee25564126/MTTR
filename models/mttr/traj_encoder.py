# Copyright (c) Ruopeng Gao. All Rights Reserved.

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class TrajEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        traj_len: int = 30,
        num_layers: int = 6,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        use_temporal_pos: bool = True,
        norm_first: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.traj_len = traj_len
        self.use_temporal_pos = use_temporal_pos

        self.trk_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.trk_pos = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, traj_len, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=norm_first,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, traj: torch.Tensor, pad_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if pad_mask is None:
            raise ValueError("pad_mask is required for TrajEncoder.")
        if traj.dim() != 3:
            raise ValueError("traj must be (N_tracks, L, C).")
        if traj.shape[1] != self.traj_len:
            raise ValueError(f"traj length {traj.shape[1]} != traj_len {self.traj_len}.")
        if traj.shape[2] != self.hidden_dim:
            raise ValueError("traj channel dim must match hidden_dim.")

        n_tracks = traj.shape[0]
        if n_tracks == 0:
            return traj.new_zeros((0, self.hidden_dim))

        if pad_mask.shape != (n_tracks, self.traj_len):
            raise ValueError("pad_mask must be (N_tracks, L) and match traj length.")
        if pad_mask.dtype != torch.bool:
            pad_mask = pad_mask.to(torch.bool)

        trk_token = self.trk_token.to(device=traj.device, dtype=traj.dtype).expand(n_tracks, -1, -1)
        x = torch.cat([trk_token, traj], dim=1)

        if self.use_temporal_pos:
            trk_pos = self.trk_pos.to(device=traj.device, dtype=traj.dtype).expand(n_tracks, -1, -1)
            pos = self.pos_embed.to(device=traj.device, dtype=traj.dtype).expand(n_tracks, -1, -1)
            x = x + torch.cat([trk_pos, pos], dim=1)

        trk_mask = torch.zeros((n_tracks, 1), dtype=torch.bool, device=traj.device)
        src_key_padding_mask = torch.cat([trk_mask, pad_mask], dim=1)

        encoded = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return encoded[:, 0, :]


def _sanity_check():
    torch.manual_seed(0)
    encoder = TrajEncoder()
    traj = torch.randn(4, 30, 256)
    pad_mask = torch.zeros(4, 30, dtype=torch.bool)
    pad_mask[:, :5] = True
    out = encoder(traj, pad_mask)
    print("track_queries:", out.shape)


if __name__ == "__main__":
    _sanity_check()
