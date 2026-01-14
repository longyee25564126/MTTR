# Copyright (c) Ruopeng Gao. All Rights Reserved.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch


@dataclass
class TrackState:
    traj: torch.Tensor
    pad_mask: torch.Tensor
    miss_count: int
    active: bool


class TrackBank:
    def __init__(
        self,
        traj_len: int = 30,
        hidden_dim: int = 256,
        max_miss: int = 30,
        miss_token: torch.Tensor | None = None,
        pad_token: torch.Tensor | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        self.traj_len = traj_len
        self.hidden_dim = hidden_dim
        self.max_miss = max_miss
        self.device = device if device is not None else torch.device("cpu")
        self.dtype = dtype if dtype is not None else torch.float32
        self.miss_token = miss_token
        self.pad_token = pad_token
        self.tracks: Dict[int, TrackState] = {}

    def ensure_track(self, label: int) -> None:
        if label in self.tracks:
            return
        pad_token = self._get_token(self.pad_token)
        traj = pad_token.unsqueeze(0).repeat(self.traj_len, 1)
        pad_mask = torch.ones(self.traj_len, dtype=torch.bool, device=self.device)
        self.tracks[label] = TrackState(
            traj=traj,
            pad_mask=pad_mask,
            miss_count=0,
            active=True,
        )

    def push_roi(self, label: int, roi_feat: torch.Tensor) -> None:
        self.ensure_track(label)
        state = self.tracks[label]
        roi_feat = roi_feat.to(device=self.device, dtype=self.dtype)
        state.traj = torch.cat([state.traj[1:], roi_feat.unsqueeze(0)], dim=0)
        state.pad_mask = torch.cat(
            [state.pad_mask[1:], torch.zeros(1, dtype=torch.bool, device=self.device)],
            dim=0,
        )
        state.miss_count = 0
        state.active = True

    def push_miss(self, label: int) -> None:
        self.ensure_track(label)
        state = self.tracks[label]
        miss_token = self._get_token(self.miss_token)
        state.traj = torch.cat([state.traj[1:], miss_token.unsqueeze(0)], dim=0)
        state.pad_mask = torch.cat(
            [state.pad_mask[1:], torch.zeros(1, dtype=torch.bool, device=self.device)],
            dim=0,
        )
        state.miss_count += 1
        self.maybe_deactivate(label)

    def maybe_deactivate(self, label: int) -> None:
        state = self.tracks.get(label, None)
        if state is None:
            return
        if state.miss_count > self.max_miss:
            state.active = False

    def get_active_labels(self) -> List[int]:
        return [label for label, state in self.tracks.items() if state.active]

    def export_batch_traj_and_mask(self, labels: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(labels) == 0:
            empty_traj = torch.zeros(
                (0, self.traj_len, self.hidden_dim),
                device=self.device,
                dtype=self.dtype,
            )
            empty_mask = torch.zeros(
                (0, self.traj_len),
                device=self.device,
                dtype=torch.bool,
            )
            return empty_traj, empty_mask
        traj_batch = torch.stack([self.tracks[label].traj for label in labels], dim=0)
        mask_batch = torch.stack([self.tracks[label].pad_mask for label in labels], dim=0)
        return traj_batch, mask_batch

    def export_batch(self, labels: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.export_batch_traj_and_mask(labels)

    def _get_token(self, token: torch.Tensor | None) -> torch.Tensor:
        if token is None:
            return torch.zeros(self.hidden_dim, device=self.device, dtype=self.dtype)
        return token.to(device=self.device, dtype=self.dtype)
