# Copyright (c) Ruopeng Gao. All Rights Reserved.

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.ops import roi_align
except Exception:  # pragma: no cover - fallback handled at runtime
    roi_align = None


class ROIEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        pool_size: int = 7,
        num_layers: int = 2,
        nhead: int = 8,
        dropout: float = 0.0,
        use_2d_pos: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pool_size = pool_size
        self.use_2d_pos = use_2d_pos

        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        if use_2d_pos:
            dim_y = hidden_dim // 2
            dim_x = hidden_dim - dim_y
            self.pos_y = nn.Parameter(torch.zeros(pool_size, dim_y))
            self.pos_x = nn.Parameter(torch.zeros(pool_size, dim_x))
        else:
            self.pos_y = None
            self.pos_x = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(
        self,
        feat_highres: torch.Tensor,
        boxes: torch.Tensor,
        keep_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if keep_mask is None:
            raise ValueError("keep_mask is required for ROIEncoder.")
        if feat_highres.dim() != 4:
            raise ValueError("feat_highres must be (B, C, H, W).")
        if boxes.dim() != 3 or boxes.shape[-1] != 4:
            raise ValueError("boxes must be (B, Nq, 4).")
        if keep_mask.shape[:2] != boxes.shape[:2]:
            raise ValueError("keep_mask must match boxes on (B, Nq).")
        if feat_highres.shape[1] != self.hidden_dim:
            raise ValueError("feat_highres channel dim must match hidden_dim.")

        device = feat_highres.device
        boxes = boxes.to(device)
        keep_mask = keep_mask.to(device)
        if keep_mask.dtype != torch.bool:
            keep_mask = keep_mask.to(torch.bool)

        indices = keep_mask.nonzero(as_tuple=False)
        if indices.numel() == 0:
            roi_feats = feat_highres.new_zeros((0, self.hidden_dim))
            empty_indices = torch.empty((0, 2), device=device, dtype=torch.long)
            return roi_feats, empty_indices

        boxes_keep = boxes[indices[:, 0], indices[:, 1]]
        rois = self._boxes_to_rois(
            batch_idx=indices[:, 0],
            boxes=boxes_keep,
            height=feat_highres.shape[-2],
            width=feat_highres.shape[-1],
        )

        if roi_align is None:
            roi_feats = self._roi_align_fallback(feat_highres, rois)
        else:
            roi_feats = roi_align(
                feat_highres,
                rois,
                output_size=self.pool_size,
                spatial_scale=1.0,
                aligned=True,
            )

        tokens = roi_feats.flatten(2).transpose(1, 2)
        if self.use_2d_pos:
            pos = self._build_2d_pos(tokens.device, tokens.dtype)
            tokens = tokens + pos.unsqueeze(0)
        cls_token = self.cls_token.to(dtype=tokens.dtype, device=tokens.device)
        cls_token = cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([cls_token, tokens], dim=1)

        encoded = self.encoder(tokens)
        roi_out = encoded[:, 0, :]
        return roi_out, indices.to(dtype=torch.long)

    def _build_2d_pos(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        pos_y = self.pos_y.to(device=device, dtype=dtype)
        pos_x = self.pos_x.to(device=device, dtype=dtype)
        pos_y = pos_y[:, None, :].expand(self.pool_size, self.pool_size, -1)
        pos_x = pos_x[None, :, :].expand(self.pool_size, self.pool_size, -1)
        pos = torch.cat([pos_y, pos_x], dim=-1)
        return pos.reshape(self.pool_size * self.pool_size, self.hidden_dim)

    def _boxes_to_rois(
        self,
        batch_idx: torch.Tensor,
        boxes: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        cx, cy, w, h = boxes.unbind(-1)
        w_scale = float(width)
        h_scale = float(height)
        x1 = (cx - 0.5 * w) * w_scale
        y1 = (cy - 0.5 * h) * h_scale
        x2 = (cx + 0.5 * w) * w_scale
        y2 = (cy + 0.5 * h) * h_scale

        max_x = w_scale - 1.0
        max_y = h_scale - 1.0
        x1 = x1.clamp(0.0, max_x)
        y1 = y1.clamp(0.0, max_y)
        x2 = x2.clamp(0.0, max_x)
        y2 = y2.clamp(0.0, max_y)

        eps = 1.0
        invalid_x = x2 <= x1
        if invalid_x.any():
            x1 = torch.where(invalid_x, (x2 - eps).clamp(0.0, max_x), x1)
            x2 = torch.where(invalid_x, (x1 + eps).clamp(0.0, max_x), x2)
        invalid_y = y2 <= y1
        if invalid_y.any():
            y1 = torch.where(invalid_y, (y2 - eps).clamp(0.0, max_y), y1)
            y2 = torch.where(invalid_y, (y1 + eps).clamp(0.0, max_y), y2)

        batch_idx = batch_idx.to(dtype=boxes.dtype)
        return torch.stack([batch_idx, x1, y1, x2, y2], dim=1)

    def _roi_align_fallback(self, feat: torch.Tensor, rois: torch.Tensor) -> torch.Tensor:
        num_rois = rois.shape[0]
        if num_rois == 0:
            return feat.new_zeros((0, feat.shape[1], self.pool_size, self.pool_size))

        _, _, height, width = feat.shape
        max_x = max(width - 1, 1)
        max_y = max(height - 1, 1)
        output = feat.new_zeros((num_rois, feat.shape[1], self.pool_size, self.pool_size))

        for i in range(num_rois):
            b = int(rois[i, 0].item())
            x1, y1, x2, y2 = rois[i, 1:]
            xs = torch.linspace(x1, x2, self.pool_size, device=feat.device, dtype=feat.dtype)
            ys = torch.linspace(y1, y2, self.pool_size, device=feat.device, dtype=feat.dtype)
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
            grid_x = (grid_x / max_x) * 2.0 - 1.0
            grid_y = (grid_y / max_y) * 2.0 - 1.0
            grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
            sampled = F.grid_sample(
                feat[b:b + 1],
                grid,
                mode="bilinear",
                align_corners=True,
            )
            output[i] = sampled[0]
        return output


def _sanity_check():
    torch.manual_seed(0)
    encoder = ROIEncoder(hidden_dim=256, pool_size=7, num_layers=2, nhead=8, dropout=0.0, use_2d_pos=True)
    feat = torch.randn(2, 256, 32, 32)
    boxes = torch.rand(2, 5, 4)
    boxes[..., 2:] = boxes[..., 2:] * 0.5
    keep_mask = torch.rand(2, 5) > 0.5
    keep_mask[0, 0] = True
    roi_feats, indices = encoder(feat, boxes, keep_mask)
    print("roi_feats:", roi_feats.shape)
    print("indices:", indices.shape)


if __name__ == "__main__":
    _sanity_check()
