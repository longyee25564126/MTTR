# Copyright (c) Ruopeng Gao. All Rights Reserved.

from __future__ import annotations

from typing import Dict, List

import torch

from utils.box_ops import box_cxcywh_to_xywh
from utils.misc import distributed_device
from models.mttr.track_bank import TrackBank


class MTTRRuntimeTracker:
    def __init__(
        self,
        model,
        sequence_hw: tuple,
        max_tracks: int = 100,
        traj_len: int = 30,
        hidden_dim: int = 256,
        miss_tolerance: int = 30,
        det_thresh: float = 0.5,
        newborn_thresh: float = 0.5,
        area_thresh: int = 0,
        only_detr: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        self.model = model
        self.model.eval()

        self.dtype = dtype
        if self.dtype != torch.float32:
            if self.dtype == torch.float16:
                self.model.half()
            else:
                raise NotImplementedError(f"Unsupported dtype {self.dtype}.")

        self.max_tracks = max_tracks
        self.traj_len = traj_len
        self.hidden_dim = hidden_dim
        self.miss_tolerance = miss_tolerance
        self.det_thresh = det_thresh
        self.newborn_thresh = newborn_thresh
        self.area_thresh = area_thresh
        self.only_detr = only_detr

        self.bbox_unnorm = torch.tensor(
            [sequence_hw[1], sequence_hw[0], sequence_hw[1], sequence_hw[0]],
            dtype=dtype,
            device=distributed_device(),
        )

        self.track_bank = TrackBank(
            traj_len=traj_len,
            hidden_dim=hidden_dim,
            max_miss=miss_tolerance,
            device=distributed_device(),
            dtype=dtype,
        )
        self.next_id = 0
        self.current_track_results: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, image):
        device = image.tensors.device

        if self.only_detr:
            out = self.model(samples=image)
            return self._update_only_detr(out)

        active_labels = self.track_bank.get_active_labels()
        labels_use = sorted(active_labels)[: self.max_tracks]
        unused_labels = set(active_labels) - set(labels_use)
        for label in unused_labels:
            self.track_bank.push_miss(label)

        if len(labels_use) > 0:
            traj_batch, pad_mask = self.track_bank.export_batch(labels_use)
            track_queries = self.model.traj_encoder(traj_batch, pad_mask)
            track_queries_batch = track_queries.new_zeros((1, self.max_tracks, self.hidden_dim))
            track_queries_batch[0, : track_queries.shape[0]] = track_queries
        else:
            track_queries_batch = torch.zeros(
                (1, self.max_tracks, self.hidden_dim),
                device=device,
                dtype=self.dtype,
            )

        out = self.model(samples=image, track_queries=track_queries_batch, num_track_slots=self.max_tracks)
        logits = out["pred_logits"][0]
        boxes = out["pred_boxes"][0]
        scores = logits.sigmoid()
        scores, categories = torch.max(scores, dim=-1)

        area = boxes[:, 2] * self.bbox_unnorm[2] * boxes[:, 3] * self.bbox_unnorm[3]
        area_ok = area > self.area_thresh

        num_queries = boxes.shape[0]
        keep_mask = torch.zeros((1, num_queries), dtype=torch.bool, device=device)
        query_to_label: Dict[int, int] = {}
        det_query_idxs: List[int] = []
        det_ids: List[int] = []

        # Track slots.
        for i, label in enumerate(labels_use):
            if i >= self.max_tracks:
                break
            if scores[i].item() >= self.det_thresh and area_ok[i].item():
                keep_mask[0, i] = True
                det_query_idxs.append(i)
                det_ids.append(label)
            else:
                self.track_bank.push_miss(label)

        # New slots.
        for q in range(self.max_tracks, num_queries):
            if scores[q].item() >= self.newborn_thresh and area_ok[q].item():
                label = self.next_id
                self.next_id += 1
                query_to_label[q] = label
                keep_mask[0, q] = True
                det_query_idxs.append(q)
                det_ids.append(label)

        if keep_mask.any():
            roi_feats, roi_indices = self.model.roi_encoder(
                feat_highres=out["feat_highres"],
                boxes=out["pred_boxes"],
                keep_mask=keep_mask,
            )
            for k in range(roi_indices.shape[0]):
                q = int(roi_indices[k, 1].item())
                if q < self.max_tracks:
                    if q >= len(labels_use):
                        continue
                    label = labels_use[q]
                else:
                    label = query_to_label.get(q, None)
                    if label is None:
                        continue
                self.track_bank.ensure_track(label)
                self.track_bank.push_roi(label, roi_feats[k])

        self._build_results(det_query_idxs, det_ids, scores, categories, boxes)
        return

    def get_track_results(self):
        return self.current_track_results

    def _build_results(self, det_query_idxs, det_ids, scores, categories, boxes):
        device = boxes.device
        if len(det_query_idxs) == 0:
            self.current_track_results = {
                "score": torch.zeros((0,), device=device),
                "category": torch.zeros((0,), dtype=torch.int64, device=device),
                "bbox": torch.zeros((0, 4), device=device),
                "id": torch.zeros((0,), dtype=torch.int64, device=device),
            }
            return

        det_query_idxs_t = torch.tensor(det_query_idxs, device=device, dtype=torch.int64)
        ids = torch.tensor(det_ids, device=device, dtype=torch.int64)
        boxes_keep = boxes[det_query_idxs_t]
        self.current_track_results = {
            "score": scores[det_query_idxs_t],
            "category": categories[det_query_idxs_t],
            "bbox": box_cxcywh_to_xywh(boxes_keep) * self.bbox_unnorm,
            "id": ids,
        }

    def _update_only_detr(self, detr_out: dict):
        logits = detr_out["pred_logits"][0]
        boxes = detr_out["pred_boxes"][0]
        scores = logits.sigmoid()
        scores, categories = torch.max(scores, dim=-1)
        area = boxes[:, 2] * self.bbox_unnorm[2] * boxes[:, 3] * self.bbox_unnorm[3]
        activate = (scores > self.det_thresh) & (area > self.area_thresh)
        keep = torch.nonzero(activate, as_tuple=False).flatten()
        if keep.numel() == 0:
            self.current_track_results = {
                "score": torch.zeros((0,), device=boxes.device),
                "category": torch.zeros((0,), dtype=torch.int64, device=boxes.device),
                "bbox": torch.zeros((0, 4), device=boxes.device),
                "id": torch.zeros((0,), dtype=torch.int64, device=boxes.device),
            }
            return
        scores = scores[keep]
        categories = categories[keep]
        boxes = boxes[keep]
        ids = torch.arange(self.next_id, self.next_id + keep.numel(), device=boxes.device, dtype=torch.int64)
        self.next_id += keep.numel()
        self.current_track_results = {
            "score": scores,
            "category": categories,
            "bbox": box_cxcywh_to_xywh(boxes) * self.bbox_unnorm,
            "id": ids,
        }
        return
