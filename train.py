# Copyright (c) Ruopeng Gao. All Rights Reserved.

import os
import math
import torch
import einops
from accelerate import Accelerator
from accelerate.state import PartialState
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from collections import defaultdict
from torchvision.transforms import v2
from typing import Any, Generator, List

from models.deformable_detr.deformable_detr import build as build_deformable_detr
from structures.args import Args
from runtime_option import runtime_option
from utils.misc import yaml_to_dict, set_seed
from configs.util import load_super_config, update_config
from log.logger import Logger
from data import build_dataset
from data.naive_sampler import NaiveSampler
from data.util import collate_fn
from log.log import TPS, Metrics
from models.misc import save_checkpoint, load_checkpoint
from models.misc import get_model
from models.mttr import ROIEncoder, TrajEncoder, TrackBank
from utils.nested_tensor import NestedTensor
from submit_and_evaluate import submit_and_evaluate_one_model


def train_engine(config: dict):
    # Init some settings:
    assert "EXP_NAME" in config and config["EXP_NAME"] is not None, "Please set the experiment name."
    outputs_dir = config["OUTPUTS_DIR"] if config["OUTPUTS_DIR"] is not None \
        else os.path.join("./outputs/", config["EXP_NAME"])

    # Init Accelerator at beginning:
    accelerator = Accelerator()
    state = PartialState()
    # Also, we set the seed:
    set_seed(config["SEED"])
    # Set the sharing strategy (to avoid error: too many open files):
    torch.multiprocessing.set_sharing_strategy('file_system')   # if not, raise error: too many open files.

    # Init Logger:
    logger = Logger(
        logdir=os.path.join(outputs_dir, "train"),
        use_wandb=config["USE_WANDB"],
        config=config,
        exp_owner=config["EXP_OWNER"],
        exp_project=config["EXP_PROJECT"],
        exp_group=config["EXP_GROUP"],
        exp_name=config["EXP_NAME"],
    )
    logger.info(f"We init the logger at {logger.logdir}.")
    if config["USE_WANDB"] is False:
        logger.warning("The wandb is not used in this experiment.")
    logger.info(f"The distributed type is {state.distributed_type}.")
    logger.config(config=config)

    # Build training dataset:
    train_dataset = build_dataset(config=config)
    logger.dataset(train_dataset)
    # Build training data sampler:
    if "DATASET_WEIGHTS" in config:
        data_weights = defaultdict(lambda: defaultdict())
        for _ in range(len(config["DATASET_WEIGHTS"])):
            data_weights[config["DATASETS"][_]][config["DATASET_SPLITS"][_]] = config["DATASET_WEIGHTS"][_]
        data_weights = dict(data_weights)
    else:
        data_weights = None
    train_sampler = NaiveSampler(
        data_source=train_dataset,
        sample_steps=config["SAMPLE_STEPS"],
        sample_lengths=config["SAMPLE_LENGTHS"],
        sample_intervals=config["SAMPLE_INTERVALS"],
        length_per_iteration=config["LENGTH_PER_ITERATION"],
        data_weights=data_weights,
    )
    debug_sanity = config.get("DEBUG_SANITY", False)
    debug_max_iters = config.get("DEBUG_SANITY_ITERS", 5)
    if debug_sanity:
        debug_max_iters = max(1, min(int(debug_max_iters), 50))
        logger.warning(
            log=f"DEBUG_SANITY enabled: logging first {debug_max_iters} iterations."
        )

    # Build training data loader:
    batch_size = config["BATCH_SIZE"]
    train_dataloader = DataLoader(
        dataset=train_dataset,
        sampler=train_sampler,
        batch_size=batch_size,
        num_workers=config["NUM_WORKERS"],
        prefetch_factor=config["PREFETCH_FACTOR"] if config["NUM_WORKERS"] > 0 else None,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Init the training states:
    train_states = {
        "start_epoch": 0,
        "global_step": 0
    }

    # Build MTTR model (DETR-only) and attach encoders:
    model, detr_criterion = build_detr_only(config=config)
    model.roi_encoder = ROIEncoder(
        hidden_dim=config.get("DETR_HIDDEN_DIM", 256),
        pool_size=config.get("MTTR_ROI_POOL_SIZE", 7),
        num_layers=config.get("MTTR_ROI_NUM_LAYERS", 2),
        nhead=config.get("MTTR_ROI_NHEAD", 8),
        dropout=config.get("MTTR_ROI_DROPOUT", 0.0),
    )
    model.traj_encoder = TrajEncoder(
        hidden_dim=config.get("DETR_HIDDEN_DIM", 256),
        traj_len=config.get("MTTR_TRAJ_LEN", 30),
        num_layers=config.get("MTTR_TRAJ_NUM_LAYERS", 6),
        nhead=config.get("MTTR_TRAJ_NHEAD", 8),
        dim_feedforward=config.get("MTTR_TRAJ_FFN_DIM", 1024),
        dropout=config.get("MTTR_TRAJ_DROPOUT", 0.0),
        norm_first=config.get("MTTR_TRAJ_NORM_FIRST", True),
    )
    if config.get("MTTR_MAX_TRACKS", 100) > config.get("DETR_NUM_QUERIES", 300):
        raise ValueError("MTTR_MAX_TRACKS must be <= DETR_NUM_QUERIES.")
    # Load the pre-trained DETR:
    load_detr_pretrain_for_detr(
        model=model,
        pretrain_path=config["DETR_PRETRAIN"],
        num_classes=config["NUM_CLASSES"],
        default_class_idx=config.get("DETR_DEFAULT_CLASS_IDX", None),
    )
    logger.success(
        log=f"Load the pre-trained DETR from '{config['DETR_PRETRAIN']}'. "
    )

    # Build Optimizer:
    if config["DETR_NUM_TRAIN_FRAMES"] == 0:
        for n, p in model.named_parameters():
            if "detr" in n:
                p.requires_grad = False     # Freeze DETR params if configured.
    param_groups = get_param_groups(model, config)
    optimizer = AdamW(
        params=param_groups,
        lr=config["LR"],
        weight_decay=config["WEIGHT_DECAY"],
    )
    scheduler = MultiStepLR(
        optimizer=optimizer,
        milestones=config["SCHEDULER_MILESTONES"],
        gamma=config["SCHEDULER_GAMMA"],
    )

    # Other infos:
    only_detr = False

    # Resuming:
    if config["RESUME_MODEL"] is not None:
        load_checkpoint(
            model=model,
            path=config["RESUME_MODEL"],
            optimizer=optimizer if config["RESUME_OPTIMIZER"] else None,
            scheduler=scheduler if config["RESUME_SCHEDULER"] else None,
            states=train_states,
        )
        # Different processing on scheduler:
        if config["RESUME_SCHEDULER"]:
            scheduler.step()
        else:
            for _ in range(0, train_states["start_epoch"]):
                scheduler.step()
        logger.success(
            log=f"Resume the model from '{config['RESUME_MODEL']}', "
                f"optimizer={config['RESUME_OPTIMIZER']}, "
                f"scheduler={config['RESUME_SCHEDULER']}, "
                f"states={train_states}. "
                f"Start from epoch {train_states['start_epoch']}, step {train_states['global_step']}."
        )

    train_dataloader, model, optimizer = accelerator.prepare(
        train_dataloader, model, optimizer,
        # device_placement=[False]        # whether to place the data on the device
    )

    for epoch in range(train_states["start_epoch"], config["EPOCHS"]):
        logger.info(log=f"Start training epoch {epoch}.")
        epoch_start_timestamp = TPS.timestamp()
        # Prepare the sampler for the current epoch:
        train_sampler.prepare_for_epoch(epoch=epoch)
        # Train one epoch:
        train_metrics = train_one_epoch(
            accelerator=accelerator,
            logger=logger,
            states=train_states,
            epoch=epoch,
            dataloader=train_dataloader,
            model=model,
            detr_criterion=detr_criterion,
            optimizer=optimizer,
            config=config,
            lr_warmup_epochs=config["LR_WARMUP_EPOCHS"],
            lr_warmup_tgt_lr=config["LR"],
            accumulate_steps=config["ACCUMULATE_STEPS"],
            separate_clip_norm=config.get("SEPARATE_CLIP_NORM", True),
            max_clip_norm=config.get("MAX_CLIP_NORM", 0.1),
            use_accelerate_clip_norm=config.get("USE_ACCELERATE_CLIP_NORM", True),
            # For multi last checkpoints:
            outputs_dir=outputs_dir,
            is_last_epochs=(epoch == config["EPOCHS"] - 1),
            multi_last_checkpoints=config["MULTI_LAST_CHECKPOINTS"],
        )

        # Get learning rate:
        lr = optimizer.state_dict()["param_groups"][-1]["lr"]
        train_metrics["lr"].update(lr)
        train_metrics["lr"].sync()
        time_per_epoch = TPS.format(TPS.timestamp() - epoch_start_timestamp)
        logger.metrics(
            log=f"[Finish epoch: {epoch}] [Time: {time_per_epoch}] ",
            metrics=train_metrics,
            fmt="{global_average:.4f}",
            statistic="global_average",
            global_step=train_states["global_step"],
            prefix="epoch",
            x_axis_step=epoch,
            x_axis_name="epoch",
        )

        # Save checkpoint:
        if (epoch + 1) % config["SAVE_CHECKPOINT_PER_EPOCH"] == 0:
            save_checkpoint(
                model=model,
                path=os.path.join(outputs_dir, f"checkpoint_{epoch}.pth"),
                states=train_states,
                optimizer=optimizer,
                scheduler=scheduler,
                only_detr=only_detr,
            )
            if config["INFERENCE_DATASET"] is not None:
                assert config["INFERENCE_SPLIT"] is not None, f"Please set the INFERENCE_SPLIT for inference."
                eval_metrics = submit_and_evaluate_one_model(
                    is_evaluate=True,
                    accelerator=accelerator,
                    state=state,
                    logger=logger,
                    model=model,
                    data_root=config["DATA_ROOT"],
                    dataset=config["INFERENCE_DATASET"],
                    data_split=config["INFERENCE_SPLIT"],
                    outputs_dir=os.path.join(outputs_dir, "train", "eval_during_train", f"epoch_{epoch}"),
                    image_max_longer=config["INFERENCE_MAX_LONGER"],
                    size_divisibility=config.get("SIZE_DIVISIBILITY", 0),
                    miss_tolerance=config.get("MISS_TOLERANCE", 30),
                    use_sigmoid=config["USE_FOCAL_LOSS"] if "USE_FOCAL_LOSS" in config else False,
                    assignment_protocol=config["ASSIGNMENT_PROTOCOL"] if "ASSIGNMENT_PROTOCOL" in config else "hungarian",
                    det_thresh=config.get("DET_THRESH", 0.5),
                    newborn_thresh=config.get("NEWBORN_THRESH", 0.5),
                    id_thresh=config.get("ID_THRESH", 0.3),
                    area_thresh=config.get("AREA_THRESH", 0),
                    inference_only_detr=config["INFERENCE_ONLY_DETR"] if config["INFERENCE_ONLY_DETR"] is not None
                    else config["ONLY_DETR"],
                    mttr_max_tracks=config.get("MTTR_MAX_TRACKS", 100),
                    mttr_traj_len=config.get("MTTR_TRAJ_LEN", 30),
                    mttr_max_miss=config.get("MTTR_MAX_MISS", config.get("MISS_TOLERANCE", 30)),
                    mttr_hidden_dim=config.get("DETR_HIDDEN_DIM", 256),
                )
                eval_metrics.sync()
                logger.metrics(
                    log=f"[Eval epoch: {epoch}] ",
                    metrics=eval_metrics,
                    fmt="{global_average:.4f}",
                    statistic="global_average",
                    global_step=train_states["global_step"],
                    prefix="epoch",
                    x_axis_step=epoch,
                    x_axis_name="epoch",
                )

        logger.success(log=f"Finish training epoch {epoch}.")
        # Prepare for next step:
        scheduler.step()
    pass


def train_one_epoch(
        # Infos:
        accelerator: Accelerator,
        logger: Logger,
        states: dict,
        epoch: int,
        dataloader: DataLoader,
        model,
        detr_criterion,
        optimizer,
        config: dict,
        lr_warmup_epochs: int,
        lr_warmup_tgt_lr: float,
        accumulate_steps: int = 1,
        separate_clip_norm: bool = True,
        max_clip_norm: float = 0.1,
        use_accelerate_clip_norm: bool = True,
        logging_interval: int = 20,
        # For multi last checkpoints:
        outputs_dir: str = None,
        is_last_epochs: bool = False,
        multi_last_checkpoints: int = 0,
):
    current_last_checkpoint_idx = 0

    model.train()
    model.roi_encoder.train()
    model.traj_encoder.train()
    tps = TPS()
    metrics = Metrics()
    optimizer.zero_grad()
    step_timestamp = tps.timestamp()
    device = accelerator.device
    detr_weight_dict = detr_criterion.weight_dict

    model_without_ddp = get_model(model)
    detr_params = []
    other_params = []
    for name, param in model_without_ddp.named_parameters():
        if "detr" in name:
            detr_params.append(param)
        else:
            other_params.append(param)

    hidden_dim = config.get("DETR_HIDDEN_DIM", 256)
    max_tracks = min(config.get("MTTR_MAX_TRACKS", 100), config.get("DETR_NUM_QUERIES", 300))
    traj_len = config.get("MTTR_TRAJ_LEN", 30)
    max_miss = config.get("MTTR_MAX_MISS", 30)
    teacher_forcing = config.get("MTTR_TEACHER_FORCING", True)

    debug_sanity = config.get("DEBUG_SANITY", False)
    debug_assert = config.get("DEBUG_ASSERT", False)
    debug_max_iters = None
    if debug_sanity:
        debug_max_iters = max(1, min(int(config.get("DEBUG_SANITY_ITERS", 5)), 50))

    def _debug_log_lookup(tag: str, masks_label: torch.Tensor, annidx_label: torch.Tensor, enabled: bool):
        if not enabled:
            return
        num_labels = int(masks_label.numel())
        num_visible = int((~masks_label).sum().item()) if num_labels > 0 else 0
        num_missing = int(masks_label.sum().item()) if num_labels > 0 else 0
        num_ann_valid = int((annidx_label >= 0).sum().item()) if num_labels > 0 else 0
        samples = []
        if num_labels > 0:
            perm = torch.randperm(num_labels, device=masks_label.device)[:min(3, num_labels)]
            for idx in perm.tolist():
                samples.append((int(idx), bool(masks_label[idx].item()), int(annidx_label[idx].item())))
        logger.info(
            log=(
                f"[DEBUG_SANITY] {tag} valid_labels={num_labels} visible={num_visible} "
                f"missing={num_missing} ann_valid={num_ann_valid} sample={samples}"
            )
        )

    def _debug_log_gt_to_label(tag: str, ann: dict, masks_label: torch.Tensor, annidx_label: torch.Tensor, enabled: bool):
        if not enabled:
            return
        if "id" not in ann:
            logger.info(log=f"[DEBUG_SANITY] {tag} gt2label=missing_ann_id")
            return
        ann_ids = ann["id"]
        if torch.is_tensor(ann_ids):
            ann_ids = ann_ids.tolist()
        num_gt = len(ann_ids)
        label_for_ann = [-1 for _ in range(num_gt)]
        mask_for_ann = [None for _ in range(num_gt)]
        num_labels = int(annidx_label.numel())
        for label in range(num_labels):
            ann_idx = int(annidx_label[label].item())
            if ann_idx < 0 or ann_idx >= num_gt:
                continue
            if label_for_ann[ann_idx] != -1 and debug_assert:
                raise AssertionError(
                    f"ann_idx {ann_idx} mapped to multiple labels: "
                    f"{label_for_ann[ann_idx]} and {label}"
                )
            label_for_ann[ann_idx] = label
            mask_for_ann[ann_idx] = bool(masks_label[label].item())
        mapping = [(int(ann_ids[i]), int(label_for_ann[i]), mask_for_ann[i]) for i in range(num_gt)]
        logger.info(log=f"[DEBUG_SANITY] {tag} gt2label={mapping}")



    for step, samples in enumerate(dataloader):
        images, annotations, metas = samples["images"], samples["annotations"], samples["metas"]
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        images.tensors = v2.functional.to_dtype(images.tensors, dtype=torch.float32, scale=True)
        images.tensors = v2.functional.normalize(images.tensors, mean=mean, std=std)
        images.tensors = images.tensors * (~images.mask[:, :, None, ...]).to(torch.float32)
        images.tensors = images.tensors.contiguous()

        if epoch < lr_warmup_epochs:
            lr_warmup(
                optimizer=optimizer,
                epoch=epoch, curr_iter=step, tgt_lr=lr_warmup_tgt_lr,
                warmup_epochs=lr_warmup_epochs, num_iter_per_epoch=len(dataloader),
            )

        _B, _T = len(annotations), len(annotations[0])
        if _T == 1:
            t = 0
        else:
            t = int(torch.randint(1, _T, (1,), device=device).item())

        group_indices = []
        for b in range(_B):
            g_total = annotations[b][0]["trajectory_id_labels"].shape[0]
            g = int(torch.randint(0, g_total, (1,), device=device).item())
            group_indices.append(g)

        debug_log_enabled = debug_sanity and (debug_max_iters is None or step < debug_max_iters)

        if debug_log_enabled:
            step_meta = []
            for b in range(_B):
                meta_list = metas[b]
                dataset = meta_list[0].get("dataset", "unknown")
                split = meta_list[0].get("split", "")
                sequence = meta_list[0].get("sequence", "")
                frame_idxs = [m.get("frame_idx", None) for m in meta_list]
                train_frame_idx = frame_idxs[t] if t < len(frame_idxs) else None
                prefix_frame_idxs = frame_idxs[:t]
                step_meta.append({
                    "dataset": dataset,
                    "split": split,
                    "sequence": sequence,
                    "frames": frame_idxs,
                    "train_frame": train_frame_idx,
                    "prefix": prefix_frame_idxs,
                })
            logger.info(log=f"[DEBUG_SANITY] step={step} data={step_meta}")

        track_banks = [
            TrackBank(
                traj_len=traj_len,
                hidden_dim=hidden_dim,
                max_miss=max_miss,
                device=device,
                dtype=images.tensors.dtype,
            )
            for _ in range(_B)
        ]

        if t > 0 and teacher_forcing:
            prefix_tensors = images.tensors[:, :t]
            prefix_masks = images.mask[:, :t]
            prefix_tensors = einops.rearrange(prefix_tensors, "b t c h w -> (b t) c h w").contiguous()
            prefix_masks = einops.rearrange(prefix_masks, "b t h w -> (b t) h w").contiguous()
            prefix_nested = NestedTensor(prefix_tensors, prefix_masks)
            with torch.no_grad():
                out_prefix = model(samples=prefix_nested)
            feat_highres = out_prefix["feat_highres"]
            feat_highres = einops.rearrange(feat_highres, "(b t) c h w -> b t c h w", b=_B, t=t)

            for tau in range(t):
                feat_tau = feat_highres[:, tau]
                nmax = annotations[0][tau]["trajectory_id_labels"].shape[-1]
                boxes_override = feat_tau.new_zeros((_B, nmax, 4))
                keep_mask = torch.zeros((_B, nmax), dtype=torch.bool, device=device)

                for b in range(_B):
                    masks_label, annidx_label = _build_label_lookup(
                        ann=annotations[b][tau],
                        group_idx=group_indices[b],
                        device=device,
                        time_idx=tau,
                        debug_assert=debug_assert,
                    )
                    if debug_log_enabled and b == 0:
                        _debug_log_lookup(
                            tag=f"lookup prefix tau={tau} g={group_indices[b]}",
                            masks_label=masks_label,
                            annidx_label=annidx_label,
                            enabled=debug_log_enabled,
                        )
                        _debug_log_gt_to_label(
                            tag=f"gt2label prefix tau={tau} g={group_indices[b]}",
                            ann=annotations[b][tau],
                            masks_label=masks_label,
                            annidx_label=annidx_label,
                            enabled=debug_log_enabled,
                        )
                    visible = (~masks_label) & (annidx_label >= 0)
                    active_labels = track_banks[b].get_active_labels()
                    for label in active_labels:
                        if label >= visible.numel() or not visible[label].item():
                            track_banks[b].push_miss(label)

                    if visible.numel() > 0:
                        ann_bbox = annotations[b][tau]["bbox"].to(device=device, dtype=feat_tau.dtype)
                        visible_idx = torch.nonzero(visible, as_tuple=False).flatten()
                        for label in visible_idx.tolist():
                            ann_idx = int(annidx_label[label].item())
                            boxes_override[b, label] = ann_bbox[ann_idx]
                            keep_mask[b, label] = True

                if keep_mask.any():
                    with torch.no_grad():
                        roi_feats, roi_indices = model.roi_encoder(
                            feat_highres=feat_tau,
                            boxes=boxes_override,
                            keep_mask=keep_mask,
                        )
                    roi_feats = roi_feats.detach()
                    for k in range(roi_indices.shape[0]):
                        b = int(roi_indices[k, 0].item())
                        label = int(roi_indices[k, 1].item())
                        track_banks[b].ensure_track(label)
                        track_banks[b].push_roi(label, roi_feats[k])

        track_queries_batch = images.tensors.new_zeros((_B, max_tracks, hidden_dim))
        slot_label_ids = torch.full((_B, max_tracks), -1, device=device, dtype=torch.long)
        k_per_b = [0 for _ in range(_B)]
        for b in range(_B):
            active_labels = track_banks[b].get_active_labels()
            labels_use = sorted(active_labels)[:max_tracks]
            k_per_b[b] = len(labels_use)
            if len(labels_use) == 0:
                continue
            traj_batch, pad_mask_batch = track_banks[b].export_batch(labels_use)
            with accelerator.autocast():
                queries = model.traj_encoder(traj_batch, pad_mask_batch)
            track_queries_batch[b, :len(labels_use)] = queries
            slot_label_ids[b, :len(labels_use)] = torch.tensor(labels_use, device=device, dtype=torch.long)

        train_frame = NestedTensor(images.tensors[:, t], images.mask[:, t])
        with accelerator.autocast():
            out_t = model(samples=train_frame, track_queries=track_queries_batch)
        pred_logits = out_t["pred_logits"].float()
        pred_boxes = out_t["pred_boxes"].float()

        logits_track = pred_logits[:, :max_tracks, :]
        boxes_track = pred_boxes[:, :max_tracks, :]

        targets_track = []
        indices_track = []
        total_boxes = 0
        num_pos_per_b = [0 for _ in range(_B)]
        num_gt_per_b = [0 for _ in range(_B)]
        for b in range(_B):
            masks_label, annidx_label = _build_label_lookup(
                ann=annotations[b][t],
                group_idx=group_indices[b],
                device=device,
                time_idx=t,
                debug_assert=debug_assert,
            )
            if debug_log_enabled and b == 0:
                _debug_log_lookup(
                    tag=f"lookup t={t} g={group_indices[b]}",
                    masks_label=masks_label,
                    annidx_label=annidx_label,
                    enabled=debug_log_enabled,
                )
                _debug_log_gt_to_label(
                    tag=f"gt2label t={t} g={group_indices[b]}",
                    ann=annotations[b][t],
                    masks_label=masks_label,
                    annidx_label=annidx_label,
                    enabled=debug_log_enabled,
                )
            ann = annotations[b][t]
            ann_bbox = ann["bbox"].to(device=device, dtype=pred_boxes.dtype)
            ann_category = ann["category"].to(device=device, dtype=torch.int64)
            if debug_assert:
                if ann_bbox.ndim != 2 or ann_bbox.shape[1] != 4:
                    raise AssertionError(f"ann_bbox shape invalid: {ann_bbox.shape}")
            num_gt_per_b[b] = int(ann_bbox.shape[0])
            pos_mask = (~masks_label) & (annidx_label >= 0)
            src_idx = []
            tgt_labels = []
            tgt_boxes = []
            for i in range(max_tracks):
                label = int(slot_label_ids[b, i].item())
                if label < 0 or label >= pos_mask.numel():
                    continue
                if pos_mask[label].item():
                    ann_idx = int(annidx_label[label].item())
                    if debug_assert:
                        if ann_idx < 0 or ann_idx >= ann_bbox.shape[0]:
                            raise AssertionError(
                                f"ann_idx {ann_idx} out of range [0, {ann_bbox.shape[0]}) for label {label}"
                            )
                    src_idx.append(i)
                    tgt_labels.append(ann_category[ann_idx])
                    tgt_boxes.append(ann_bbox[ann_idx])

            num_pos_per_b[b] = len(src_idx)
            if len(tgt_labels) == 0:
                targets_track.append({
                    "labels": ann_category.new_zeros((0,), dtype=torch.int64),
                    "boxes": ann_bbox.new_zeros((0, 4)),
                })
                empty = torch.zeros((0,), dtype=torch.int64, device=device)
                indices_track.append((empty, empty))
            else:
                targets_track.append({
                    "labels": torch.stack(tgt_labels, dim=0),
                    "boxes": torch.stack(tgt_boxes, dim=0),
                })
                src_idx_tensor = torch.tensor(src_idx, dtype=torch.int64, device=device)
                tgt_idx_tensor = torch.arange(len(src_idx), dtype=torch.int64, device=device)
                indices_track.append((src_idx_tensor, tgt_idx_tensor))
                total_boxes += len(src_idx)

        num_boxes = max(1, total_boxes)

        def _compute_track_losses(pred_logits, pred_boxes, add_suffix: str | None = None):
            pred_logits = pred_logits.float()
            pred_boxes = pred_boxes.float()
            outputs_track = {"pred_logits": pred_logits, "pred_boxes": pred_boxes}
            losses = detr_criterion.loss_labels(
                outputs_track,
                targets_track,
                indices_track,
                num_boxes,
                log=True,
            )
            if total_boxes > 0:
                losses.update(detr_criterion.loss_boxes(
                    outputs_track,
                    targets_track,
                    indices_track,
                    num_boxes,
                ))
            else:
                zero = pred_boxes.new_tensor(0.0)
                losses.update({"loss_bbox": zero, "loss_giou": zero})
            losses.update(detr_criterion.loss_cardinality(
                outputs_track,
                targets_track,
                indices_track,
                num_boxes,
            ))
            if add_suffix is None:
                return losses
            return {f"{k}_{add_suffix}": v for k, v in losses.items()}

        loss_dict = _compute_track_losses(logits_track, boxes_track)

        if "aux_outputs" in out_t:
            for i, aux in enumerate(out_t["aux_outputs"]):
                aux_logits = aux["pred_logits"][:, :max_tracks, :]
                aux_boxes = aux["pred_boxes"][:, :max_tracks, :]
                loss_dict.update(_compute_track_losses(aux_logits, aux_boxes, add_suffix=str(i)))

        with accelerator.autocast():
            loss = sum(
                loss_dict[k] * detr_weight_dict[k]
                for k in loss_dict.keys()
                if k in detr_weight_dict
            )
            loss_total_value = loss.item()
            if debug_log_enabled:
                loss_ce_val = loss_dict.get("loss_ce", loss.new_tensor(0.0)).item()
                loss_bbox_val = loss_dict.get("loss_bbox", loss.new_tensor(0.0)).item()
                loss_giou_val = loss_dict.get("loss_giou", loss.new_tensor(0.0)).item()
                logger.info(
                    log=(
                        f"[DEBUG_SANITY] step={step} t={t} g={group_indices} "
                        f"K={k_per_b} num_pos={num_pos_per_b} num_gt={num_gt_per_b} "
                        f"num_pos_total={total_boxes} "
                        f"loss_ce={loss_ce_val:.4f} loss_bbox={loss_bbox_val:.4f} "
                        f"loss_giou={loss_giou_val:.4f} loss_total={loss_total_value:.4f} "
                        f"pred_logits_dtype={pred_logits.dtype} pred_boxes_dtype={pred_boxes.dtype}"
                    )
                )
            metrics.update(name="loss", value=loss_total_value)
            for key in loss_dict.keys():
                if key.startswith((
                    "loss_ce",
                    "loss_bbox",
                    "loss_giou",
                    "class_error",
                    "cardinality_error",
                )):
                    metrics.update(name=key, value=loss_dict[key].item())
            loss = loss / accumulate_steps
            accelerator.backward(loss)
            if (step + 1) % accumulate_steps == 0:
                if use_accelerate_clip_norm:
                    if separate_clip_norm:
                        detr_grad_norm = accelerator.clip_grad_norm_(detr_params, max_norm=max_clip_norm)
                        other_grad_norm = accelerator.clip_grad_norm_(other_params, max_norm=max_clip_norm)
                    else:
                        detr_grad_norm = other_grad_norm = accelerator.clip_grad_norm_(model.parameters(), max_norm=max_clip_norm)
                else:
                    if separate_clip_norm:
                        accelerator.unscale_gradients()
                        detr_grad_norm = torch.nn.utils.clip_grad_norm_(detr_params, max_clip_norm)
                        other_grad_norm = torch.nn.utils.clip_grad_norm_(other_params, max_clip_norm)
                    else:
                        accelerator.unscale_gradients()
                        detr_grad_norm = other_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_clip_norm)
                metrics.update(name="detr_grad_norm", value=detr_grad_norm.item())
                metrics.update(name="other_grad_norm", value=other_grad_norm.item())
                optimizer.step()
                optimizer.zero_grad()

        tps.update(tps=tps.timestamp() - step_timestamp)
        step_timestamp = tps.timestamp()
        if step % logging_interval == 0:
            _lr = optimizer.state_dict()["param_groups"][-1]["lr"]
            torch.cuda.synchronize()
            _cuda_memory = torch.cuda.max_memory_allocated(device) / 1024 / 1024
            _cuda_memory = torch.tensor([_cuda_memory], device=device)
            _gathered_cuda_memory = accelerator.gather(_cuda_memory)
            _max_cuda_memory = _gathered_cuda_memory.max().item()
            accelerator.wait_for_everyone()
            metrics["lr"].clear()
            metrics["max_cuda_mem(MB)"].clear()
            metrics.update(name="lr", value=_lr)
            metrics.update(name="max_cuda_mem(MB)", value=_max_cuda_memory)
            metrics.sync()
            eta = tps.eta(total_steps=len(dataloader), current_steps=step)
            logger.metrics(
                log=f"[Epoch: {epoch}] [{step}/{len(dataloader)}] "
                    f"[tps: {tps.average:.2f}s] [eta: {TPS.format(eta)}] ",
                metrics=metrics,
                global_step=states["global_step"],
            )
        if is_last_epochs and multi_last_checkpoints > 0:
            if (step + 1) == int(math.ceil((len(dataloader) / multi_last_checkpoints) * (current_last_checkpoint_idx + 1))):
                _dir = os.path.join(outputs_dir, "multi_last_checkpoints")
                os.makedirs(_dir, exist_ok=True)
                save_checkpoint(
                    model=model,
                    path=os.path.join(_dir, f"last_checkpoint_{current_last_checkpoint_idx}.pth"),
                    states=states,
                    optimizer=None,
                    scheduler=None,
                    only_detr=False,
                )
                logger.info(
                    log=f"Save the last checkpoint {current_last_checkpoint_idx} at step {step}."
                )
                current_last_checkpoint_idx += 1


        states["global_step"] += 1
    states["start_epoch"] += 1
    return metrics


def build_detr_only(config: dict):
    detr_args = Args()
    detr_args.backbone = config["BACKBONE"]
    detr_args.lr_backbone = config["LR"] * config["LR_BACKBONE_SCALE"]
    detr_args.dilation = config["DILATION"]
    detr_args.num_classes = config["NUM_CLASSES"]
    detr_args.device = config["DEVICE"]
    detr_args.num_queries = config["DETR_NUM_QUERIES"]
    detr_args.num_feature_levels = config["DETR_NUM_FEATURE_LEVELS"]
    detr_args.aux_loss = config["DETR_AUX_LOSS"]
    detr_args.with_box_refine = config["DETR_WITH_BOX_REFINE"]
    detr_args.two_stage = config["DETR_TWO_STAGE"]
    detr_args.hidden_dim = config["DETR_HIDDEN_DIM"]
    detr_args.masks = config["DETR_MASKS"]
    detr_args.position_embedding = config["DETR_POSITION_EMBEDDING"]
    detr_args.nheads = config["DETR_NUM_HEADS"]
    detr_args.enc_layers = config["DETR_ENC_LAYERS"]
    detr_args.dec_layers = config["DETR_DEC_LAYERS"]
    detr_args.dim_feedforward = config["DETR_DIM_FEEDFORWARD"]
    detr_args.dropout = config["DETR_DROPOUT"]
    detr_args.dec_n_points = config["DETR_DEC_N_POINTS"]
    detr_args.enc_n_points = config["DETR_ENC_N_POINTS"]
    detr_args.cls_loss_coef = config["DETR_CLS_LOSS_COEF"]
    detr_args.bbox_loss_coef = config["DETR_BBOX_LOSS_COEF"]
    detr_args.giou_loss_coef = config["DETR_GIOU_LOSS_COEF"]
    detr_args.focal_alpha = config["DETR_FOCAL_ALPHA"]
    detr_args.set_cost_class = config["DETR_SET_COST_CLASS"]
    detr_args.set_cost_bbox = config["DETR_SET_COST_BBOX"]
    detr_args.set_cost_giou = config["DETR_SET_COST_GIOU"]
    model, criterion, _ = build_deformable_detr(args=detr_args)
    return model, criterion


def load_detr_pretrain_for_detr(model, pretrain_path: str, num_classes: int | None, default_class_idx: int | None = None):
    pretrain_model = torch.load(pretrain_path, map_location=lambda storage, loc: storage, weights_only=False)
    pretrain_state_dict = pretrain_model["model"]
    model_state_dict = model.state_dict()
    transfer_state = {}
    for k, v in pretrain_state_dict.items():
        if "class_embed" in k:
            if num_classes is None:
                num_classes = len(v)
            if len(v) == 91:
                if num_classes == 1:
                    if default_class_idx is None:
                        v = v[1:2]
                    else:
                        v = v[default_class_idx:default_class_idx + 1]
                else:
                    raise NotImplementedError(f"Do not support detr pretrain loading for num_classes={num_classes}")
            elif num_classes == len(v):
                pass
            else:
                raise NotImplementedError(
                    f"Pretrained detr has a class head for {len(v)} classes, "
                    f"we do not support this pretrained model."
                )
        if "label_enc" in k:
            if len(v) != len(model_state_dict.get(k, v)):
                if len(model_state_dict.get(k, v)) == 2:
                    v = torch.cat((v[1:2], v[91:92]), dim=0)
                else:
                    raise NotImplementedError(f"Do not implement the pretrain loading processing for num_classes={num_classes}")
        if k in model_state_dict:
            transfer_state[k] = v

    model_state_dict.update(transfer_state)
    model.load_state_dict(state_dict=model_state_dict, strict=True)
    return


def _build_label_lookup(ann: dict, group_idx: int, device: torch.device, time_idx: int | None = None, debug_assert: bool = False):
    id_labels = ann["trajectory_id_labels"].to(device)
    id_masks = ann["trajectory_id_masks"].to(device)
    ann_idxs = ann["trajectory_ann_idxs"].to(device)

    def _select_time(tensor: torch.Tensor, name: str) -> torch.Tensor:
        if tensor.dim() == 3:
            tensor = tensor[group_idx]
            if tensor.dim() != 2:
                raise ValueError(f"{name} must have shape (G, T, N).")
            t_dim = tensor.shape[0]
            if t_dim == 1:
                return tensor[0]
            if time_idx is None:
                raise ValueError(f"{name} has T={t_dim} but time_idx is None.")
            if time_idx < 0 or time_idx >= t_dim:
                raise ValueError(f"time_idx {time_idx} out of range for {name} with T={t_dim}.")
            return tensor[time_idx]
        if tensor.dim() == 2:
            return tensor[group_idx]
        raise ValueError(f"{name} must have shape (G, T, N) or (G, N).")

    id_labels = _select_time(id_labels, "trajectory_id_labels")
    id_masks = _select_time(id_masks, "trajectory_id_masks")
    ann_idxs = _select_time(ann_idxs, "trajectory_ann_idxs")
    valid = id_labels >= 0
    if valid.sum().item() == 0:
        empty_mask = id_masks.new_zeros((0,), dtype=torch.bool)
        empty_idx = ann_idxs.new_zeros((0,), dtype=torch.long)
        return empty_mask, empty_idx
    perm_v = id_labels[valid]
    if debug_assert:
        unique_count = torch.unique(perm_v).numel()
        if unique_count != perm_v.numel():
            raise AssertionError(
                f"Duplicate label indices in trajectory_id_labels at time_idx={time_idx}, group={group_idx}."
            )
    inv = torch.argsort(perm_v)
    masks_base = id_masks[valid]
    annidx_base = ann_idxs[valid]
    masks_label = masks_base[inv]
    annidx_label = annidx_base[inv]
    return masks_label, annidx_label


def get_param_groups(model, config) -> list[dict]:
    def _match_names(_name, _key_names):
        for _k in _key_names:
            if _k in _name:
                return True
        return False

    # Keywords:
    backbone_names = config["LR_BACKBONE_NAMES"]
    linear_proj_names = config["LR_LINEAR_PROJ_NAMES"]
    dictionary_names = config["LR_DICTIONARY_NAMES"]
    pass
    # Param groups:
    param_groups = [
        {
            "params": [p for n, p in model.named_parameters() if _match_names(n, backbone_names) and p.requires_grad],
            "lr_scale": config["LR_BACKBONE_SCALE"],
            "lr": config["LR"] * config["LR_BACKBONE_SCALE"]
        },
        {
            "params": [p for n, p in model.named_parameters() if _match_names(n, linear_proj_names) and p.requires_grad],
            "lr_scale": config["LR_LINEAR_PROJ_SCALE"],
            "lr": config["LR"] * config["LR_LINEAR_PROJ_SCALE"]
        },
        {
            "params": [p for n, p in model.named_parameters() if _match_names(n, dictionary_names) and p.requires_grad],
            "lr_scale": config["LR_DICTIONARY_SCALE"],
            "lr": config["LR"] * config["LR_DICTIONARY_SCALE"]
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if not _match_names(n, backbone_names)
                       and not _match_names(n, linear_proj_names)
                       and not _match_names(n, dictionary_names)
                       and p.requires_grad],
        }
    ]
    return param_groups


def lr_warmup(optimizer, epoch: int, curr_iter: int, tgt_lr: float, warmup_epochs: int, num_iter_per_epoch: int):
    # min_lr = 1e-8
    total_warmup_iters = warmup_epochs * num_iter_per_epoch
    current_lr_ratio = (epoch * num_iter_per_epoch + curr_iter + 1) / total_warmup_iters
    current_lr = tgt_lr * current_lr_ratio
    for param_grop in optimizer.param_groups:
        if "lr_scale" in param_grop:
            param_grop["lr"] = current_lr * param_grop["lr_scale"]
        else:
            param_grop["lr"] = current_lr
        pass
    return


def annotations_to_flatten_detr_targets(annotations: list, device):
    """
    Args:
        annotations: annotations from the dataloader.
        device: move the targets to the device.

    Returns:
        A list of targets for the DETR model supervision, len=(B*T).
    """
    targets = []
    for annotation in annotations:      # scan by batch
        for ann in annotation:          # scan by frame
            targets.append(
                {
                    "boxes": ann["bbox"].to(device),
                    "labels": ann["category"].to(device),
                }
            )
    return targets


def nested_tensor_index_select(nested_tensor: NestedTensor, dim: int, index: torch.Tensor):
    tensors, mask = nested_tensor.decompose()
    _device = tensors.device
    index = index.to(_device)
    selected_tensors = torch.index_select(input=tensors, dim=dim, index=index).contiguous()
    selected_mask = torch.index_select(input=mask, dim=dim, index=index).contiguous()
    return NestedTensor(tensors=selected_tensors, mask=selected_mask)


def batch_iterator(batch_size: int, *args) -> Generator[List[Any], None, None]:
    assert len(args) > 0 and all(
        len(a) == len(args[0]) for a in args
    ), "Batched iteration must have inputs of all the same size."
    n_batches = len(args[0]) // batch_size + int(len(args[0]) % batch_size != 0)
    for b in range(n_batches):
        yield [arg[b * batch_size: (b + 1) * batch_size] for arg in args]


def tensor_dict_cat(tensor_dict1, tensor_dict2, dim=0):
    if tensor_dict1 is None or tensor_dict2 is None:
        assert tensor_dict1 is not None or tensor_dict2 is not None, "One of the tensor dict should be not None."
        return tensor_dict1 if tensor_dict2 is None else tensor_dict2
    else:
        res_tensor_dict = defaultdict()
        for k in tensor_dict1.keys():
            if isinstance(tensor_dict1[k], torch.Tensor):
                res_tensor_dict[k] = torch.cat([tensor_dict1[k], tensor_dict2[k]], dim=dim)
            elif isinstance(tensor_dict1[k], dict):
                res_tensor_dict[k] = tensor_dict_cat(tensor_dict1[k], tensor_dict2[k], dim=dim)
            elif isinstance(tensor_dict1[k], list):
                assert len(tensor_dict1[k]) == len(tensor_dict2[k]), "The list should have the same length."
                res_tensor_dict[k] = [
                    tensor_dict_cat(tensor_dict1[k][_], tensor_dict2[k][_], dim=dim)
                    for _ in range(len(tensor_dict1[k]))
                ]
            else:
                raise ValueError(f"Unsupported type {type(tensor_dict1[k])} in the tensor dict concat.")
        return dict(res_tensor_dict)


def tensor_dict_index_select(tensor_dict, index, dim=0):
    res_tensor_dict = defaultdict()
    for k in tensor_dict.keys():
        if isinstance(tensor_dict[k], torch.Tensor):
            res_tensor_dict[k] = torch.index_select(tensor_dict[k], index=index, dim=dim).contiguous()
        elif isinstance(tensor_dict[k], dict):
            res_tensor_dict[k] = tensor_dict_index_select(tensor_dict[k], index=index, dim=dim)
        elif isinstance(tensor_dict[k], list):
            res_tensor_dict[k] = [
                tensor_dict_index_select(tensor_dict[k][_], index=index, dim=dim)
                for _ in range(len(tensor_dict[k]))
            ]
        else:
            raise ValueError(f"Unsupported type {type(tensor_dict[k])} in the tensor dict index select.")
    return dict(res_tensor_dict)


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # from issue: https://github.com/pytorch/pytorch/issues/11201
    # import torch.multiprocessing
    # torch.multiprocessing.set_sharing_strategy('file_system')

    # Get runtime option:
    opt = runtime_option()
    cfg = yaml_to_dict(opt.config_path)

    # Loading super config:
    if opt.super_config_path is not None:   # the runtime option is priority
        cfg = load_super_config(cfg, opt.super_config_path)
    else:                                   # if not, use the default super config path in the config file
        cfg = load_super_config(cfg, cfg["SUPER_CONFIG_PATH"])

    # Combine the config and runtime into config dict:
    cfg = update_config(config=cfg, option=opt)

    # Call the "train_engine" function:
    train_engine(config=cfg)
