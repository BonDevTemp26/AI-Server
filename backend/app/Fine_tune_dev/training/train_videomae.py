"""Fine-tune VideoMAEv2 on labeled theft clips.

Usage (from the repository root):

    python -m training.train_videomae --config configs/temporal_model.yaml
    python -m training.train_videomae --config configs/temporal_model.yaml \
        --epochs 20 --freeze-blocks 6 --no-mixup        # small-dataset recipe

Implements the standard VideoMAE fine-tuning recipe: AdamW + layer-wise LR
decay, cosine schedule with warmup, mixup/cutmix with soft-target CE, AMP,
gradient accumulation, balanced sampling, early stopping, and best-checkpoint
selection on a precision-oriented metric (theft PR-AUC by default).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from temporal.videomae_finetune import (
    build_model,
    build_param_groups,
    freeze_blocks,
    save_finetuned_checkpoint,
)
from training.datasets import VideoClipDataset, make_balanced_sampler
from training.eval_metrics import collect_probs, compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("train_videomae")


# ── Small helpers ────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_amp(mode: str, device: str) -> torch.dtype | None:
    if device == "cpu" or mode == "off":
        return None
    if mode == "bf16":
        return torch.bfloat16
    if mode == "fp16":
        return torch.float16
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def cosine_lr(step: int, total: int, warmup: int, base: float, min_lr: float) -> float:
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return min_lr + (base - min_lr) * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def one_hot_smooth(y: torch.Tensor, num_classes: int, smoothing: float) -> torch.Tensor:
    off = smoothing / num_classes
    target = torch.full((y.size(0), num_classes), off, device=y.device)
    target.scatter_(1, y.unsqueeze(1), 1.0 - smoothing + off)
    return target


def apply_mixup(x: torch.Tensor, target: torch.Tensor, mixup_alpha: float,
                cutmix_alpha: float, switch_prob: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Batch-level mixup/cutmix on video clips; targets must already be one-hot."""
    use_cutmix = cutmix_alpha > 0 and (mixup_alpha <= 0 or random.random() < switch_prob)
    alpha = cutmix_alpha if use_cutmix else mixup_alpha
    lam = float(np.random.beta(alpha, alpha))
    if use_cutmix:
        _, _, _, h, w = x.shape
        rh, rw = int(h * math.sqrt(1 - lam)), int(w * math.sqrt(1 - lam))
        top = random.randint(0, h - rh) if rh < h else 0
        left = random.randint(0, w - rw) if rw < w else 0
        x[:, :, :, top:top + rh, left:left + rw] = \
            x.flip(0)[:, :, :, top:top + rh, left:left + rw]
        lam = 1.0 - (rh * rw) / (h * w)  # correct lam for the actual box
    else:
        flipped = x.flip(0).mul_(1.0 - lam)  # flip copies, so scale it first
        x.mul_(lam).add_(flipped)
    target = lam * target + (1.0 - lam) * target.flip(0)
    return x, target


def soft_ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return -(target * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


# ── Training ─────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/temporal_model.yaml")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--base-lr", type=float)
    ap.add_argument("--freeze-blocks", type=int)
    ap.add_argument("--output-dir")
    ap.add_argument("--resume", help="checkpoint to resume optimizer/epoch state from")
    ap.add_argument("--no-mixup", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    mcfg, dcfg, tcfg = cfg["model"], cfg["data"], cfg["train"]
    acfg = cfg.get("augment", {})
    if args.epochs: tcfg["epochs"] = args.epochs
    if args.batch_size: tcfg["batch_size"] = args.batch_size
    if args.base_lr: tcfg["base_lr"] = args.base_lr
    if args.freeze_blocks is not None: tcfg["freeze_blocks"] = args.freeze_blocks
    if args.output_dir: tcfg["output_dir"] = args.output_dir
    if args.no_mixup: tcfg["mixup_alpha"] = tcfg["cutmix_alpha"] = 0.0

    set_seed(int(tcfg.get("seed", 42)))
    torch.backends.cudnn.benchmark = True
    device = args.device
    class_names = list(dcfg["classes"])
    num_classes = len(class_names)
    out_dir = Path(tcfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "label_map.json").write_text(json.dumps(
        {i: c for i, c in enumerate(class_names)}, indent=2))
    (out_dir / "config_snapshot.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    # Data ------------------------------------------------------------------
    common = dict(class_names=class_names, num_frames=mcfg["num_frames"],
                  sampling_rate=dcfg["sampling_rate"], input_size=mcfg["input_size"])
    train_ds = VideoClipDataset(dcfg["train_manifest"], mode="train", augment=acfg, **common)
    val_ds = VideoClipDataset(dcfg["val_manifest"], mode="val", **common)
    sampler = make_balanced_sampler(train_ds) if dcfg.get("balanced_sampling", True) else None
    train_loader = DataLoader(
        train_ds, batch_size=int(tcfg["batch_size"]), sampler=sampler,
        shuffle=sampler is None, num_workers=int(dcfg.get("num_workers", 4)),
        pin_memory=(device == "cuda"), drop_last=True,
        persistent_workers=int(dcfg.get("num_workers", 4)) > 0,
    )
    val_loader = DataLoader(val_ds, batch_size=max(2 * int(tcfg["batch_size"]), 2),
                            shuffle=False, num_workers=int(dcfg.get("num_workers", 4)),
                            pin_memory=(device == "cuda"))
    logger.info("train clips: %d | val clips: %d | classes: %s",
                train_ds.num_clips, val_ds.num_clips, class_names)

    # Model / optimizer -------------------------------------------------------
    model = build_model(mcfg, num_classes=num_classes).to(device)
    freeze_blocks(model, int(tcfg.get("freeze_blocks", 0)))

    accum = max(1, int(tcfg.get("accum_steps", 1)))
    eff_batch = int(tcfg["batch_size"]) * accum
    lr = float(tcfg["base_lr"]) * eff_batch / 256.0
    min_lr = float(tcfg.get("min_lr", 1e-6))
    groups = build_param_groups(model, float(tcfg.get("weight_decay", 0.05)),
                                float(tcfg.get("layer_decay", 0.75)))
    optimizer = torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.999))

    steps_per_epoch = max(1, len(train_loader) // accum)
    total_steps = int(tcfg["epochs"]) * steps_per_epoch
    warmup_steps = int(tcfg.get("warmup_epochs", 5)) * steps_per_epoch
    logger.info("effective batch %d | lr %.2e | %d steps (%d warmup)",
                eff_batch, lr, total_steps, warmup_steps)

    amp_dtype = resolve_amp(str(tcfg.get("amp", "auto")), device)
    scaler = torch.amp.GradScaler(device, enabled=amp_dtype == torch.float16)

    mixup_on = float(tcfg.get("mixup_alpha", 0)) > 0 or float(tcfg.get("cutmix_alpha", 0)) > 0
    smoothing = float(tcfg.get("label_smoothing", 0.1))
    theft_weights = dcfg.get("theft_class_weights", {})
    monitor = str(tcfg.get("monitor", "theft_pr_auc"))
    patience = int(tcfg.get("early_stop_patience", 0))

    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(out_dir / "tb"))
    except Exception:
        writer = None

    start_epoch, best_metric, best_epoch, global_step = 0, -1.0, -1, 0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = state["epoch"] + 1
        best_metric = state.get("best_metric", -1.0)
        global_step = start_epoch * steps_per_epoch
        logger.info("Resumed from %s at epoch %d", args.resume, start_epoch)

    # Loop --------------------------------------------------------------------
    for epoch in range(start_epoch, int(tcfg["epochs"])):
        model.train()
        t0, running, seen = time.time(), 0.0, 0
        optimizer.zero_grad(set_to_none=True)

        for it, (clips, labels) in enumerate(train_loader):
            clips = clips.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            target = one_hot_smooth(labels, num_classes, smoothing)
            if mixup_on and random.random() < float(tcfg.get("mixup_prob", 1.0)):
                clips, target = apply_mixup(
                    clips, target, float(tcfg.get("mixup_alpha", 0)),
                    float(tcfg.get("cutmix_alpha", 0)),
                    float(tcfg.get("mixup_switch_prob", 0.5)))

            with torch.autocast(device, dtype=amp_dtype, enabled=amp_dtype is not None):
                loss = soft_ce(model(clips), target) / accum
            scaler.scale(loss).backward()
            running += loss.item() * accum
            seen += 1

            if (it + 1) % accum == 0:
                step_lr = cosine_lr(global_step, total_steps, warmup_steps, lr, min_lr)
                for g in optimizer.param_groups:
                    g["lr"] = step_lr * g.get("lr_scale", 1.0)
                scaler.unscale_(optimizer)
                clip_grad = float(tcfg.get("clip_grad", 0) or 0)
                if clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % 20 == 0:
                    logger.info("epoch %d step %d/%d | loss %.4f | lr %.2e",
                                epoch, global_step, total_steps, running / seen, step_lr)
                    if writer:
                        writer.add_scalar("train/loss", running / seen, global_step)
                        writer.add_scalar("train/lr", step_lr, global_step)
                    running, seen = 0.0, 0

        # Validation ----------------------------------------------------------
        y_true, probs, _ = collect_probs(model, val_loader, device, amp_dtype=amp_dtype)
        metrics = compute_metrics(y_true, probs, class_names, theft_weights)
        logger.info(
            "epoch %d done in %.0fs | val top1 %.3f | macro_f1 %.3f | "
            "theft_pr_auc %.3f | theft_roc_auc %.3f",
            epoch, time.time() - t0, metrics["top1"], metrics["macro_f1"],
            metrics["theft_pr_auc"], metrics["theft_roc_auc"])
        if writer:
            for k in ("top1", "macro_f1", "theft_pr_auc", "theft_roc_auc"):
                writer.add_scalar(f"val/{k}", metrics[k], epoch)

        # Checkpoints ----------------------------------------------------------
        trainer_state = {"optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                         "epoch": epoch, "best_metric": best_metric,
                         "val_metrics": {k: metrics[k] for k in
                                         ("top1", "macro_f1", "theft_pr_auc", "theft_roc_auc")}}
        save_finetuned_checkpoint(out_dir / "last.pth", model, class_names, mcfg,
                                  extra=trainer_state)
        if metrics[monitor] > best_metric:
            best_metric, best_epoch = metrics[monitor], epoch
            save_finetuned_checkpoint(out_dir / "best.pth", model, class_names, mcfg,
                                      extra={"epoch": epoch, "val_metrics": trainer_state["val_metrics"]})
            logger.info("new best %s=%.4f -> %s", monitor, best_metric, out_dir / "best.pth")
        elif patience > 0 and epoch - best_epoch >= patience:
            logger.info("early stop: no %s improvement for %d epochs", monitor, patience)
            break

    logger.info("training finished | best %s=%.4f (epoch %d) | checkpoints in %s",
                monitor, best_metric, best_epoch, out_dir)
    if writer:
        writer.close()


if __name__ == "__main__":
    main()
