"""Model construction and fine-tuning utilities for VideoMAEv2.

Bridges the raw architecture (``temporal/videomae_model.py``) and the training /
inference entry points: builds a model from the YAML config, loads pretrained
weights, assembles layer-wise LR-decay parameter groups, and restores
fine-tuned checkpoints for inference.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from temporal.videomae_model import (
    MODEL_REGISTRY,
    VisionTransformer,
    load_videomae_checkpoint,
)

logger = logging.getLogger(__name__)


def build_model(model_cfg: dict, num_classes: int, load_pretrained: bool = True) -> VisionTransformer:
    """Build a VideoMAEv2 classifier from the ``model:`` section of the config."""
    arch = model_cfg["arch"]
    if arch not in MODEL_REGISTRY:
        raise KeyError(f"Unknown arch '{arch}'. Available: {sorted(MODEL_REGISTRY)}")

    model = MODEL_REGISTRY[arch](
        num_classes=num_classes,
        all_frames=int(model_cfg.get("num_frames", 16)),
        tubelet_size=int(model_cfg.get("tubelet_size", 2)),
        img_size=int(model_cfg.get("input_size", 224)),
        drop_path_rate=float(model_cfg.get("drop_path_rate", 0.1)),
        grad_checkpointing=bool(model_cfg.get("grad_checkpointing", False)),
    )

    pretrained = model_cfg.get("pretrained_checkpoint")
    if load_pretrained and pretrained:
        path = Path(pretrained)
        if not path.is_file():
            raise FileNotFoundError(
                f"Pretrained checkpoint not found: {path}\n"
                f"Download it first:  python scripts/download_pretrained.py --model base"
            )
        load_videomae_checkpoint(model, str(path))
    return model


def freeze_blocks(model: VisionTransformer, num_blocks: int) -> None:
    """Freeze the patch embedding and the first ``num_blocks`` transformer blocks.

    Useful on very small datasets (< ~300 clips) to limit overfitting.
    """
    if num_blocks <= 0:
        return
    for p in model.patch_embed.parameters():
        p.requires_grad = False
    for blk in model.blocks[:num_blocks]:
        for p in blk.parameters():
            p.requires_grad = False
    frozen = sum(1 for p in model.parameters() if not p.requires_grad)
    logger.info("Froze patch_embed + first %d blocks (%d tensors)", num_blocks, frozen)


def _layer_id(name: str, depth: int) -> int:
    """BEiT/VideoMAE layer-id assignment for layer-wise LR decay."""
    if name.startswith(("patch_embed", "pos_embed", "cls_token", "mask_token")):
        return 0
    if name.startswith("blocks."):
        return int(name.split(".")[1]) + 1
    return depth + 1  # fc_norm / norm / head


def build_param_groups(model: VisionTransformer, weight_decay: float,
                       layer_decay: float) -> list[dict]:
    """Parameter groups with per-layer LR scales and no-decay handling.

    Each group carries ``lr_scale``; the training loop multiplies the scheduled
    LR by it every step. Biases, LayerNorm weights, and the q/v bias and gamma
    parameters are excluded from weight decay (standard ViT recipe).
    """
    depth = model.depth
    num_layers = depth + 2  # embed(0) .. blocks(1..depth) .. head(depth+1)
    scales = [layer_decay ** (num_layers - 1 - i) for i in range(num_layers)]

    groups: dict[str, dict] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        no_decay = param.ndim <= 1 or name.endswith((".q_bias", ".v_bias", ".gamma_1", ".gamma_2"))
        lid = _layer_id(name, depth)
        key = f"layer{lid}_{'no_decay' if no_decay else 'decay'}"
        if key not in groups:
            groups[key] = {
                "params": [],
                "weight_decay": 0.0 if no_decay else weight_decay,
                "lr_scale": scales[lid],
                "name": key,
            }
        groups[key]["params"].append(param)
    return list(groups.values())


def save_finetuned_checkpoint(path: str | Path, model: VisionTransformer,
                              class_names: list[str], model_cfg: dict,
                              extra: dict | None = None) -> None:
    """Save a self-describing checkpoint (weights + label map + architecture)."""
    payload = {
        "model": model.state_dict(),
        "class_names": list(class_names),
        "model_cfg": {
            "arch": model_cfg["arch"],
            "num_frames": int(model_cfg.get("num_frames", 16)),
            "tubelet_size": int(model_cfg.get("tubelet_size", 2)),
            "input_size": int(model_cfg.get("input_size", 224)),
        },
    }
    if extra:
        payload.update(extra)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(path))


def load_for_inference(ckpt_path: str | Path, device: str = "cuda"
                       ) -> tuple[VisionTransformer, list[str], dict]:
    """Restore a fine-tuned checkpoint saved by ``training/train_videomae.py``.

    Returns ``(model.eval() on device, class_names, model_cfg)``. The checkpoint
    is self-describing, so no YAML is needed at inference time.
    """
    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    except Exception:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    if "class_names" not in ckpt or "model_cfg" not in ckpt:
        raise ValueError(
            f"{ckpt_path} is not a fine-tuned pipeline checkpoint (missing metadata). "
            "Pretrained backbones must go through training/train_videomae.py first."
        )
    class_names, model_cfg = list(ckpt["class_names"]), dict(ckpt["model_cfg"])

    model = MODEL_REGISTRY[model_cfg["arch"]](
        num_classes=len(class_names),
        all_frames=model_cfg["num_frames"],
        tubelet_size=model_cfg["tubelet_size"],
        img_size=model_cfg["input_size"],
        drop_path_rate=0.0,
    )
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval().to(device)
    logger.info("Loaded fine-tuned model %s (classes: %s)", ckpt_path, class_names)
    return model, class_names, model_cfg
