"""Load the VideoMAEv2 architecture from the training stack (single source of truth).

The fine-tuning stack in ``<repo>/Fine_tune_dev/`` (``temporal/videomae_model.py``,
built alongside this app) contains the checkpoint-compatible VideoMAEv2
implementation. Instead of duplicating ~350 lines of model code here, we load
that module by file path — both packages are named ``temporal``, so a normal
import cannot reach it.

Configure the location with ``backbone.training_stack_path`` in
``configs/temporal_model.yaml`` (default ``../Fine_tune_dev``).
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_MODULE_CACHE: dict[str, object] = {}


def load_videomae_module(training_stack_path: str | Path):
    """Return the training stack's ``videomae_model`` module (cached)."""
    root = Path(training_stack_path).resolve()
    model_file = root / "temporal" / "videomae_model.py"
    key = str(model_file)
    if key in _MODULE_CACHE:
        return _MODULE_CACHE[key]
    if not model_file.is_file():
        raise FileNotFoundError(
            f"VideoMAEv2 architecture not found at {model_file}.\n"
            f"Set backbone.training_stack_path in configs/temporal_model.yaml to "
            f"the repository root that contains the training stack."
        )
    spec = importlib.util.spec_from_file_location("videomae_model_stack", model_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules["videomae_model_stack"] = module
    spec.loader.exec_module(module)
    _MODULE_CACHE[key] = module
    logger.info("Loaded VideoMAEv2 architecture from %s", model_file)
    return module


def build_backbone(training_stack_path: str | Path, arch: str, num_classes: int,
                   num_frames: int, input_size: int, checkpoint: str | None,
                   device: str = "cuda"):
    """Build a VisionTransformer and optionally load a checkpoint into it."""
    import torch

    vm = load_videomae_module(training_stack_path)
    model = vm.MODEL_REGISTRY[arch](num_classes=num_classes, all_frames=num_frames,
                                    img_size=input_size, drop_path_rate=0.0)
    if checkpoint:
        ckpt_path = Path(checkpoint)
        if not ckpt_path.is_file():
            raise FileNotFoundError(
                f"Backbone checkpoint not found: {ckpt_path}\n"
                f"Download it:  cd ../Fine_tune_dev && python scripts/download_pretrained.py --model base"
            )
        vm.load_videomae_checkpoint(model, str(ckpt_path))
    model.eval().to(device)
    return model, vm
