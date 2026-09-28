"""FINAL Stage-2 model — adapter for your fine-tuned VideoMAEv2 classifier.

This is the placeholder the application is built around. When the fine-tuned
checkpoint exists (produced by the fine-tuning stack in Fine_tune_dev/:
``python -m training.train_videomae`` → ``best.pth``), switch it on with:

    # configs/temporal_model.yaml
    mode: finetuned
    finetuned:
      checkpoint: ../Fine_tune_dev/models/checkpoints/videomaev2_theft_ft/best.pth

No application code changes are needed: checkpoints are self-describing
(weights + class names + architecture), and this adapter emits the same
``TemporalResult`` contract as the temporary model — with real per-class
action probabilities (`concealment`, `grab_and_run`, ...) instead of a
novelty heuristic.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from pipeline.events import TemporalResult
from temporal.backbone_loader import load_videomae_module
from temporal.base import TemporalModel
from temporal.rtfm_infer import RTFMHead

logger = logging.getLogger(__name__)


class FinetunedVideoMAEv2Model(TemporalModel):
    name = "videomaev2_finetuned"

    def __init__(self, cfg: dict, device: str = "cuda"):
        import torch
        self._torch = torch
        self.device = device
        fcfg = cfg["finetuned"]
        ckpt_path = Path(fcfg["checkpoint"])
        if not ckpt_path.is_file():
            raise FileNotFoundError(
                f"Fine-tuned checkpoint not found: {ckpt_path}\n"
                f"Train it first (from Fine_tune_dev/):  python -m training.train_videomae\n"
                f"Until then run the app with  mode: novelty  (temporary model)."
            )

        vm = load_videomae_module(cfg["backbone"].get("training_stack_path", "../Fine_tune_dev"))
        try:
            ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        except Exception:
            ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        if "class_names" not in ckpt or "model_cfg" not in ckpt:
            raise ValueError(f"{ckpt_path} is not a self-describing fine-tuned "
                             f"checkpoint from training/train_videomae.py")
        self.class_names: list[str] = list(ckpt["class_names"])
        mc = ckpt["model_cfg"]
        self.model = vm.MODEL_REGISTRY[mc["arch"]](
            num_classes=len(self.class_names), all_frames=mc["num_frames"],
            tubelet_size=mc["tubelet_size"], img_size=mc["input_size"],
            drop_path_rate=0.0)
        self.model.load_state_dict(ckpt["model"], strict=True)
        self.model.eval().to(device)
        self._amp = (torch.bfloat16 if device == "cuda"
                     and torch.cuda.is_bf16_supported() else None)

        weights = fcfg.get("theft_class_weights", {})
        unknown = set(weights) - set(self.class_names)
        if unknown:
            raise ValueError(f"theft_class_weights refer to classes not in the "
                             f"checkpoint: {unknown} (has {self.class_names})")
        self.theft_class_weights = {c: float(w) for c, w in weights.items()}
        self.rtfm = RTFMHead.from_config(cfg.get("rtfm", {}), self.model.embed_dim, device)
        logger.info("Temporal model ready: %s (classes: %s%s)", self.name,
                    self.class_names, ", +RTFM" if self.rtfm else "")

        wcfg = cfg["window"]
        if int(wcfg["num_frames"]) != mc["num_frames"] or int(wcfg["input_size"]) != mc["input_size"]:
            raise ValueError(
                f"window config ({wcfg['num_frames']}f @ {wcfg['input_size']}px) does not "
                f"match the checkpoint ({mc['num_frames']}f @ {mc['input_size']}px) — "
                f"align configs/temporal_model.yaml window: with the training settings.")

    def process_window(self, camera_id: str, ts: float, clip: np.ndarray) -> TemporalResult:
        torch = self._torch
        with torch.inference_mode():
            x = torch.from_numpy(np.ascontiguousarray(clip[None])).to(self.device)
            with torch.autocast(self.device, dtype=self._amp, enabled=self._amp is not None):
                feat = self.model.forward_features(x)          # (1, C)
                logits = self.model.head(feat)
            probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()[0]
            emb = feat.float().cpu().numpy()[0]

        action_probs = {c: float(p) for c, p in zip(self.class_names, probs)}
        theft = sum(w * action_probs.get(c, 0.0)
                    for c, w in self.theft_class_weights.items())
        anomaly = self.rtfm.score(emb) if self.rtfm else None
        return TemporalResult(camera_id=camera_id, ts=ts,
                              theft_score=float(min(max(theft, 0.0), 1.0)),
                              action_probs=action_probs, anomaly_score=anomaly,
                              model_name=self.name)
