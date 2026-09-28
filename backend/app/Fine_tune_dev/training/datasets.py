"""Datasets and clip transforms for VideoMAEv2 fine-tuning.

Manifest CSV format (written by ``scripts/prepare_dataset.py``):

    path,label,video_id,camera_id
    data/labeled_clips/concealment/cam03_0012.mp4,concealment,cam03_day2,cam03
    ...

Paths are resolved relative to the repository root (the training scripts are
run from there). All augmentations are *clip-consistent*: one set of random
parameters is drawn per clip and applied to every frame, which is essential
for temporal models.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from temporal.video_io import probe, read_frames
from temporal.videomae_model import IMAGENET_MEAN, IMAGENET_STD

_MEAN = np.array(IMAGENET_MEAN, dtype=np.float32) * 255.0
_STD = np.array(IMAGENET_STD, dtype=np.float32) * 255.0

import cv2  # noqa: E402  (after numpy for OpenCV's sake on some builds)


# ── Frame/clip transform helpers (numpy in, numpy out) ──────────────────────

def _resize_clip(frames: np.ndarray, short_side: int) -> np.ndarray:
    t, h, w, _ = frames.shape
    if min(h, w) == short_side:
        return frames
    scale = short_side / min(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    return np.stack([cv2.resize(f, (nw, nh), interpolation=cv2.INTER_LINEAR) for f in frames])


def _crop_clip(frames: np.ndarray, top: int, left: int, size: int) -> np.ndarray:
    return frames[:, top:top + size, left:left + size]


def _random_resized_crop(frames: np.ndarray, size: int,
                         scale: tuple[float, float]) -> np.ndarray:
    t, h, w, _ = frames.shape
    area = h * w
    for _ in range(10):
        target_area = area * random.uniform(*scale)
        aspect = math.exp(random.uniform(math.log(3 / 4), math.log(4 / 3)))
        cw = int(round(math.sqrt(target_area * aspect)))
        ch = int(round(math.sqrt(target_area / aspect)))
        if cw <= w and ch <= h:
            top = random.randint(0, h - ch)
            left = random.randint(0, w - cw)
            cropped = frames[:, top:top + ch, left:left + cw]
            return np.stack([cv2.resize(f, (size, size), interpolation=cv2.INTER_LINEAR)
                             for f in cropped])
    # Fallback: center crop of the short side
    frames = _resize_clip(frames, size)
    t, h, w, _ = frames.shape
    top, left = (h - size) // 2, (w - size) // 2
    return _crop_clip(frames, top, left, size)


def _color_jitter(frames: np.ndarray, strength: float) -> np.ndarray:
    """Brightness / contrast / saturation jitter with one factor set per clip."""
    x = frames.astype(np.float32)
    b = random.uniform(1 - strength, 1 + strength)
    c = random.uniform(1 - strength, 1 + strength)
    s = random.uniform(1 - strength, 1 + strength)
    x = x * b
    gray = (0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2])[..., None]
    x = (x - gray) * s + gray                       # saturation
    x = (x - x.mean()) * c + x.mean()               # contrast (clip-level mean)
    return np.clip(x, 0, 255)


def _normalize_to_tensor(frames: np.ndarray) -> torch.Tensor:
    """(T, H, W, 3) uint8/float RGB → (3, T, H, W) float32, ImageNet-normalized."""
    x = (frames.astype(np.float32) - _MEAN) / _STD
    return torch.from_numpy(np.ascontiguousarray(x.transpose(3, 0, 1, 2)))


def _random_erase(clip: torch.Tensor, prob: float) -> torch.Tensor:
    """Erase one static box (same location on all frames) with probability ``prob``."""
    if prob <= 0 or random.random() > prob:
        return clip
    _, _, h, w = clip.shape
    area = random.uniform(0.02, 0.2) * h * w
    aspect = math.exp(random.uniform(math.log(0.3), math.log(3.3)))
    eh = min(h - 1, int(round(math.sqrt(area / aspect))))
    ew = min(w - 1, int(round(math.sqrt(area * aspect))))
    if eh < 1 or ew < 1:
        return clip
    top = random.randint(0, h - eh)
    left = random.randint(0, w - ew)
    clip[:, :, top:top + eh, left:left + ew] = 0.0  # 0 == channel mean post-normalization
    return clip


# ── Dataset ──────────────────────────────────────────────────────────────────

class VideoClipDataset(Dataset):
    """Labeled video clips for train / val / multi-view test evaluation.

    * ``mode='train'`` → random temporal window + augmentation, returns
      ``(clip, label)``.
    * ``mode='val'``   → centered window + center crop, returns ``(clip, label)``.
    * ``mode='test'``  → expands each clip into ``temporal_views × spatial_views``
      deterministic views, returns ``(clip, label, sample_idx)`` so logits can be
      averaged per source clip.
    """

    def __init__(self, manifest: str, class_names: list[str], num_frames: int = 16,
                 sampling_rate: int = 4, input_size: int = 224, mode: str = "train",
                 augment: dict | None = None, temporal_views: int = 2,
                 spatial_views: int = 3, root: str | Path = "."):
        if mode not in ("train", "val", "test"):
            raise ValueError(f"mode must be train|val|test, got {mode}")
        self.df = pd.read_csv(manifest)
        for col in ("path", "label"):
            if col not in self.df.columns:
                raise ValueError(f"{manifest} must have a '{col}' column")
        unknown = set(self.df["label"]) - set(class_names)
        if unknown:
            raise ValueError(f"{manifest} contains labels not in config classes: {unknown}")

        self.class_to_idx = {c: i for i, c in enumerate(class_names)}
        self.labels = [self.class_to_idx[l] for l in self.df["label"]]
        self.root = Path(root)
        self.num_frames = num_frames
        self.sampling_rate = sampling_rate
        self.input_size = input_size
        self.mode = mode
        self.augment = augment or {}
        self.temporal_views = max(1, temporal_views) if mode == "test" else 1
        self.spatial_views = max(1, spatial_views) if mode == "test" else 1

    def __len__(self) -> int:
        return len(self.df) * self.temporal_views * self.spatial_views

    @property
    def num_clips(self) -> int:
        return len(self.df)

    def _path(self, row) -> str:
        p = Path(row["path"])
        return str(p if p.is_absolute() else self.root / p)

    def _temporal_indices(self, total: int, view: int, views: int) -> np.ndarray:
        span = self.num_frames * self.sampling_rate
        if total >= span:
            slack = total - span
            if self.mode == "train":
                start = random.randint(0, slack)
            elif views == 1:
                start = slack // 2
            else:
                start = int(round(view * slack / (views - 1)))
        else:
            start = 0
        idx = start + np.arange(self.num_frames) * self.sampling_rate
        return np.clip(idx, 0, total - 1)

    def _spatial_view(self, frames: np.ndarray, view: int) -> np.ndarray:
        size = self.input_size
        frames = _resize_clip(frames, size)
        _, h, w, _ = frames.shape
        if self.spatial_views == 1 or (h == size and w == size):
            return _crop_clip(frames, (h - size) // 2, (w - size) // 2, size)
        if h > w:   # tall: top / center / bottom
            tops = [0, (h - size) // 2, h - size]
            return _crop_clip(frames, tops[view], 0, size)
        lefts = [0, (w - size) // 2, w - size]  # wide: left / center / right
        return _crop_clip(frames, 0, lefts[view], size)

    def __getitem__(self, index: int):
        views_per_clip = self.temporal_views * self.spatial_views
        clip_idx = index // views_per_clip
        view_idx = index % views_per_clip
        t_view, s_view = divmod(view_idx, self.spatial_views)

        row = self.df.iloc[clip_idx]
        path = self._path(row)
        total, _ = probe(path)
        indices = self._temporal_indices(max(total, 1), t_view, self.temporal_views)
        frames = read_frames(path, indices)  # (T, H, W, 3) RGB uint8

        if self.mode == "train":
            frames = _random_resized_crop(frames, self.input_size,
                                          tuple(self.augment.get("crop_scale", (0.5, 1.0))))
            if random.random() < float(self.augment.get("hflip", 0.5)):
                frames = frames[:, :, ::-1]
            cj = float(self.augment.get("color_jitter", 0.0))
            if cj > 0:
                frames = _color_jitter(frames, cj)
            clip = _normalize_to_tensor(frames)
            clip = _random_erase(clip, float(self.augment.get("random_erase", 0.0)))
            return clip, self.labels[clip_idx]

        clip = _normalize_to_tensor(self._spatial_view(frames, s_view))
        if self.mode == "val":
            return clip, self.labels[clip_idx]
        return clip, self.labels[clip_idx], clip_idx


def make_balanced_sampler(dataset: VideoClipDataset) -> WeightedRandomSampler:
    """Oversample minority classes so each epoch sees a roughly uniform class mix."""
    counts = np.bincount(dataset.labels, minlength=len(dataset.class_to_idx))
    weights = np.array([1.0 / max(counts[l], 1) for l in dataset.labels], dtype=np.float64)
    return WeightedRandomSampler(torch.from_numpy(weights), num_samples=len(dataset.labels),
                                 replacement=True)
