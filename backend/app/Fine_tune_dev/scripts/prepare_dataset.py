#!/usr/bin/env python3
"""Cut annotated CCTV footage into training clips and build leakage-safe splits.

Input: an annotation CSV (see data/annotations/example_annotations.csv):

    video,start_s,end_s,label,camera_id
    cam03/2026-07-01_14.mp4,732.5,738.0,concealment,cam03
    ...

For every row this script cuts ``[start_s - context, end_s + context]`` from
the source video, re-encodes it to a normalized fps / short-side resolution
(fast to decode during training), and files it under
``data/labeled_clips/<label>/``. It can also mine "normal" clips from the
unlabeled gaps of the same videos, and writes train/val/test manifests split
**by source video** (or camera) so near-duplicate frames never leak across
splits.

    python scripts/prepare_dataset.py \
        --annotations data/annotations/my_annotations.csv \
        --videos-root data/raw_clips \
        --negatives-per-video 4
"""

from __future__ import annotations

import argparse
import random
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent


def ffprobe_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def cut_clip(src: Path, dst: Path, start: float, duration: float,
             fps: int, short_side: int) -> None:
    """Frame-accurate cut + normalization (re-encode makes -ss accurate)."""
    scale = (f"scale='if(gt(iw,ih),-2,{short_side})'"
             f":'if(gt(iw,ih),{short_side},-2)'")
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-ss", f"{max(0.0, start):.3f}", "-i", str(src), "-t", f"{duration:.3f}",
         "-vf", scale, "-r", str(fps),
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
         "-an", "-movflags", "+faststart", str(dst)],
        check=True)


def mine_negative_windows(video_duration: float, busy: list[tuple[float, float]],
                          count: int, length: float, margin: float,
                          rng: random.Random) -> list[float]:
    """Random window starts at least ``margin`` away from any labeled interval."""
    starts, attempts = [], 0
    while len(starts) < count and attempts < count * 30:
        attempts += 1
        if video_duration <= length:
            break
        s = rng.uniform(0.0, video_duration - length)
        window = (s - margin, s + length + margin)
        if any(not (window[1] <= b0 or window[0] >= b1) for b0, b1 in busy):
            continue
        if any(abs(s - prev) < length for prev in starts):  # avoid near-duplicates
            continue
        starts.append(s)
    return starts


def grouped_split(df: pd.DataFrame, group_col: str, val_frac: float,
                  test_frac: float, seed: int) -> pd.DataFrame:
    """Assign whole groups (videos/cameras) to splits — no cross-split leakage."""
    rng = random.Random(seed)
    groups = sorted(df[group_col].unique())
    rng.shuffle(groups)
    n = len(df)
    split_of: dict[str, str] = {}
    counts = {"test": 0, "val": 0}
    targets = {"test": test_frac * n, "val": val_frac * n}
    for g in groups:
        size = int((df[group_col] == g).sum())
        for name in ("test", "val"):
            if counts[name] < targets[name]:
                split_of[g] = name
                counts[name] += size
                break
        else:
            split_of[g] = "train"
    df = df.copy()
    df["split"] = df[group_col].map(split_of)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--videos-root", default="data/raw_clips")
    ap.add_argument("--out-root", default="data/labeled_clips")
    ap.add_argument("--manifest-dir", default="data/annotations")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--short-side", type=int, default=320)
    ap.add_argument("--context-s", type=float, default=1.0,
                    help="padding added before/after each annotated interval")
    ap.add_argument("--min-clip-s", type=float, default=1.5)
    ap.add_argument("--max-clip-s", type=float, default=10.0)
    ap.add_argument("--negatives-per-video", type=int, default=0,
                    help="mine N 'normal' clips from unlabeled gaps per video")
    ap.add_argument("--negative-length-s", type=float, default=4.0)
    ap.add_argument("--negative-label", default="normal")
    ap.add_argument("--negative-margin-s", type=float, default=5.0)
    ap.add_argument("--group-by", choices=["video", "camera"], default="video")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ann = pd.read_csv(args.annotations)
    required = {"video", "start_s", "end_s", "label"}
    if missing := required - set(ann.columns):
        sys.exit(f"{args.annotations} is missing columns: {missing}")
    if "camera_id" not in ann.columns:
        ann["camera_id"] = ann["video"].map(lambda v: Path(v).stem.split("_")[0])

    videos_root = Path(args.videos_root)
    out_root = Path(args.out_root)
    rng = random.Random(args.seed)
    rows, failures = [], 0

    def emit(src: Path, video_rel: str, camera: str, label: str,
             start: float, end: float, idx: int) -> None:
        nonlocal failures
        duration = min(max(end - start, args.min_clip_s), args.max_clip_s)
        stem = Path(video_rel).stem
        dst = out_root / label / f"{camera}_{stem}_{idx:04d}_{int(start * 1000)}.mp4"
        try:
            cut_clip(src, dst, start, duration, args.fps, args.short_side)
        except subprocess.CalledProcessError:
            failures += 1
            print(f"  ! ffmpeg failed on {video_rel} @ {start:.1f}s — skipped")
            return
        rows.append({
            "path": dst.relative_to(REPO_ROOT).as_posix() if dst.is_relative_to(REPO_ROOT)
                    else dst.as_posix(),
            "label": label, "video_id": video_rel, "camera_id": camera,
            "start_s": round(start, 3), "end_s": round(start + duration, 3),
        })

    for video_rel, group in ann.groupby("video"):
        src = videos_root / video_rel
        if not src.is_file():
            print(f"  ! missing source video: {src} — {len(group)} annotations skipped")
            continue
        print(f"processing {video_rel} ({len(group)} annotations)")
        camera = str(group["camera_id"].iloc[0])
        intervals = []
        for i, r in enumerate(group.itertuples(index=False)):
            start = max(0.0, float(r.start_s) - args.context_s)
            end = float(r.end_s) + args.context_s
            intervals.append((start, end))
            emit(src, str(video_rel), camera, str(r.label), start, end, i)

        if args.negatives_per_video > 0:
            duration = ffprobe_duration(src)
            for j, s in enumerate(mine_negative_windows(
                    duration, intervals, args.negatives_per_video,
                    args.negative_length_s, args.negative_margin_s, rng)):
                emit(src, str(video_rel), camera, args.negative_label,
                     s, s + args.negative_length_s, 9000 + j)

    if not rows:
        sys.exit("No clips produced — check --annotations and --videos-root.")

    df = pd.DataFrame(rows)
    group_col = "video_id" if args.group_by == "video" else "camera_id"
    df = grouped_split(df, group_col, args.val_frac, args.test_frac, args.seed)

    manifest_dir = Path(args.manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        part = df[df["split"] == split].drop(columns=["split"])
        part.to_csv(manifest_dir / f"{split}.csv", index=False)

    print(f"\n{len(df)} clips written ({failures} failures) → {out_root}")
    print(df.groupby(["split", "label"]).size().unstack(fill_value=0).to_string())
    for split in ("train", "val", "test"):
        missing_cls = set(df["label"].unique()) - set(df[df.split == split]["label"])
        if missing_cls:
            print(f"  ⚠ split '{split}' has no clips for: {sorted(missing_cls)} "
                  f"(add data or adjust fractions)")
    print(f"manifests → {manifest_dir}/train.csv, val.csv, test.csv")


if __name__ == "__main__":
    main()
