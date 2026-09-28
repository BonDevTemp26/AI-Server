#!/usr/bin/env python3
"""Download official VideoMAEv2 pretrained weights for fine-tuning.

Defaults to the K710-distilled checkpoints from the official release
(HuggingFace repo ``OpenGVLab/VideoMAE2``) — these distill the ViT-giant model
into ViT-S/ViT-B and are the recommended initialization for fine-tuning
(stronger and cheaper than the raw MAE-pretrained weights).

    python scripts/download_pretrained.py --model base     # ViT-B  (default)
    python scripts/download_pretrained.py --model small    # ViT-S  (Jetson/edge)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ID = "OpenGVLab/VideoMAE2"
FILES = {
    "small": "distill/vit_s_k710_dl_from_giant.pth",   # pairs with vit_small_patch16_224
    "base": "distill/vit_b_k710_dl_from_giant.pth",    # pairs with vit_base_patch16_224
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=sorted(FILES), default="base")
    ap.add_argument("--repo-id", default=REPO_ID)
    ap.add_argument("--filename", help="override the file path inside the repo")
    ap.add_argument("--out-dir", default="models/checkpoints")
    args = ap.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit("huggingface_hub is required:  pip install huggingface_hub")

    filename = args.filename or FILES[args.model]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.repo_id}/{filename} → {out_dir}/ ...")
    try:
        path = hf_hub_download(repo_id=args.repo_id, filename=filename,
                               local_dir=str(out_dir))
    except Exception as exc:
        sys.exit(
            f"Download failed: {exc}\n\n"
            f"Check the official model zoo for current locations:\n"
            f"  https://github.com/OpenGVLab/VideoMAEv2/blob/master/docs/MODEL_ZOO.md\n"
            f"then retry with  --repo-id/--filename  overrides."
        )

    # Flatten distill/ subfolder so configs can point at models/checkpoints/<name>.pth
    src = Path(path)
    dst = out_dir / src.name
    if src.resolve() != dst.resolve():
        dst.write_bytes(src.read_bytes())
    print(f"Done: {dst}")
    print(f"Set  model.pretrained_checkpoint: {dst.as_posix()}  in configs/temporal_model.yaml")


if __name__ == "__main__":
    main()
