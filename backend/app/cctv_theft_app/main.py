#!/usr/bin/env python3
"""CCTV Theft Detection — application entrypoint.

    python main.py                                        # cameras from configs/app.yaml
    python main.py --source rtsp://user:pass@cam/stream   # single camera override
    python main.py --source video.mp4 --temporal mock --detector mock --vlm mock
                                                          # GPU-less pipeline test

Backends per phase are chosen in configs/*.yaml and can be overridden here.
Secrets (VLM key, Telegram token, ...) load from .env in this directory.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml

APP_ROOT = Path(__file__).resolve().parent
ORIG_CWD = Path.cwd()                       # where the user invoked us from
sys.path.insert(0, str(APP_ROOT))          # allow `python main.py` from anywhere
os.chdir(APP_ROOT)                          # configs use app-relative paths


def load_dotenv(path: Path) -> None:
    """Tiny .env loader (KEY=VALUE lines; no external dependency)."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/app.yaml")
    ap.add_argument("--source", help="override configs with a single camera source "
                                     "(rtsp://... | file path | webcam:0)")
    ap.add_argument("--camera-id", default="cam01")
    ap.add_argument("--loop", action="store_true", help="loop file sources")
    ap.add_argument("--detector", choices=["yolo", "mock"], help="Stage 1 backend")
    ap.add_argument("--temporal", choices=["novelty", "finetuned", "mock"],
                    help="Stage 2 mode (see configs/temporal_model.yaml)")
    ap.add_argument("--vlm", choices=["openai_compatible", "hf_local", "mock"],
                    help="Stage 3 backend")
    ap.add_argument("--device", help="cuda | cpu")
    ap.add_argument("--duration", type=float, help="stop after N seconds")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    load_dotenv(APP_ROOT / ".env")

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.source:
        source = args.source
        if "://" not in source and not source.startswith("webcam:") \
                and not Path(source).is_absolute():
            source = str((ORIG_CWD / source).resolve())   # relative to invoking cwd
        cfg["cameras"] = [{"id": args.camera_id, "source": source,
                           "loop": args.loop}]
    if args.duration:
        cfg.setdefault("runtime", {})["max_runtime_s"] = args.duration

    from pipeline.orchestrator import PipelineOrchestrator
    orchestrator = PipelineOrchestrator(cfg, overrides={
        "detector_backend": args.detector, "temporal_mode": args.temporal,
        "vlm_backend": args.vlm, "device": args.device})
    stats = orchestrator.run()

    print(f"\n── summary ─────────────────────────────────────────")
    print(f" frames processed     : {stats['frames']}")
    print(f" temporal windows     : {stats['windows']}")
    print(f" candidate events     : {stats['candidates']}")
    print(f" alerts dispatched    : {stats['alerts']}")
    print(f" review queue / clips : data/review/review.jsonl , data/clips/")
    print(f" dashboard            : python -m dashboard.server")


if __name__ == "__main__":
    main()
