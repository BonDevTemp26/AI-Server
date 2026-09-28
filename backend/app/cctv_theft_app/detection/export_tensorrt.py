"""Export the Stage-1 detector to TensorRT (run on the deployment device).

    python -m detection.export_tensorrt --config configs/detector.yaml --half

The resulting ``.engine`` file is device-specific; point ``detector.yaml``'s
``model:`` at it afterwards (Ultralytics loads .engine files transparently,
including inside ``model.track``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/detector.yaml")
    ap.add_argument("--model", help="override detector.yaml model")
    ap.add_argument("--half", action="store_true", help="fp16 engine (recommended)")
    ap.add_argument("--int8", action="store_true", help="int8 engine (needs calibration data)")
    ap.add_argument("--device", default=0, type=int)
    args = ap.parse_args()

    from ultralytics import YOLO

    cfg = yaml.safe_load(Path(args.config).read_text())
    model_name = args.model or cfg["model"]
    model = YOLO(model_name)
    engine_path = model.export(
        format="engine",
        imgsz=int(cfg.get("imgsz", 640)),
        half=args.half and not args.int8,
        int8=args.int8,
        device=args.device,
    )
    print(f"\nTensorRT engine: {engine_path}")
    print(f"Update configs/detector.yaml →  model: {engine_path}")


if __name__ == "__main__":
    main()
