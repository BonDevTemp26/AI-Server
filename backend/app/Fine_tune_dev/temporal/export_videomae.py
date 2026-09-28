"""Export the fine-tuned VideoMAEv2 classifier for deployment.

Produces:
  * an ONNX graph with dynamic batch (``models/exported/*.onnx``),
  * a ``*.labels.json`` sidecar (class names + preprocessing contract) so the
    ONNX backend of :class:`temporal.videomae_infer.VideoMAEv2Classifier` is
    self-describing,
  * optionally a TensorRT engine (via ``trtexec``) — build this **on the
    deployment device** (Jetson engines are not portable across GPUs).

Usage:

    python -m temporal.export_videomae --config configs/temporal_model.yaml
    python -m temporal.export_videomae --config configs/temporal_model.yaml --build-engine
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch
import yaml

from temporal.videomae_finetune import load_for_inference

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("export_videomae")


def export_onnx(checkpoint: str, onnx_path: str, opset: int = 17) -> dict:
    model, class_names, model_cfg = load_for_inference(checkpoint, device="cpu")
    t, s = model_cfg["num_frames"], model_cfg["input_size"]
    dummy = torch.randn(1, 3, t, s, s)

    out = Path(onnx_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, dummy, str(out),
        input_names=["clip"], output_names=["logits"],
        dynamic_axes={"clip": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
    )
    sidecar = out.with_suffix(".labels.json")
    sidecar.write_text(json.dumps({
        "class_names": class_names,
        "input": {"layout": "NCTHW", "num_frames": t, "input_size": s,
                  "normalize": "imagenet"},
        "source_checkpoint": str(checkpoint),
    }, indent=2))
    logger.info("ONNX graph -> %s (+ %s)", out, sidecar.name)
    return {"model": model, "dummy": dummy, "onnx_path": out,
            "class_names": class_names, "model_cfg": model_cfg}


def check_parity(model: torch.nn.Module, dummy: torch.Tensor, onnx_path: Path) -> float:
    """Compare torch vs onnxruntime logits on random input; returns max |Δ|."""
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    with torch.inference_mode():
        ref = model(dummy).numpy()
    got = sess.run(None, {"clip": dummy.numpy()})[0]
    max_diff = float(np.abs(ref - got).max())
    status = "OK" if max_diff < 1e-3 else "SUSPICIOUS"
    logger.info("Parity torch↔onnxruntime: max |Δlogit| = %.2e [%s]", max_diff, status)
    return max_diff


def trtexec_command(onnx_path: str, engine_path: str, precision: str,
                    max_batch: int, shape: tuple[int, int]) -> list[str]:
    t, s = shape
    dims = f"3x{t}x{s}x{s}"
    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--minShapes=clip:1x{dims}",
        f"--optShapes=clip:{max(1, max_batch // 2)}x{dims}",
        f"--maxShapes=clip:{max_batch}x{dims}",
    ]
    if precision == "fp16":
        cmd.append("--fp16")
    return cmd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/temporal_model.yaml")
    ap.add_argument("--checkpoint", help="defaults to inference.checkpoint from config")
    ap.add_argument("--output", help="defaults to export.onnx_path from config")
    ap.add_argument("--no-check", action="store_true", help="skip onnxruntime parity check")
    ap.add_argument("--build-engine", action="store_true",
                    help="run trtexec here (do this on the deployment device)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    xcfg = cfg.get("export", {})
    checkpoint = args.checkpoint or cfg["inference"]["checkpoint"]
    onnx_path = args.output or xcfg.get("onnx_path", "models/exported/videomaev2_theft.onnx")

    result = export_onnx(checkpoint, onnx_path, opset=int(xcfg.get("opset", 17)))

    if not args.no_check:
        try:
            check_parity(result["model"], result["dummy"], result["onnx_path"])
        except ImportError:
            logger.warning("onnxruntime not installed — skipping parity check")

    mc = result["model_cfg"]
    cmd = trtexec_command(str(result["onnx_path"]),
                          xcfg.get("trt_engine_path", "models/exported/videomaev2_theft.plan"),
                          str(xcfg.get("trt_precision", "fp16")),
                          int(xcfg.get("max_batch", 8)),
                          (mc["num_frames"], mc["input_size"]))
    if args.build_engine:
        if shutil.which("trtexec") is None:
            raise SystemExit("trtexec not found on PATH — install TensorRT or run "
                             "this on the deployment device:\n  " + " ".join(cmd))
        logger.info("Building TensorRT engine: %s", " ".join(cmd))
        subprocess.run(cmd, check=True)
    else:
        print("\nTensorRT engine build (run on the deployment device):\n  " + " ".join(cmd))


if __name__ == "__main__":
    main()
