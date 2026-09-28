"""Evaluation for the fine-tuned VideoMAEv2 theft classifier.

Two uses:
  * imported by the training loop for per-epoch validation
    (``collect_probs`` + ``compute_metrics``), and
  * run as a CLI for the full multi-view test evaluation + threshold sweep:

        python -m training.eval_metrics --config configs/temporal_model.yaml \
            --checkpoint models/checkpoints/videomaev2_theft_ft/best.pth --split test

The headline numbers are precision-oriented: the pipeline's real cost is false
alerts, so beyond top-1 we report PR-AUC / ROC-AUC of the *binarized theft
score* (weighted sum of theft-class probabilities — the same score the live
aggregator thresholds) and a threshold sweep to pick the operating point.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("eval_metrics")


def theft_score(probs: np.ndarray, class_names: list[str],
                theft_class_weights: dict[str, float]) -> np.ndarray:
    """Weighted sum of theft-class probabilities — the live trigger score."""
    score = np.zeros(len(probs), dtype=np.float64)
    for cls, w in theft_class_weights.items():
        score += float(w) * probs[:, class_names.index(cls)]
    return np.clip(score, 0.0, 1.0)


@torch.inference_mode()
def collect_probs(model, loader: DataLoader, device: str,
                  amp_dtype: torch.dtype | None = None
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the model over a loader; average softmax over views sharing a clip id.

    Returns ``(y_true, probs, clip_ids)`` with one row per source clip.
    """
    model.eval()
    all_probs, all_labels, all_ids = [], [], []
    offset = 0  # val batches carry no clip ids -> assign unique consecutive ones
    for batch in loader:
        clips, labels = batch[0], batch[1]
        if len(batch) > 2:
            ids = batch[2]
        else:
            ids = torch.arange(offset, offset + labels.size(0))
            offset += labels.size(0)
        with torch.autocast(device, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(clips.to(device, non_blocking=True))
        all_probs.append(torch.softmax(logits.float(), dim=-1).cpu())
        all_labels.append(labels)
        all_ids.append(torch.as_tensor(ids))

    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()
    ids = torch.cat(all_ids).numpy()

    uniq = np.unique(ids)
    agg_probs = np.stack([probs[ids == u].mean(axis=0) for u in uniq])
    agg_labels = np.array([labels[ids == u][0] for u in uniq])
    return agg_labels, agg_probs, uniq


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, class_names: list[str],
                    theft_class_weights: dict[str, float]) -> dict:
    y_pred = probs.argmax(axis=1)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(class_names)), zero_division=0)

    is_theft = np.isin(y_true, [class_names.index(c) for c in theft_class_weights]).astype(int)
    score = theft_score(probs, class_names, theft_class_weights)
    both_classes = len(np.unique(is_theft)) == 2
    return {
        "top1": float((y_pred == y_true).mean()),
        "macro_f1": float(f1.mean()),
        "per_class": {
            c: {"precision": float(prec[i]), "recall": float(rec[i]),
                "f1": float(f1[i]), "support": int(support[i])}
            for i, c in enumerate(class_names)
        },
        "confusion": confusion_matrix(y_true, y_pred,
                                      labels=range(len(class_names))).tolist(),
        "theft_pr_auc": float(average_precision_score(is_theft, score)) if both_classes else 0.0,
        "theft_roc_auc": float(roc_auc_score(is_theft, score)) if both_classes else 0.0,
    }


def threshold_sweep(y_true: np.ndarray, probs: np.ndarray, class_names: list[str],
                    theft_class_weights: dict[str, float]) -> list[dict]:
    """Precision/recall of the binary theft decision across trigger thresholds."""
    is_theft = np.isin(y_true, [class_names.index(c) for c in theft_class_weights]).astype(bool)
    score = theft_score(probs, class_names, theft_class_weights)
    rows = []
    for thr in np.arange(0.05, 1.0, 0.05):
        fired = score >= thr
        tp = int((fired & is_theft).sum())
        fp = int((fired & ~is_theft).sum())
        fn = int((~fired & is_theft).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        rows.append({"threshold": round(float(thr), 2), "precision": round(precision, 4),
                     "recall": round(recall, 4), "true_pos": tp, "false_pos": fp,
                     "false_neg": fn})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/temporal_model.yaml")
    ap.add_argument("--checkpoint", help="defaults to inference.checkpoint from config")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--manifest", help="explicit manifest CSV (overrides --split)")
    ap.add_argument("--report-dir", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    from temporal.videomae_finetune import load_for_inference
    from training.datasets import VideoClipDataset

    cfg = yaml.safe_load(Path(args.config).read_text())
    dcfg, ecfg = cfg["data"], cfg.get("eval", {})
    ckpt = args.checkpoint or cfg["inference"]["checkpoint"]
    model, class_names, model_cfg = load_for_inference(ckpt, args.device)

    manifest = args.manifest or dcfg[f"{args.split}_manifest"]
    ds = VideoClipDataset(
        manifest, class_names=class_names, num_frames=model_cfg["num_frames"],
        sampling_rate=dcfg["sampling_rate"], input_size=model_cfg["input_size"],
        mode="test", temporal_views=int(ecfg.get("temporal_views", 2)),
        spatial_views=int(ecfg.get("spatial_views", 3)))
    loader = DataLoader(ds, batch_size=int(ecfg.get("batch_size", 4)), shuffle=False,
                        num_workers=int(dcfg.get("num_workers", 4)))
    logger.info("Evaluating %d clips (%d views) from %s", ds.num_clips, len(ds), manifest)

    y_true, probs, _ = collect_probs(model, loader, args.device)
    theft_w = dcfg.get("theft_class_weights", {})
    metrics = compute_metrics(y_true, probs, class_names, theft_w)
    sweep = threshold_sweep(y_true, probs, class_names, theft_w)

    print(f"\n== {manifest} — {ds.num_clips} clips ==")
    print(f"top1 {metrics['top1']:.3f} | macro-F1 {metrics['macro_f1']:.3f} | "
          f"theft PR-AUC {metrics['theft_pr_auc']:.3f} | theft ROC-AUC {metrics['theft_roc_auc']:.3f}")
    print(f"\n{'class':<22}{'prec':>7}{'rec':>7}{'f1':>7}{'n':>6}")
    for c, m in metrics["per_class"].items():
        print(f"{c:<22}{m['precision']:>7.3f}{m['recall']:>7.3f}{m['f1']:>7.3f}{m['support']:>6}")
    print("\nthreshold sweep (binary theft trigger):")
    print(f"{'thr':>5}{'prec':>8}{'rec':>8}{'TP':>5}{'FP':>5}{'FN':>5}")
    for r in sweep:
        print(f"{r['threshold']:>5.2f}{r['precision']:>8.3f}{r['recall']:>8.3f}"
              f"{r['true_pos']:>5}{r['false_pos']:>5}{r['false_neg']:>5}")

    report_dir = Path(args.report_dir or Path(ckpt).parent / f"eval_{args.split}")
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "report.json").write_text(json.dumps(
        {"checkpoint": str(ckpt), "manifest": str(manifest),
         "metrics": metrics, "threshold_sweep": sweep}, indent=2))
    np.savetxt(report_dir / "confusion_matrix.csv", np.array(metrics["confusion"]),
               fmt="%d", delimiter=",", header=",".join(class_names))
    logger.info("Report written to %s", report_dir)


if __name__ == "__main__":
    main()
