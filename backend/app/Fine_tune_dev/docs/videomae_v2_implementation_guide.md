# VideoMAEv2 Fine-Tuning — End-to-End Implementation Guide

This guide walks the full life cycle of the **Stage 2 temporal model** from
[README_Architec.md](../../README_Architec.md): a VideoMAEv2 clip classifier
fine-tuned on your own labeled theft clips, deployed as a streaming
sliding-window detector that emits *theft candidate events* for the rolling
buffer and Stage-3 VLM verification.

> Architecture recap — the sweet spot: *"VideoMAEv2 fine-tuned on your own
> labeled clips, plus an RTFM-style anomaly head as a safety net."* Everything
> below implements the VideoMAEv2 half; the fusion interface for the RTFM score
> is already wired in (`temporal/anomaly_scorer.py`), so the anomaly head can
> be added later without touching this code.

## What was implemented (file map)

| File | Role |
|---|---|
| `configs/temporal_model.yaml` | Single config for training, eval, inference, export |
| `temporal/videomae_model.py` | VideoMAEv2 ViT architecture (checkpoint-compatible with the official release) + checkpoint loader |
| `temporal/videomae_finetune.py` | Model factory, layer-wise LR decay, block freezing, checkpoint save/restore |
| `temporal/video_io.py` | Video decoding (decord → PyAV → OpenCV fallback) |
| `temporal/videomae_infer.py` | `VideoMAEv2Classifier` (torch/ONNX) + `StreamingTemporalAnalyzer` (per-camera sliding window) |
| `temporal/anomaly_scorer.py` | `TheftEventAggregator`: score fusion, smoothing, hysteresis → `TheftCandidateEvent` |
| `temporal/export_videomae.py` | ONNX export + parity check + TensorRT engine build |
| `training/datasets.py` | Manifest dataset, clip-consistent augmentation, balanced sampler |
| `training/train_videomae.py` | Fine-tuning CLI (AMP, mixup, cosine + warmup, early stopping) |
| `training/eval_metrics.py` | Multi-view evaluation, per-class metrics, threshold sweep |
| `scripts/download_pretrained.py` | Fetch official K710-distilled VideoMAEv2 weights |
| `scripts/prepare_dataset.py` | Cut annotated footage into clips, mine negatives, leakage-safe splits |

---

## Step 0 — Environment

Use **Python 3.10–3.12** (best wheel coverage for torch/decord; the system
Python 3.14 here is too new for some wheels):

```bash
cd /home/siva-danu/Drive-D/BonTech/Theft_work_flow/Fine_tune_dev
python3.12 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Sanity check: `python -c "import torch; print(torch.cuda.is_available())"` → `True`.

**GPU sizing.** The default config is tuned to fit a **4 GB GPU** (RTX 3050 Ti):
ViT-B, batch 2 × accum 8, gradient checkpointing, AMP. On a 24 GB card set
`batch_size: 16`, `accum_steps: 1`, `grad_checkpointing: false` — same effective
recipe, ~6× faster. Training also works (slowly) on CPU for pipeline testing.

## Step 1 — Pretrained weights

```bash
python scripts/download_pretrained.py --model base    # ViT-B, ~700 MB (default)
python scripts/download_pretrained.py --model small   # ViT-S — for Jetson/edge
```

This fetches the **K710-distilled** checkpoints
(`vit_b_k710_dl_from_giant.pth`) from the official release
(`OpenGVLab/VideoMAE2` on Hugging Face). Distilled-from-giant weights are the
recommended starting point: they carry the accuracy of the ViT-g teacher at
ViT-S/B cost and fine-tune very efficiently on small datasets — precisely the
"fine-tunes very efficiently on small theft datasets" property the
architecture doc calls out.

If you switch to `--model small`, also set in `configs/temporal_model.yaml`:
`model.arch: vit_small_patch16_224`, `model.drop_path_rate: 0.05`,
`model.pretrained_checkpoint: models/checkpoints/vit_s_k710_dl_from_giant.pth`.

## Step 2 — Dataset strategy

### Label taxonomy

The config ships with four classes (edit `data.classes` to change; the first
class must remain the background class):

| Class | Definition | Typical duration |
|---|---|---|
| `normal` | Browsing, comparing items, putting items in a store basket/cart, staff restocking | any |
| `concealment` | Item moves into pocket, waistband, own bag, stroller, or under clothing | 2–6 s |
| `grab_and_run` | Merchandise grabbed and carried toward exit at speed | 3–8 s |
| `suspicious_handling` | Tag removal, packaging swap, repeated pick-up/put-back with scanning of surroundings | 3–10 s |

Start with fewer classes if unsure — `normal` + `concealment` alone already
drives the alert pipeline; you can add classes and re-run training later
(checkpoints are self-describing, so downstream code adapts automatically).

### How much data

- **Minimum viable**: ~150 clips per theft class + 400–600 `normal` clips → expect a usable model with `freeze_blocks: 6`.
- **Solid**: 500+ per theft class, 2000+ normal, spanning **many cameras, viewpoints, lighting conditions, and clothing seasons**. Diversity beats volume.
- **Hard negatives are the highest-value data**: putting a phone back in one's own pocket, reaching into a handbag for a wallet, staff pocketing box cutters, customers bagging items at self-checkout. These are what separate 90% from 99% precision. The human-review feedback loop (Step 8) generates them continuously.

Public data (UCF-Crime shoplifting subset, XD-Violence) is useful for
*pre-validation of the training loop* and as extra `normal` footage, but — as
README_Architec.md warns — it will not reach production accuracy on your
cameras. Treat it as a warm-up, not a substitute.

> **Privacy note.** You are processing footage of identifiable people. Confirm
> signage/consent requirements, retention limits, and who may view review-queue
> clips under your local regulations before collecting training data.

## Step 3 — Annotation

Any tool that yields temporal segments works (CVAT and Label Studio both do);
export to the CSV schema consumed by `scripts/prepare_dataset.py`
(one row per event — see [example_annotations.csv](../data/annotations/example_annotations.csv)):

```csv
video,start_s,end_s,label,camera_id
cam03/2026-07-01_afternoon.mp4,732.5,738.0,concealment,cam03
```

Labeling rules that keep the classifier learnable:

1. **Segment = the action, not the person's whole visit.** Start when the hand
   engages the merchandise, end ~1 s after the action completes (the script
   adds `--context-s 1.0` padding on both sides anyway).
2. Keep segments **2–8 s**. Longer incidents → annotate the decisive moment(s).
3. Annotate `normal` segments explicitly for busy scenes (crowds, staff), and
   let `--negatives-per-video` mine quiet-period negatives automatically.
4. **Double-annotate 10%** of clips with a second person; disagreement > 10% on
   a class means its definition needs tightening before you train on it.
5. Log ambiguous events as `suspicious_handling` rather than forcing them into
   `concealment` — label noise in the trigger classes is what poisons precision.

## Step 4 — Preprocessing

```bash
python scripts/prepare_dataset.py \
    --annotations data/annotations/my_annotations.csv \
    --videos-root data/raw_clips \
    --negatives-per-video 4
```

What it does, and why:

- **Frame-accurate cuts** with ffmpeg, re-encoded to **25 fps, short side 320,
  H.264** — small, fast-to-decode files; the dataloader samples 16 frames with
  stride 4 (≈2.56 s of context) and resizes to 224², so 320p sources lose nothing.
- **Negative mining**: samples `normal` windows from unlabeled gaps ≥ 5 s away
  from any annotation.
- **Leakage-safe splits**: train/val/test are split **by source video**
  (`--group-by camera` for the stricter variant). Random per-clip splits leak
  near-identical frames into val and inflate metrics by 10–20 points — the
  numbers would collapse in production.
- Writes `data/annotations/{train,val,test}.csv` manifests
  (`path,label,video_id,camera_id,...`) and prints a per-split × per-class
  count table. **Check that table**: every class present in every split, and
  val/test containing cameras the model never trained on.

## Step 5 — Fine-tuning

```bash
python -m training.train_videomae --config configs/temporal_model.yaml
```

### What the recipe does

The loop implements the standard VideoMAE fine-tuning protocol, adapted for
small imbalanced datasets:

| Mechanism | Config keys | Why |
|---|---|---|
| AdamW + **layer-wise LR decay** 0.75 | `base_lr`, `layer_decay` | early layers hold generic motion features; the head learns fastest |
| LR = `base_lr × eff_batch/256`, cosine + 5-epoch warmup | `warmup_epochs`, `min_lr` | stable fine-tuning at any batch size |
| **Balanced sampling** | `data.balanced_sampling` | theft clips are rare; uniform class mix each epoch |
| Mixup/cutmix + label smoothing + drop-path + random erasing | `augment.*`, `mixup_*`, `drop_path_rate` | ViTs overfit small video datasets in a few epochs without this |
| AMP (bf16/fp16) + grad accumulation + grad checkpointing | `amp`, `accum_steps`, `grad_checkpointing` | fits ViT-B on 4 GB |
| Best-checkpoint on **theft PR-AUC** + early stopping | `monitor`, `early_stop_patience` | selects for the alerting task, not raw top-1 |

### Recipes

| Situation | Command |
|---|---|
| Standard (≥300 clips/class) | defaults as shipped |
| Small dataset (<300 total theft clips) | `--freeze-blocks 6 --no-mixup --epochs 25` and set `augment.color_jitter: 0.3` |
| Edge model for Jetson | ViT-S config (Step 1) + defaults |
| Sanity run (any data, CPU ok) | `--epochs 2 --batch-size 1` |

Monitor with `tensorboard --logdir models/checkpoints/videomaev2_theft_ft/tb`.
Healthy run: train loss falls smoothly through warmup; `val/theft_pr_auc`
climbs for 15–30 epochs then plateaus (early stopping handles the rest).
Val accuracy pinned at chance → check the manifest/label table from Step 4.
Train loss ≈ 0 while val degrades → overfitting: freeze blocks, raise
augmentation, or get more data.

Outputs in `train.output_dir`: `best.pth`, `last.pth` (both self-describing:
weights + class names + architecture), `label_map.json`, `config_snapshot.yaml`.

## Step 6 — Evaluation

```bash
python -m training.eval_metrics --config configs/temporal_model.yaml --split test
```

Runs multi-view inference (2 temporal × 3 spatial crops averaged per clip —
the standard protocol, worth ~1–2 points over single-view) and reports:

- top-1, macro-F1, per-class precision/recall/F1;
- **theft PR-AUC / ROC-AUC** on the *binarized theft score* — the same
  weighted score (`data.theft_class_weights`) the live aggregator thresholds;
- a **threshold sweep** table (precision/recall/TP/FP/FN at each threshold).

**Choosing the operating point** — false positives are the enemy, so pick the
threshold from the sweep that meets your precision target (e.g. smallest
threshold with precision ≥ 0.9) and set it as `inference.trigger_threshold`.
Remember the live trigger is *more* conservative than the offline number: it
also requires `min_consecutive` windows and EMA smoothing, and Stage-3 VLM
verification sits behind it. It is fine to run Stage 2 at ~0.8 precision /
high recall and let the VLM buy the rest of the precision.

Finally, validate **event-level** behavior on a few held-out full-length
videos (not clips) with the demo below — clip metrics don't capture duplicate
alerts or trigger latency; hysteresis and cooldown handle those.

## Step 7 — Streaming inference

```bash
python -m temporal.videomae_infer --video path/to/held_out_recording.mp4 \
    --config configs/temporal_model.yaml
```

How the live path works (`StreamingTemporalAnalyzer`):

1. Every decoded frame goes to `process_frame(camera_id, frame_rgb, ts, ...)`.
2. Frames are subsampled to the training rate (25/4 ≈ 6.25 fps), preprocessed
   (short-side resize → center crop 224 → ImageNet normalize) and kept in a
   16-slot ring buffer ≈ **2.6 s of temporal context** per camera.
3. Every `window_stride_s` (1 s) the window is classified; probabilities go to
   the `TheftEventAggregator`, which applies **EMA smoothing → hysteresis
   (trigger 0.70 / re-arm 0.40) → `min_consecutive` → cooldown** and emits a
   `TheftCandidateEvent` with the `T-15s → T+15s` snapshot window — exactly the
   contract the rolling buffer expects (README stage 2 → buffer → VLM).
4. Pass `persons_present=False` from Stage-1 tracking to skip classification
   on empty scenes (typically 70–90% of retail hours — the single biggest
   compute saver), and `anomaly_score=` once the RTFM head exists; fusion mode
   and weights are in `inference.fusion`.

Compute: one window = one ViT-B forward at 16×224² ≈ 15–25 ms on a modern
discrete GPU (TensorRT fp16), ≈ 40–80 ms on Jetson Orin with ViT-S — at 1 window/s
per camera, one Orin comfortably serves several cameras alongside YOLO.

## Step 8 — Export & deployment

```bash
python -m temporal.export_videomae --config configs/temporal_model.yaml            # ONNX + parity check
python -m temporal.export_videomae --config configs/temporal_model.yaml --build-engine  # on the deployment box
```

- ONNX graph (dynamic batch, opset 17) + `*.labels.json` sidecar → the ONNX
  backend is fully self-describing: `VideoMAEv2Classifier(path, backend="onnx")`.
- Parity check reports max |Δlogit| torch↔onnxruntime (expect < 1e-4).
- **Build TensorRT engines on the target device** (engines are not portable);
  the tool prints the exact `trtexec --fp16` command with min/opt/max batch
  shapes. On Jetson, JetPack ships `trtexec` at
  `/usr/src/tensorrt/bin/trtexec`. `onnxruntime-gpu` with the TensorRT
  execution provider gives ~90% of native-TRT speed with zero extra code and
  is the recommended first deployment.
- Validate the exported model end-to-end with the Step-7 demo using
  `--backend onnx` before shipping.

## Step 9 — Integration with the rest of the pipeline

Glue sketch for the edge process (Stage 1 + Stage 2 per camera):

```python
import yaml
from temporal import VideoMAEv2Classifier, StreamingTemporalAnalyzer, TheftEventAggregator

cfg = yaml.safe_load(open("configs/temporal_model.yaml"))

def on_candidate(event):                      # Stage 2 → Stage 3 handoff
    clip_path = ring_buffer.snapshot(          # buffering/ring_buffer.py
        event.camera_id, event.clip_start_ts, event.clip_end_ts)
    vlm_queue.submit(clip_path, event)         # verification/vlm_client.py
                                               # confirmed → alerts/, rejected → hard-negative set

classifier = VideoMAEv2Classifier(cfg["inference"]["checkpoint"])   # or backend="onnx"
aggregator = TheftEventAggregator.from_config(cfg["inference"], cfg["data"], on_event=on_candidate)
analyzer   = StreamingTemporalAnalyzer(classifier, aggregator,
                                       clip_fps=cfg["data"]["clip_fps"],
                                       sampling_rate=cfg["data"]["sampling_rate"],
                                       window_stride_s=cfg["inference"]["window_stride_s"])

for camera_id, ts, frame_rgb, tracks in detection_stream():         # Stage 1 output
    analyzer.process_frame(camera_id, frame_rgb, ts,
                           persons_present=bool(tracks.person_ids))
```

**Retraining loop** (the 90%→99% mechanism): every VLM/human verdict is a new
label. Append confirmed thefts and rejected false positives (as `normal` /
hard negatives) to the annotation CSV, re-run Steps 4–6, and redeploy when the
new checkpoint beats the old one on the *frozen* test split. Keep the test
split frozen so numbers stay comparable across retrains; version each
deployment by its `config_snapshot.yaml` + checkpoint hash.

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `decord` fails to install | Fine — `video_io.py` falls back to PyAV (`pip install av`), then OpenCV |
| CUDA OOM during training | Lower `batch_size` (raise `accum_steps` to keep eff. batch 16), keep `grad_checkpointing: true`, or switch to ViT-S |
| Val metrics look great, live demo noisy | Split leakage (re-check Step 4) or val cameras too similar to train; add unseen-camera test data |
| Many triggers on one incident | Raise `cooldown_s` / lower `rearm_threshold` (hysteresis gap too small) |
| Misses fast grab-and-run | Lower `min_consecutive` to 1 and `window_stride_s` to 0.5 for that class of scene |
| Day model fails at night / new store | Domain shift — add that footage to training; this is expected, not a bug |
| Pretrained checkpoint "missing backbone keys" warning | Wrong arch/checkpoint pairing (e.g. ViT-B config with ViT-S weights) |
