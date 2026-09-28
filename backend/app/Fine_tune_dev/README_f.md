# 🧠 Fine_tune_dev — VideoMAEv2 Fine-Tuning Stack

Everything needed to produce the fine-tuned Stage-2 theft classifier that
plugs into [cctv_theft_app](../cctv_theft_app/) (`mode: finetuned`).

**Full step-by-step guide:** [docs/videomae_v2_implementation_guide.md](docs/videomae_v2_implementation_guide.md)

## Layout

| Path | Purpose |
|---|---|
| `temporal/videomae_model.py` | VideoMAEv2 architecture (official-checkpoint compatible) — also loaded by the app via `training_stack_path` |
| `temporal/videomae_finetune.py` | Model factory, layer-wise LR decay, checkpoint save/restore |
| `temporal/videomae_infer.py` · `anomaly_scorer.py` | Offline streaming inference + trigger logic (for evaluating checkpoints on recordings) |
| `temporal/export_videomae.py` | ONNX / TensorRT export |
| `training/` | `train_videomae.py` (fine-tune CLI), `eval_metrics.py`, `datasets.py` |
| `scripts/` | `download_pretrained.py` (K710-distilled weights), `prepare_dataset.py` (clip cutting, negative mining, leakage-safe splits) |
| `configs/temporal_model.yaml` | Training/eval/inference/export settings |
| `data/` · `models/` | Clips + manifests · checkpoints + exports |

## Typical cycle (run everything from this folder)

```bash
cd fine_tune_dev
python scripts/download_pretrained.py --model base            # once
python scripts/prepare_dataset.py --annotations data/annotations/my_annotations.csv \
    --videos-root data/raw_clips --negatives-per-video 4
python -m training.train_videomae --config configs/temporal_model.yaml
python -m training.eval_metrics   --config configs/temporal_model.yaml --split test
```

Result: `models/checkpoints/videomaev2_theft_ft/best.pth` — then switch the app:
`cctv_theft_app/configs/temporal_model.yaml → mode: finetuned`.
