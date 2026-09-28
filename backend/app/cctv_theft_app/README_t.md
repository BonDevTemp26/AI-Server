# 🎥 CCTV Theft Detection App

Runtime application implementing all four phases of
[README_Architec.md](../README_Architec.md): RTSP ingest → YOLO26-S + ByteTrack
→ temporal understanding (VideoMAEv2, swappable) → Qwen2.5-VL verification →
clip extraction + free alert channels + human review queue.

## Phase → module map

| Phase | Implementation | Config |
|---|---|---|
| **Ingest** (RTSP → DeepStream) | [ingest/rtsp_client.py](ingest/rtsp_client.py) (default, reconnecting OpenCV/FFmpeg) · [ingest/gstreamer_pipeline.py](ingest/gstreamer_pipeline.py) (Jetson HW decode + DeepStream reference pipeline) | `app.yaml → runtime.ingest_backend` |
| **1. Detection & tracking** | [detection/yolo26_infer.py](detection/yolo26_infer.py) — YOLO26-S + ByteTrack via Ultralytics (auto-fallback YOLO11-S) · [detection/export_tensorrt.py](detection/export_tensorrt.py) | `detector.yaml`, `tracker.yaml` |
| **2. Temporal understanding** | [temporal/base.py](temporal/base.py) — **swappable `TemporalModel` interface** · temporary: [temporal/videomae_infer.py](temporal/videomae_infer.py) (VideoMAEv2 novelty) · **your model: [temporal/finetuned_adapter.py](temporal/finetuned_adapter.py)** · safety net: [temporal/rtfm_infer.py](temporal/rtfm_infer.py) · trigger: [temporal/anomaly_scorer.py](temporal/anomaly_scorer.py) | `temporal_model.yaml → mode` |
| **3. VLM verification** | [verification/vlm_client.py](verification/vlm_client.py) — free Hugging Face VLMs, two backends: hosted HF Inference Providers router (`openai_compatible`, default) or fully-local transformers inference (`hf_local`: Qwen2.5-VL / SmolVLM2 / LLaVA-OneVision ...) | `vlm_verifier.yaml` |
| **4. Clips & alerts** | [buffering/ring_buffer.py](buffering/ring_buffer.py) + [buffering/clip_extractor.py](buffering/clip_extractor.py) (T-15s → T+15s) · [ingest/segment_muxer.py](ingest/segment_muxer.py) (ffmpeg ring for RTSP prod) · [alerts/](alerts/) — MQTT, Telegram, webhooks (all free) | `app.yaml → buffering`, `alerts.yaml` |
| **Human-in-the-loop** | [dashboard/server.py](dashboard/server.py) — unified dashboard with a page per stage (live streaming · detection & tracking · temporal · VLM · alert review) + [scripts/retrain_pipeline.sh](scripts/retrain_pipeline.sh) | dashboard reads/writes `configs/*.yaml` |

## Quickstart

```bash
cd cctv_theft_app
python3.12 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
cp .env.example .env                                  # add HF_TOKEN etc.

# 1. Smoke-test the plumbing — no GPU, no weights, no API key:
python main.py --source any_video.mp4 --detector mock --temporal mock --vlm mock

# 2. Real run (downloads YOLO weights on first use; Stage-2 backbone:
#    cd ../Fine_tune_dev && python scripts/download_pretrained.py --model base)
python main.py --source rtsp://user:pass@camera-ip:554/stream1

# 3. Dashboard (add cameras, pick models per stage, review + label alerts):
python -m dashboard.server                           # http://localhost:8080
```

Scripted end-to-end demo (forces a trigger at t=8–14 s of any video):

```yaml
# configs/temporal_model.yaml
mock:
  script: [[8.0, 14.0, 0.95]]
```
```bash
python main.py --source demo.mp4 --detector mock --temporal mock --vlm mock
# → candidate event → clip in data/clips/ → alert in data/alerts/alerts.jsonl
```

## Swapping in your fine-tuned model (the whole point)

The pipeline never talks to a concrete model — only to the `TemporalModel`
interface in [temporal/base.py](temporal/base.py). Today it runs the
**temporary** VideoMAEv2 novelty scorer (`mode: novelty`): the frozen
K710-distilled backbone embeds each 2.6 s window and flags behavioral novelty
against a per-camera baseline. It detects "something unusual", not "theft" —
run it with a high threshold and let Qwen2.5-VL carry precision.

When your fine-tuned checkpoint is ready (Fine_tune_dev training stack,
`python -m training.train_videomae`):

```yaml
# configs/temporal_model.yaml — the only change needed
mode: finetuned
finetuned:
  checkpoint: ../Fine_tune_dev/models/checkpoints/videomaev2_theft_ft/best.pth
```

[temporal/finetuned_adapter.py](temporal/finetuned_adapter.py) loads the
self-describing checkpoint (classes, architecture, weights) and starts emitting
real per-class probabilities (`concealment`, `grab_and_run`, ...). Set
`trigger.trigger_threshold` from the evaluation threshold sweep
(`python -m training.eval_metrics --split test`, from Fine_tune_dev/). The RTFM anomaly
head ([temporal/rtfm_infer.py](temporal/rtfm_infer.py)) plugs into either mode
via `rtfm.enabled: true` once trained.

## Free-resource choices

- **VLM (Hugging Face)**: default is the HF Inference Providers router
  (`Qwen/Qwen2.5-VL-72B-Instruct`, free monthly credits — put an `HF_TOKEN` in
  `.env`). For unlimited key-free verification switch `backend: hf_local` in
  `vlm_verifier.yaml`: the model (default `SmolVLM2-500M-Video-Instruct`, sized
  for a 4 GB GPU) downloads from the Hub and runs in-process; needs
  `pip install transformers accelerate`. OpenRouter/vLLM still work via the
  hosted backend's endpoint override.
- **Alerts**: local Mosquitto MQTT (`docker compose up mosquitto`), Telegram Bot
  API (sends the clip as video), Slack/Discord webhooks. Paid channels
  (FCM/Twilio) are extension points in [alerts/alert_manager.py](alerts/alert_manager.py).
- **Everything else** runs on your own hardware.

## Notes & limits

- The **ring buffer** (in-memory JPEG, ~60 s/camera) suits development and a
  few cameras; for many real RTSP feeds switch `buffering.backend: segment`
  concept — run [ingest/segment_muxer.py](ingest/segment_muxer.py) per camera
  (zero-CPU `-c copy` disk ring; wall-clock timestamps).
- Ultralytics keeps ByteTrack state per stream, so each camera thread owns its
  detector instance; the Stage-2 model is shared behind a lock.
- DeepStream `nvinfer`-batched multi-camera detection is a deployment task:
  the reference pipeline is generated in
  [ingest/gstreamer_pipeline.py](ingest/gstreamer_pipeline.py) (`deepstream_pipeline_desc`);
  the appsink path works on Jetson today.
- Alerts and review items are plain JSONL under `data/` — no database to lose.
