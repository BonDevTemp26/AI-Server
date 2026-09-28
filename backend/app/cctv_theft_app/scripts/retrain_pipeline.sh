#!/usr/bin/env bash
# Human-in-the-loop retraining (README feedback loop):
# review labels → labeled clips → fine-tune VideoMAEv2 (Fine_tune_dev training
# stack) → evaluate → switch the app to the new checkpoint.
set -euo pipefail
cd "$(dirname "$0")/.."
STACK_ROOT=$(cd ../Fine_tune_dev && pwd)

echo "── 1/4 export reviewed clips into the training set"
python3 - <<'PY'
import json, shutil
from pathlib import Path

labels = {json.loads(l)["event_id"]: json.loads(l)["label"]
          for l in Path("data/review/labels.jsonl").read_text().splitlines() if l.strip()} \
         if Path("data/review/labels.jsonl").is_file() else {}
reviews = [json.loads(l) for l in Path("data/review/review.jsonl").read_text().splitlines()
           if l.strip()] if Path("data/review/review.jsonl").is_file() else {}
out = Path("../Fine_tune_dev/data/labeled_clips"); n = 0
for r in reviews:
    lbl, clip = labels.get(r["event_id"]), r.get("clip_path")
    if not lbl or not clip or not Path(clip).is_file():
        continue
    cls = r["top_action"] if lbl == "confirmed_theft" else "normal"
    dst = out / cls / f"review_{r['event_id']}.mp4"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(clip, dst); n += 1
print(f"exported {n} human-labeled clips → {out}")
print("→ add rows for them to your annotation CSV / manifests before training")
PY

echo "── 2/4 fine-tune (Fine_tune_dev training stack)"
echo "  cd $STACK_ROOT && python -m training.train_videomae --config configs/temporal_model.yaml"
echo "── 3/4 evaluate on the frozen test split"
echo "  cd $STACK_ROOT && python -m training.eval_metrics --split test"
echo "── 4/4 switch the app to the new model"
echo "  edit cctv_theft_app/configs/temporal_model.yaml →  mode: finetuned"
