#!/usr/bin/env bash
# Edge deployment (Jetson Orin / GPU box): stages 1–2 + buffering run here;
# only candidate clips leave the device (README edge/cloud split).
set -euo pipefail
cd "$(dirname "$0")/.."

echo "── 1/4 dependencies"
python3 -m pip install -r requirements.txt

echo "── 2/4 pretrained backbone (Stage 2 temporary model)"
python3 ../Fine_tune_dev/scripts/download_pretrained.py --model base --out-dir ../Fine_tune_dev/models/checkpoints

echo "── 3/4 TensorRT engine for the detector (device-specific)"
python3 -m detection.export_tensorrt --config configs/detector.yaml --half || \
  echo "   (TensorRT export skipped — .pt weights will be used)"

echo "── 4/4 systemd unit (optional)"
cat <<EOF
[Unit]
Description=CCTV Theft Detection Pipeline
After=network-online.target

[Service]
WorkingDirectory=$(pwd)
ExecStart=$(command -v python3) main.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
echo "Save as /etc/systemd/system/cctv-theft.service, then:"
echo "  sudo systemctl enable --now cctv-theft"
