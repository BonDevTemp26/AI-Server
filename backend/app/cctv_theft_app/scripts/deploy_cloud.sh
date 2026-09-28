#!/usr/bin/env bash
# Cloud side: MQTT broker + review dashboard (+ optionally a self-hosted
# Qwen2.5-VL behind vLLM instead of a free API — point
# configs/vlm_verifier.yaml endpoint at it).
set -euo pipefail
cd "$(dirname "$0")/.."

[ -f .env ] || { cp .env.example .env; echo "Created .env — fill in secrets."; }
docker compose up -d mosquitto dashboard
echo "MQTT broker  : localhost:1883"
echo "Review queue : http://localhost:8080"
echo
echo "Self-hosted VLM (needs a 24GB+ GPU):"
echo "  vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8000"
echo "  → configs/vlm_verifier.yaml endpoint: http://<host>:8000/v1/chat/completions"
