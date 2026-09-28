FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# CPU torch by default; for CUDA use an nvidia/cuda base image and the cu121 index.
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir torch torchvision \
        --index-url https://download.pytorch.org/whl/rocm6.1

COPY . .
# docker-compose.yml mounts the repo-root training stack at /temporal and the
# checkpoints at /models, so the config defaults (training_stack_path: "..",
# checkpoints under ../models/...) resolve the same inside the container.

CMD ["python", "main.py"]
