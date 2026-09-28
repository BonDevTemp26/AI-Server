#!/bin/bash
echo "Uninstalling NVIDIA PyTorch..."
pip3 uninstall -y torch torchvision torchaudio

echo "Installing AMD ROCm PyTorch..."
pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1

echo "Setting AMD GPU Override Environment Variable for session..."
export HSA_OVERRIDE_GFX_VERSION=11.0.0

echo "Done! To run the backend with AMD GPU enabled, use:"
echo "HSA_OVERRIDE_GFX_VERSION=11.0.0 python3 backend/app/main.py"
