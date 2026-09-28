#!/usr/bin/env python3
import sys
import torch
import onnx
import onnxruntime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("export")

sys.path.insert(0, "/home/ai-mini-playback/AI-Server/backend/app/cctv_theft_app")

from temporal.backbone_loader import build_backbone

def export_videomae():
    device = "cpu"
    logger.info("Loading VideoMAEv2 model...")
    # Load exactly as videomae_infer.py does
    model, _ = build_backbone(
        "../Fine_tune_dev", "vit_base_patch16_224", num_classes=710,
        num_frames=16, input_size=224,
        checkpoint="../Fine_tune_dev/models/checkpoints/vit_b_k710_dl_from_giant.pth", 
        device=device
    )
    model.eval()

    # clip shape is likely [B, C, T, H, W] for PyTorchVideo/VideoMAE
    dummy_input = torch.randn(1, 3, 16, 224, 224, device=device)

    class VideoMAEv2Features(torch.nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base = base_model
        
        def forward(self, x):
            return self.base.forward_features(x)

    export_model = VideoMAEv2Features(model)
    export_model.eval()

    output_path = "videomaev2.onnx"
    logger.info(f"Exporting to {output_path}...")
    
    with torch.no_grad():
        torch.onnx.export(
            export_model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
    logger.info("Export completed successfully!")

if __name__ == "__main__":
    export_videomae()
