#!/usr/bin/env python3
"""
Oskín ML — Export to ONNX

Usage:
    python export.py --config configs/export.yaml
    python export.py --config configs/export.yaml --checkpoint models/best_model_stage2.pth
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import onnx
import onnxruntime as ort
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from training.utils import load_config, load_class_mapping, get_device
from models.model import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Oskín ML - ONNX Export")
    parser.add_argument("--config", type=str, default="configs/export.yaml", help="Export config path")
    parser.add_argument("--checkpoint", type=str, default=None, help="Override checkpoint path")
    return parser.parse_args()


def export_to_onnx(
    model: torch.nn.Module,
    onnx_path: str,
    image_size: int,
    opset_version: int,
    device: torch.device,
) -> str:
    model.eval()
    Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.randn(1, 3, image_size, image_size).to(device)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        opset_version=opset_version,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        do_constant_folding=True,
    )

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX model validated: {onnx_path}")
    return onnx_path


def validate_onnx(onnx_path: str, image_size: int) -> float:
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    dummy = np.random.randn(1, 3, image_size, image_size).astype(np.float32)
    outputs = session.run(None, {"input": dummy})
    print(f"ONNX inference test passed. Output shape: {outputs[0].shape}")
    return float(np.max(outputs[0]))


def get_model_size(path: str) -> float:
    return Path(path).stat().st_size / (1024 * 1024)


def main():
    args = parse_args()
    config = load_config(args.config)
    export_cfg = config["export"]

    checkpoint_path = args.checkpoint or export_cfg["model_checkpoint"]
    class_mapping_path = export_cfg["class_mapping_path"]
    onnx_path = export_cfg["onnx_path"]
    image_size = export_cfg["image_size"]
    opset_version = export_cfg.get("opset_version", 13)

    device = get_device()

    class_mapping = load_class_mapping(class_mapping_path)
    num_classes = len(class_mapping)
    class_names = [k for k, v in sorted(class_mapping.items(), key=lambda x: x[1])]

    print(f"Exporting model: {export_cfg['architecture']}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Classes: {num_classes}")

    state = torch.load(checkpoint_path, map_location="cpu")
    ckpt_state = state.get("model_state_dict", state)

    model = build_model(
        architecture=export_cfg["architecture"],
        num_classes=num_classes,
        pretrained=False,
        dropout=0.0,
    )
    model.load_state_dict(ckpt_state, strict=False)
    model = model.to(device)
    model.eval()

    print(f"\nExporting to ONNX: {onnx_path}")
    export_to_onnx(model, onnx_path, image_size, opset_version, device)

    validate_onnx(onnx_path, image_size)

    onnx_size = get_model_size(onnx_path)
    print(f"ONNX model size: {onnx_size:.2f} MB")

    export_meta = {
        "onnx_path": onnx_path,
        "architecture": export_cfg["architecture"],
        "image_size": image_size,
        "num_classes": num_classes,
        "class_names": class_names,
        "class_mapping": class_mapping,
        "input_mean": [0.485, 0.456, 0.406],
        "input_std": [0.229, 0.224, 0.225],
    }
    meta_path = str(Path(onnx_path).parent / "export_meta.json")
    with open(meta_path, "w") as f:
        json.dump(export_meta, f, indent=2)

    print(f"\nExport metadata saved: {meta_path}")
    print("\nExport complete!")
    print(f"  ONNX: {onnx_path} ({onnx_size:.2f} MB)")
    print(f"\nNext step: python convert_to_tflite.py --config configs/export.yaml")


if __name__ == "__main__":
    main()
