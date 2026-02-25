#!/usr/bin/env python3
"""
Oskín ML — Evaluation Script

Usage:
    python evaluate.py --config configs/stage2_finetune.yaml --checkpoint models/best_model_stage2.pth
    python evaluate.py --config configs/stage1_plantvillage.yaml --checkpoint models/best_model_stage1.pth
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent))

from training.utils import set_seed, load_config, load_class_mapping, get_device
from training.dataset import get_dataloaders
from training.trainer import save_confusion_matrix, MetricsTracker
from models.model import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Oskín ML - Evaluation")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="Output directory")
    return parser.parse_args()


@torch.no_grad()
def run_evaluation(model, loader, device, class_names):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    tracker = MetricsTracker(len(class_names), class_names)
    all_preds = []
    all_labels = []
    all_probs = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        tracker.update(preds, labels, loss.item())
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())

    return tracker.compute(), np.array(all_preds), np.array(all_labels), np.array(all_probs)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    set_seed(config["experiment"]["seed"])
    device = get_device()

    class_mapping_path = config["data"]["class_mapping_path"]
    class_mapping = load_class_mapping(class_mapping_path)
    class_names = [k for k, v in sorted(class_mapping.items(), key=lambda x: x[1])]
    num_classes = len(class_names)

    print(f"Evaluating {num_classes} classes")
    print(f"Loading checkpoint: {args.checkpoint}")

    state = torch.load(args.checkpoint, map_location="cpu")
    ckpt_state = state.get("model_state_dict", state)

    model = build_model(
        architecture=config["model"]["architecture"],
        num_classes=num_classes,
        pretrained=False,
        dropout=config["model"]["dropout"],
    )
    model.load_state_dict(ckpt_state, strict=False)
    model = model.to(device)

    _, _, test_loader, _ = get_dataloaders(config=config, class_mapping=class_mapping)

    print(f"Test set size: {len(test_loader.dataset)}")
    print("Running inference...")

    metrics, all_preds, all_labels, all_probs = run_evaluation(model, test_loader, device, class_names)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Accuracy:         {metrics['accuracy']:.4f}")
    print(f"Macro F1:         {metrics['f1_macro']:.4f}")
    print(f"Macro Precision:  {metrics['precision_macro']:.4f}")
    print(f"Macro Recall:     {metrics['recall_macro']:.4f}")
    print(f"Average Loss:     {metrics['loss']:.4f}")

    print("\nClassification Report:")
    print("-" * 60)
    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        zero_division=0,
    )
    print(report)

    report_path = output_dir / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Config: {args.config}\n\n")
        f.write(f"Accuracy:        {metrics['accuracy']:.4f}\n")
        f.write(f"Macro F1:        {metrics['f1_macro']:.4f}\n")
        f.write(f"Macro Precision: {metrics['precision_macro']:.4f}\n")
        f.write(f"Macro Recall:    {metrics['recall_macro']:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    print(f"\nReport saved: {report_path}")

    cm = confusion_matrix(all_labels, all_preds)
    cm_path = str(output_dir / "confusion_matrix.png")
    save_confusion_matrix(cm, class_names, cm_path, title="Test Set Confusion Matrix")
    print(f"Confusion matrix saved: {cm_path}")

    print("\nPer-class Accuracy (sorted by accuracy):")
    per_class = sorted(metrics["per_class_accuracy"].items(), key=lambda x: x[1])
    for cls, acc in per_class:
        print(f"  {cls:<40} {acc:.4f}")

    np.save(output_dir / "predictions.npy", all_preds)
    np.save(output_dir / "labels.npy", all_labels)
    np.save(output_dir / "probabilities.npy", all_probs)
    print(f"\nPredictions saved to: {output_dir}/")


if __name__ == "__main__":
    main()
