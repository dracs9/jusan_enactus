#!/usr/bin/env python3
"""
Oskín ML — Training Entry Point

Stage 1: python train.py --config configs/stage1_plantvillage.yaml
Stage 2: python train.py --config configs/stage2_finetune.yaml
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from training.utils import set_seed, load_config, save_class_mapping, load_class_mapping, get_device, print_config
from training.dataset import get_dataloaders
from training.trainer import Trainer
from models.model import build_model, count_parameters


def parse_args():
    parser = argparse.ArgumentParser(description="Oskín Plant Disease Classifier - Training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    return parser.parse_args()


def main():
    args = parse_args()

    config = load_config(args.config)
    stage = config["experiment"]["stage"]

    set_seed(config["experiment"]["seed"])
    device = get_device()

    print("=" * 60)
    print(f"Oskín ML Training — Stage {stage}: {config['experiment']['name']}")
    print("=" * 60)
    print_config(config)
    print("=" * 60)

    class_mapping_path = config["data"]["class_mapping_path"]
    existing_mapping = None

    if stage == 2 and Path(class_mapping_path).exists():
        existing_mapping = load_class_mapping(class_mapping_path)
        print(f"Stage 2: Using existing class mapping ({len(existing_mapping)} classes)")

    train_loader, val_loader, test_loader, class_mapping = get_dataloaders(
        config=config,
        class_mapping=existing_mapping,
    )

    save_class_mapping(class_mapping, class_mapping_path)

    num_classes = len(class_mapping)
    class_names = [k for k, v in sorted(class_mapping.items(), key=lambda x: x[1])]

    print(f"\nDataset: {num_classes} classes")
    print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")

    checkpoint_path = None
    if stage == 2:
        checkpoint_path = config["model"].get("checkpoint")
        if checkpoint_path and not Path(checkpoint_path).exists():
            print(f"WARNING: Stage 2 checkpoint not found at {checkpoint_path}")
            checkpoint_path = None

    model = build_model(
        architecture=config["model"]["architecture"],
        num_classes=num_classes,
        pretrained=config["model"]["pretrained"],
        dropout=config["model"]["dropout"],
        checkpoint_path=checkpoint_path,
    )

    param_info = count_parameters(model)
    print(f"\nModel: {config['model']['architecture']}")
    print(f"  Total params:     {param_info['total']:,}")
    print(f"  Trainable params: {param_info['trainable']:,}")
    print(f"  Frozen params:    {param_info['frozen']:,}")

    full_train_dataset = train_loader.dataset
    try:
        from torch.utils.data import Subset
        if hasattr(full_train_dataset, "dataset"):
            base = full_train_dataset.dataset
        else:
            base = full_train_dataset
        class_weights = base.get_class_weights()
    except Exception:
        class_weights = None

    trainer = Trainer(
        model=model,
        config=config,
        class_names=class_names,
        device=device,
    )

    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
    )

    print("\nRunning final evaluation on test set...")
    import torch.nn as nn
    criterion = nn.CrossEntropyLoss()
    test_metrics = trainer.validate(test_loader, criterion)

    print("\n" + "=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)
    print(f"Accuracy:       {test_metrics['accuracy']:.4f}")
    print(f"Macro F1:       {test_metrics['f1_macro']:.4f}")
    print(f"Macro Precision:{test_metrics['precision_macro']:.4f}")
    print(f"Macro Recall:   {test_metrics['recall_macro']:.4f}")
    print("\nPer-class F1:")
    for cls, f1 in sorted(test_metrics["f1_per_class"].items(), key=lambda x: x[1]):
        print(f"  {cls:<40} {f1:.4f}")


if __name__ == "__main__":
    main()
