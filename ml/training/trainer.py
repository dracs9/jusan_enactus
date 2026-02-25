import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.should_stop = False

    def __call__(self, value: float) -> bool:
        if self.best_value is None:
            self.best_value = value
            return False

        if self.mode == "max":
            improved = value > self.best_value + self.min_delta
        else:
            improved = value < self.best_value - self.min_delta

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


class MetricsTracker:
    def __init__(self, num_classes: int, class_names: List[str]):
        self.num_classes = num_classes
        self.class_names = class_names
        self.reset()

    def reset(self):
        self.all_preds: List[int] = []
        self.all_labels: List[int] = []
        self.total_loss = 0.0
        self.num_batches = 0

    def update(self, preds: torch.Tensor, labels: torch.Tensor, loss: float):
        self.all_preds.extend(preds.cpu().numpy().tolist())
        self.all_labels.extend(labels.cpu().numpy().tolist())
        self.total_loss += loss
        self.num_batches += 1

    def compute(self) -> Dict:
        preds = np.array(self.all_preds)
        labels = np.array(self.all_labels)

        acc = accuracy_score(labels, preds)
        f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
        precision = precision_score(labels, preds, average="macro", zero_division=0)
        recall = recall_score(labels, preds, average="macro", zero_division=0)
        f1_per_class = f1_score(labels, preds, average=None, zero_division=0)
        avg_loss = self.total_loss / max(self.num_batches, 1)

        per_class_acc = {}
        for class_idx, class_name in enumerate(self.class_names):
            mask = labels == class_idx
            if mask.sum() > 0:
                per_class_acc[class_name] = float(accuracy_score(labels[mask], preds[mask]))

        return {
            "loss": avg_loss,
            "accuracy": acc,
            "f1_macro": f1_macro,
            "precision_macro": precision,
            "recall_macro": recall,
            "f1_per_class": {
                self.class_names[i]: float(f1_per_class[i])
                for i in range(min(len(f1_per_class), len(self.class_names)))
            },
            "per_class_accuracy": per_class_acc,
        }


def get_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    train_cfg = config["training"]
    opt_name = train_cfg.get("optimizer", "adamw").lower()
    lr = train_cfg["learning_rate"]
    wd = train_cfg.get("weight_decay", 1e-4)

    if opt_name == "adamw":
        return AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=wd,
        )
    elif opt_name == "sgd":
        return SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            momentum=0.9,
            weight_decay=wd,
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict,
    steps_per_epoch: int,
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    train_cfg = config["training"]
    scheduler_name = train_cfg.get("scheduler", "cosine").lower()
    epochs = train_cfg["epochs"]

    if scheduler_name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    elif scheduler_name == "onecycle":
        return OneCycleLR(
            optimizer,
            max_lr=train_cfg["learning_rate"],
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
        )
    else:
        return None


def save_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: str,
    title: str = "Confusion Matrix",
):
    plt.figure(figsize=(max(10, len(class_names)), max(8, len(class_names) - 2)))
    sns.heatmap(
        cm,
        annot=len(class_names) <= 20,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names if len(class_names) <= 30 else False,
        yticklabels=class_names if len(class_names) <= 30 else False,
    )
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        config: dict,
        class_names: List[str],
        device: torch.device,
    ):
        self.model = model.to(device)
        self.config = config
        self.class_names = class_names
        self.device = device
        self.num_classes = len(class_names)
        self.train_cfg = config["training"]
        self.log_cfg = config["logging"]

        self.checkpoint_dir = Path(self.log_cfg["checkpoint_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.best_model_path = self.log_cfg["best_model_path"]
        Path(self.best_model_path).parent.mkdir(parents=True, exist_ok=True)

        self.early_stopping = EarlyStopping(
            patience=self.train_cfg.get("early_stopping_patience", 10),
            mode="max",
        )
        self.best_val_f1 = 0.0
        self.history: Dict[str, List] = {
            "train_loss": [], "train_acc": [], "train_f1": [],
            "val_loss": [], "val_acc": [], "val_f1": [],
            "lr": [],
        }

    def train_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        scheduler=None,
        warmup_scheduler=None,
        epoch: int = 0,
    ) -> Dict:
        self.model.train()
        tracker = MetricsTracker(self.num_classes, self.class_names)
        grad_clip = self.train_cfg.get("gradient_clip", 1.0)
        log_interval = self.config["logging"].get("log_interval", 10)

        pbar = tqdm(loader, desc=f"Train Epoch {epoch}", leave=False)
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            optimizer.zero_grad()
            logits = self.model(images)
            loss = criterion(logits, labels)
            loss.backward()

            if grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

            optimizer.step()

            if warmup_scheduler is not None:
                warmup_scheduler.step()

            preds = logits.argmax(dim=1)
            tracker.update(preds, labels, loss.item())

            if batch_idx % log_interval == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        if scheduler is not None and warmup_scheduler is None:
            scheduler.step()

        return tracker.compute()

    @torch.no_grad()
    def validate(self, loader: DataLoader, criterion: nn.Module) -> Dict:
        self.model.eval()
        tracker = MetricsTracker(self.num_classes, self.class_names)

        for images, labels in tqdm(loader, desc="Validation", leave=False):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            logits = self.model(images)
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=1)
            tracker.update(preds, labels, loss.item())

        return tracker.compute()

    def save_checkpoint(self, epoch: int, metrics: Dict, optimizer: torch.optim.Optimizer, is_best: bool = False):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "class_names": self.class_names,
            "config": self.config,
        }
        ckpt_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pth"
        torch.save(checkpoint, ckpt_path)

        if is_best:
            torch.save(checkpoint, self.best_model_path)
            print(f"  → Best model saved: val_f1={metrics['f1_macro']:.4f}")

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_weights: Optional[torch.Tensor] = None,
    ):
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
            criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        else:
            criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        optimizer = get_optimizer(self.model, self.config)
        scheduler = get_scheduler(optimizer, self.config, len(train_loader))

        freeze_epochs = self.config["model"].get("freeze_backbone_epochs", 0)
        warmup_epochs = self.train_cfg.get("warmup_epochs", 0)

        from models.model import freeze_backbone
        if freeze_epochs > 0:
            freeze_backbone(self.model, freeze=True)
            print(f"Backbone frozen for {freeze_epochs} epochs")

        warmup_scheduler = None
        if warmup_epochs > 0:
            warmup_steps = warmup_epochs * len(train_loader)
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps
            )

        total_epochs = self.train_cfg["epochs"]
        warmup_done = False

        print(f"\nStarting training for {total_epochs} epochs on {self.device}")
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

        for epoch in range(1, total_epochs + 1):
            t0 = time.time()

            if epoch > freeze_epochs and freeze_epochs > 0 and not warmup_done:
                freeze_backbone(self.model, freeze=False)
                print(f"Epoch {epoch}: Backbone unfrozen")

            if epoch > warmup_epochs and not warmup_done:
                warmup_done = True
                warmup_scheduler = None

            train_metrics = self.train_epoch(
                train_loader, optimizer, criterion,
                scheduler=scheduler if warmup_done else None,
                warmup_scheduler=warmup_scheduler,
                epoch=epoch,
            )
            val_metrics = self.validate(val_loader, criterion)

            elapsed = time.time() - t0
            current_lr = optimizer.param_groups[0]["lr"]

            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_acc"].append(train_metrics["accuracy"])
            self.history["train_f1"].append(train_metrics["f1_macro"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_acc"].append(val_metrics["accuracy"])
            self.history["val_f1"].append(val_metrics["f1_macro"])
            self.history["lr"].append(current_lr)

            is_best = val_metrics["f1_macro"] > self.best_val_f1
            if is_best:
                self.best_val_f1 = val_metrics["f1_macro"]

            self.save_checkpoint(epoch, val_metrics, optimizer, is_best=is_best)

            print(
                f"Epoch {epoch:3d}/{total_epochs} [{elapsed:.1f}s] | "
                f"LR: {current_lr:.6f} | "
                f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']:.4f} F1: {train_metrics['f1_macro']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f} F1: {val_metrics['f1_macro']:.4f}"
            )

            if self.early_stopping(val_metrics["f1_macro"]):
                print(f"Early stopping triggered at epoch {epoch}")
                break

        self._save_training_curves()
        print(f"\nTraining complete. Best val F1: {self.best_val_f1:.4f}")
        print(f"Best model saved at: {self.best_model_path}")

    def _save_training_curves(self):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].plot(self.history["train_loss"], label="Train")
        axes[0].plot(self.history["val_loss"], label="Val")
        axes[0].set_title("Loss")
        axes[0].legend()

        axes[1].plot(self.history["train_acc"], label="Train")
        axes[1].plot(self.history["val_acc"], label="Val")
        axes[1].set_title("Accuracy")
        axes[1].legend()

        axes[2].plot(self.history["train_f1"], label="Train")
        axes[2].plot(self.history["val_f1"], label="Val")
        axes[2].set_title("Macro F1")
        axes[2].legend()

        plt.tight_layout()
        save_path = Path(self.log_cfg["checkpoint_dir"]) / "training_curves.png"
        plt.savefig(save_path, dpi=120)
        plt.close()
        print(f"Training curves saved: {save_path}")
