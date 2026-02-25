import os
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2


def build_augmentation_pipeline(config: dict, is_train: bool, image_size: int = 224) -> A.Compose:
    mean = config.get("normalize_mean", [0.485, 0.456, 0.406])
    std = config.get("normalize_std", [0.229, 0.224, 0.225])

    if not is_train:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])

    transform_list = [A.Resize(image_size, image_size)]

    if config.get("horizontal_flip", True):
        transform_list.append(A.HorizontalFlip(p=0.5))

    if config.get("random_brightness_contrast", True):
        transform_list.append(
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7)
        )

    if config.get("gaussian_blur", True):
        transform_list.append(A.GaussianBlur(blur_limit=(3, 7), p=0.3))

    if config.get("shadow_simulation", True):
        transform_list.append(
            A.RandomShadow(
                shadow_roi=(0, 0.5, 1, 1),
                num_shadows_lower=1,
                num_shadows_upper=2,
                shadow_dimension=5,
                p=0.3,
            )
        )

    if config.get("perspective_distortion", True):
        transform_list.append(
            A.Perspective(scale=(0.02, 0.08), p=0.4)
        )

    if config.get("random_noise", True):
        transform_list.append(
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3)
        )

    transform_list.extend([
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.ColorJitter(hue=0.1, saturation=0.2, p=0.4),
        A.CoarseDropout(max_holes=4, max_height=32, max_width=32, p=0.2),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

    return A.Compose(transform_list)


class PlantDiseaseDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        transform: Optional[A.Compose] = None,
        class_mapping: Optional[Dict[str, int]] = None,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []

        if class_mapping is None:
            class_names = sorted([
                d.name for d in self.root_dir.iterdir()
                if d.is_dir() and not d.name.startswith(".")
            ])
            self.class_mapping = {name: idx for idx, name in enumerate(class_names)}
        else:
            self.class_mapping = class_mapping

        self.idx_to_class = {v: k for k, v in self.class_mapping.items()}
        self._load_samples()

    def _load_samples(self):
        valid_extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
        for class_name, class_idx in self.class_mapping.items():
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue
            for img_path in class_dir.iterdir():
                if img_path.suffix in valid_extensions:
                    self.samples.append((img_path, class_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, label

    def get_class_weights(self) -> torch.Tensor:
        label_counts = torch.zeros(len(self.class_mapping))
        for _, label in self.samples:
            label_counts[label] += 1
        weights = 1.0 / (label_counts + 1e-6)
        weights = weights / weights.sum() * len(self.class_mapping)
        return weights


def create_data_splits(
    root_dir: str,
    val_split: float,
    test_split: float,
    seed: int,
    class_mapping: Optional[Dict[str, int]] = None,
) -> Tuple[List[int], List[int], List[int], Dict[str, int]]:
    dataset = PlantDiseaseDataset(root_dir=root_dir, class_mapping=class_mapping)

    if class_mapping is None:
        class_mapping = dataset.class_mapping

    n = len(dataset)
    indices = list(range(n))
    random.seed(seed)
    random.shuffle(indices)

    test_size = int(n * test_split)
    val_size = int(n * val_split)

    test_indices = indices[:test_size]
    val_indices = indices[test_size:test_size + val_size]
    train_indices = indices[test_size + val_size:]

    return train_indices, val_indices, test_indices, class_mapping


def get_dataloaders(
    config: dict,
    class_mapping: Optional[Dict[str, int]] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    data_cfg = config["data"]
    aug_cfg = config["augmentation"]
    image_size = data_cfg["image_size"]

    train_indices, val_indices, test_indices, class_mapping = create_data_splits(
        root_dir=data_cfg["root"],
        val_split=data_cfg["val_split"],
        test_split=data_cfg["test_split"],
        seed=config["experiment"]["seed"],
        class_mapping=class_mapping,
    )

    train_transform = build_augmentation_pipeline(aug_cfg, is_train=True, image_size=image_size)
    eval_transform = build_augmentation_pipeline(aug_cfg, is_train=False, image_size=image_size)

    full_dataset_train = PlantDiseaseDataset(
        root_dir=data_cfg["root"],
        transform=train_transform,
        class_mapping=class_mapping,
    )
    full_dataset_eval = PlantDiseaseDataset(
        root_dir=data_cfg["root"],
        transform=eval_transform,
        class_mapping=class_mapping,
    )

    train_dataset = Subset(full_dataset_train, train_indices)
    val_dataset = Subset(full_dataset_eval, val_indices)
    test_dataset = Subset(full_dataset_eval, test_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, class_mapping
