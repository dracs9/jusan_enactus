from typing import Optional
import torch
import torch.nn as nn
import timm


def build_model(
    architecture: str,
    num_classes: int,
    pretrained: bool = True,
    dropout: float = 0.3,
    checkpoint_path: Optional[str] = None,
) -> nn.Module:
    model = timm.create_model(
        architecture,
        pretrained=pretrained,
        num_classes=0,
    )

    in_features = model.num_features

    classifier = nn.Sequential(
        nn.AdaptiveAvgPool2d(1) if hasattr(model, "global_pool") else nn.Identity(),
        nn.Flatten(),
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.SiLU(),
        nn.Dropout(p=dropout),
        nn.Linear(512, num_classes),
    )

    model.classifier = classifier
    model.reset_classifier(num_classes=0)

    model = _attach_head(architecture, model, in_features, num_classes, dropout)

    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location="cpu")
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint from {checkpoint_path}")

    return model


def _attach_head(
    architecture: str,
    backbone: nn.Module,
    in_features: int,
    num_classes: int,
    dropout: float,
) -> nn.Module:
    class OskinModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.head = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.BatchNorm1d(512),
                nn.SiLU(),
                nn.Dropout(p=dropout),
                nn.Linear(512, num_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            features = self.backbone(x)
            if features.dim() > 2:
                features = features.flatten(1)
            return self.head(features)

        def get_features(self, x: torch.Tensor) -> torch.Tensor:
            features = self.backbone(x)
            if features.dim() > 2:
                features = features.flatten(1)
            return features

    return OskinModel()


def freeze_backbone(model: nn.Module, freeze: bool = True):
    for name, param in model.backbone.named_parameters():
        param.requires_grad = not freeze
    for param in model.head.parameters():
        param.requires_grad = True


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}
