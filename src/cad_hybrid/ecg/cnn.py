"""ImageNet-pretrained CNN fine-tuned on ECG images (PyTorch + timm).

Two-stage fine-tuning:
1. Freeze the backbone; train only the new classification head.
2. Unfreeze the last few blocks; continue at a lower learning rate.
"""

from __future__ import annotations


def build_model(backbone: str, num_classes: int, pretrained: bool = True):
    """Create a timm model with a fresh head for our classes."""
    import timm

    return timm.create_model(backbone, pretrained=pretrained, num_classes=num_classes)


def freeze_backbone(model) -> None:
    """Stage 1: train only the classifier head."""
    for p in model.parameters():
        p.requires_grad = False
    for p in model.get_classifier().parameters():
        p.requires_grad = True


def unfreeze_last_blocks(model, n_blocks: int) -> None:
    """Stage 2: also train the last ``n_blocks`` feature blocks.

    TODO: block naming differs by backbone (``blocks`` for EfficientNet, ``layer4`` for
    ResNet); confirm for the chosen backbone.
    """
    raise NotImplementedError


def train(config: dict, train_loader, val_loader):
    """Run both fine-tuning stages with early stopping; return the best model.

    TODO (weeks 1-2): training loop, class weighting for imbalance, checkpointing to
    ``models/`` (gitignored).
    """
    raise NotImplementedError
