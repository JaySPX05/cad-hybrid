"""Clean and augment ECG images.

Rules:
- Crop to the trace area and straighten tilted photos before resizing.
- Augment only in ways that keep an ECG valid: small rotations, brightness,
  contrast, blur. Never flip; a mirrored ECG is not a real ECG.
"""

from __future__ import annotations


def crop_and_deskew(image):
    """Crop to the ECG trace area and correct rotation.

    TODO (week 1-2): implement after looking at the real images (OpenCV: find the
    grid/trace bounding box, estimate skew from grid lines).
    """
    raise NotImplementedError


def build_transforms(config: dict, train: bool):
    """torchvision transforms for training (with safe augmentation) or evaluation."""
    from torchvision import transforms as T

    size = config["preprocess"]["image_size"]
    aug = config["preprocess"]["augment"]
    steps = [T.Resize((size, size))]
    if train:
        steps += [
            T.RandomRotation(aug["rotation_degrees"]),
            T.ColorJitter(brightness=aug["brightness"], contrast=aug["contrast"]),
            T.RandomApply([T.GaussianBlur(3)], p=aug["blur_prob"]),
        ]
    steps += [
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet stats
    ]
    return T.Compose(steps)
