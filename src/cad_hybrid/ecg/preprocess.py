
from pathlib import Path

import cv2
import numpy as np
import torch

from PIL import Image, ImageOps
from torchvision import transforms


ROOT = Path(__file__).resolve().parents[3]
IMAGE_SIZE = 224


def crop_and_deskew(image):
    """Correct slight rotation in an ECG image."""

    if image is None:
        raise ValueError("Input image is empty.")

    processed = image.copy()

    gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=300,
        maxLineGap=10
    )

    angles = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = np.asarray(line).reshape(-1)[:4]

            angle = np.degrees(
                np.arctan2(y2 - y1, x2 - x1)
            )

            if abs(angle) <= 3:
                angles.append(angle)

    if angles:
        rotation_angle = float(np.median(angles))

        if 0.2 <= abs(rotation_angle) <= 3:
            height, width = processed.shape[:2]

            center = (width // 2, height // 2)

            matrix = cv2.getRotationMatrix2D(
                center,
                rotation_angle,
                1.0
            )

            processed = cv2.warpAffine(
                processed,
                matrix,
                (width, height),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )

    return processed


def resize_ecg(image, target_size=(512, 512)):
    """Resize ECG image while preserving aspect ratio."""

    if image is None:
        raise ValueError("Input image is empty.")

    height, width = image.shape[:2]

    scale = min(
        target_size[0] / width,
        target_size[1] / height
    )

    new_width = int(width * scale)
    new_height = int(height * scale)

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )

    return resized


image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def preprocess_image(image_path):
    """
    Reproduce the original ECG preprocessing pipeline.
    Returns a tensor with shape (1, 3, 224, 224).
    """

    image_path = Path(image_path)

    if not image_path.is_absolute():
        image_path = ROOT / image_path

    if not image_path.exists():
        raise FileNotFoundError(
            f"ECG image not found: {image_path}"
        )

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError("Unable to read ECG image.")

    image = crop_and_deskew(image)

    image = resize_ecg(
        image,
        target_size=(512, 512)
    )

    success, encoded_image = cv2.imencode(".jpg", image)

    if not success:
        raise ValueError("Unable to encode processed ECG image.")

    image = cv2.imdecode(
        encoded_image,
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise ValueError("Unable to decode processed ECG image.")

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = Image.fromarray(image)

    image = ImageOps.contain(
        image,
        (IMAGE_SIZE, IMAGE_SIZE),
        method=Image.Resampling.LANCZOS
    )

    canvas = Image.new(
        "RGB",
        (IMAGE_SIZE, IMAGE_SIZE),
        "white"
    )

    x = (IMAGE_SIZE - image.width) // 2
    y = (IMAGE_SIZE - image.height) // 2

    canvas.paste(image, (x, y))

    tensor = image_transform(canvas)

    return tensor.unsqueeze(0)


# Keep the existing training transform utility.
def build_transforms(config: dict, train: bool):
    from torchvision import transforms as T

    size = config["preprocess"]["image_size"]
    aug = config["preprocess"]["augment"]

    steps = [T.Resize((size, size))]

    if train:
        steps += [
            T.RandomRotation(aug["rotation_degrees"]),
            T.ColorJitter(
                brightness=aug["brightness"],
                contrast=aug["contrast"]
            ),
            T.RandomApply(
                [T.GaussianBlur(3)],
                p=aug["blur_prob"]
            ),
        ]

    steps += [
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]

    return T.Compose(steps)
