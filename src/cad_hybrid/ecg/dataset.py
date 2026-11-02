
from pathlib import Path

import pandas as pd
import torch

from torch.utils.data import Dataset

from .preprocess import preprocess_image


ROOT = Path(__file__).resolve().parents[3]

CLASS_NAMES = [
    "cad_positive",
    "negative",
    "arrhythmia",
]

LABEL_MAP = {
    name: index
    for index, name in enumerate(CLASS_NAMES)
}


class ECGDataset(Dataset):

    def __init__(self, csv_file):

        csv_file = Path(csv_file)

        if not csv_file.is_absolute():
            csv_file = ROOT / csv_file

        if not csv_file.exists():
            raise FileNotFoundError(
                f"Split CSV not found: {csv_file}"
            )

        self.data = pd.read_csv(csv_file)

        required_columns = {
            "path",
            "label",
            "dup_group"
        }

        missing = required_columns - set(self.data.columns)

        if missing:
            raise ValueError(
                f"Missing CSV columns: {sorted(missing)}"
            )

        if self.data.empty:
            raise ValueError("ECG dataset is empty.")

        unknown = set(self.data["label"]) - set(LABEL_MAP)

        if unknown:
            raise ValueError(
                f"Unknown ECG labels: {unknown}"
            )

        if self.data["dup_group"].isnull().any():
            raise ValueError("Missing duplicate-group IDs.")

        self.data["dup_group"] = (
            self.data["dup_group"].astype(str)
        )

        missing_images = []

        for relative_path in self.data["path"]:

            image_path = Path(relative_path)

            if not image_path.is_absolute():
                image_path = ROOT / image_path

            if not image_path.exists():
                missing_images.append(str(image_path))

        if missing_images:
            raise FileNotFoundError(
                f"{len(missing_images)} ECG images are missing. "
                f"First missing image: {missing_images[0]}"
            )

        print(
            f"Loaded {len(self.data)} ECG images from "
            f"{csv_file.name}"
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        image_path = Path(row["path"])

        if not image_path.is_absolute():
            image_path = ROOT / image_path

        # Use exactly the same preprocessing as prediction.
        image_tensor = preprocess_image(
            str(image_path)
        ).squeeze(0)

        label = LABEL_MAP[row["label"]]

        group = str(row["dup_group"])

        return (
            image_tensor,
            torch.tensor(label, dtype=torch.long),
            group
        )
