"""PyTorch Dataset / DataLoader / augmentation pipeline for the split dataset."""

from __future__ import annotations

from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

from config import BATCH_SIZE, CLASS_NAMES, IMG_SIZE, SPLIT_DATA_DIR

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_transforms(train: bool) -> A.Compose:
    if train:
        return A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=25, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
            A.OneOf([
                A.GaussNoise(p=1.0),
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            ], p=0.3),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


class FishDiseaseDataset(Dataset):
    """Reads dataset/split/<train|val|test>/<class_name>/*.jpg"""

    def __init__(self, split: str, transform: A.Compose | None = None):
        self.split_dir = Path(SPLIT_DATA_DIR) / split
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        for class_idx, class_name in enumerate(CLASS_NAMES):
            class_dir = self.split_dir / class_name
            if not class_dir.exists():
                continue
            for img_path in class_dir.glob("*"):
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    self.samples.append((img_path, class_idx))

        if not self.samples:
            raise RuntimeError(
                f"No images found under {self.split_dir}. "
                "Run src/dataset_prep.py first."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)["image"]
        else:
            image = torch.from_numpy(np.transpose(image, (2, 0, 1))).float()

        return image, label


def get_dataloaders(batch_size: int = BATCH_SIZE, num_workers: int = 2):
    train_ds = FishDiseaseDataset("train", get_transforms(train=True))
    val_ds = FishDiseaseDataset("val", get_transforms(train=False))
    test_ds = FishDiseaseDataset("test", get_transforms(train=False))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
