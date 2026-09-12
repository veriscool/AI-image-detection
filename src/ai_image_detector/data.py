"""Dataset helpers.

The dataset is expected to be an ImageFolder-style tree::

    <data_dir>/
        train/
            Fake/   *.jpg
            Real/   *.jpg
        test/
            Fake/   ...
            Real/   ...
        validation/   (optional)
            Fake/   ...
            Real/   ...
"""

from __future__ import annotations

import os

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from .model import build_transform


def make_loader(
    split_dir: str,
    batch_size: int = 64,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """Build a DataLoader for one split directory (train/test/validation)."""
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    dataset = ImageFolder(split_dir, transform=build_transform())
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
