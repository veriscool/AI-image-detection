"""Print the flattened feature size after the two conv/pool blocks.

Useful when changing the input resolution: the number it prints is what the
first nn.Linear layer in the model must expect.

Example::

    python scripts/compute_flatten_size.py --size 250
    python scripts/compute_flatten_size.py --size 500
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=250, help="Square input resolution")
    args = parser.parse_args()

    conv1 = nn.Conv2d(3, 6, 5)
    pool = nn.MaxPool2d(2, 2)
    conv2 = nn.Conv2d(6, 16, 5)

    dummy = torch.zeros(1, 3, args.size, args.size)
    x = pool(F.relu(conv1(dummy)))
    x = pool(F.relu(conv2(x)))

    print(f"Shape after conv+pool: {tuple(x.shape)}")
    print(f"Flattened size: {x.view(1, -1).shape[1]}")


if __name__ == "__main__":
    main()
