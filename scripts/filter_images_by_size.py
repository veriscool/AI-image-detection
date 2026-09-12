"""Move images that meet a minimum resolution into a destination folder.

Used while building the dataset: it samples files from a source folder, keeps
only those at least ``--min-size`` pixels on both axes, and moves up to
``--limit`` of them into the destination.

Example::

    python scripts/filter_images_by_size.py \
        --src ./raw/fake --dst ./data/train/Fake --limit 57549 --min-size 500
"""

from __future__ import annotations

import argparse
import os
import random
import shutil

from PIL import Image
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", required=True, help="Source folder of images")
    p.add_argument("--dst", required=True, help="Destination folder")
    p.add_argument("--limit", type=int, required=True, help="How many valid images to move")
    p.add_argument("--min-size", type=int, default=500, help="Minimum width and height")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    os.makedirs(args.dst, exist_ok=True)

    files = [
        f for f in os.listdir(args.src)
        if os.path.isfile(os.path.join(args.src, f))
    ]
    random.shuffle(files)

    moved = 0
    for name in tqdm(files, desc="scanning"):
        if moved >= args.limit:
            break
        src_path = os.path.join(args.src, name)
        try:
            with Image.open(src_path) as img:
                width, height = img.size
        except Exception as exc:  # noqa: BLE001 - skip anything Pillow cannot read
            print(f"skip {name}: {exc}")
            continue
        if width >= args.min_size and height >= args.min_size:
            shutil.move(src_path, os.path.join(args.dst, name))
            moved += 1

    print(f"Moved {moved} images to {args.dst}")


if __name__ == "__main__":
    main()
