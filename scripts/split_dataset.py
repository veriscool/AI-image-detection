"""Randomly move a fixed number of files from one folder to another.

Used to carve a test/validation split out of a training folder.

Example::

    python scripts/split_dataset.py --src ./data/train/Fake --dst ./data/test/Fake --count 7678
"""

from __future__ import annotations

import argparse
import os
import random
import shutil


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--count", type=int, required=True)
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
    if args.count > len(files):
        raise SystemExit(f"Asked for {args.count} files but only {len(files)} available")

    for name in random.sample(files, args.count):
        shutil.move(os.path.join(args.src, name), os.path.join(args.dst, name))

    print(f"Moved {args.count} files from {args.src} to {args.dst}")


if __name__ == "__main__":
    main()
