"""Evaluate a trained checkpoint on the test split.

Example::

    python -m ai_image_detector.evaluate \
        --data-dir ./data --weights models/ai_image_detector.pth

Prints overall and per-class accuracy and a 2x2 confusion matrix. With
``--plot`` it also writes reports/confusion_matrix.png.
"""

from __future__ import annotations

import argparse
import os

import torch

from .data import make_loader
from .model import CLASS_NAMES, load_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--weights", default="models/ai_image_detector.pth")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--plot", action="store_true", help="Save confusion_matrix.png")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.weights, device)
    loader = make_loader(os.path.join(args.data_dir, "test"), args.batch_size)

    # confusion[true][pred]
    confusion = [[0, 0], [0, 0]]
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            _, predicted = torch.max(model(images), 1)
            for t, p in zip(labels.tolist(), predicted.cpu().tolist()):
                confusion[t][p] += 1

    total = sum(sum(row) for row in confusion)
    correct = confusion[0][0] + confusion[1][1]
    print(f"Overall accuracy: {100.0 * correct / total:.2f}%  (n={total})")
    for i, name in enumerate(CLASS_NAMES):
        support = sum(confusion[i])
        acc = 100.0 * confusion[i][i] / support if support else 0.0
        print(f"  {name:<12} accuracy: {acc:.2f}%  (n={support})")

    print("\nConfusion matrix (rows = true, cols = predicted):")
    print(f"{'':<14}" + "".join(f"{n:<14}" for n in CLASS_NAMES))
    for i, name in enumerate(CLASS_NAMES):
        print(f"{name:<14}" + "".join(f"{v:<14}" for v in confusion[i]))

    if args.plot:
        _save_plot(confusion)


def _save_plot(confusion: list[list[int]]) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(confusion, cmap="Blues")
    ax.set_xticks([0, 1], labels=CLASS_NAMES)
    ax.set_yticks([0, 1], labels=CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(confusion[i][j]), ha="center", va="center")
    fig.tight_layout()
    os.makedirs("reports", exist_ok=True)
    fig.savefig("reports/confusion_matrix.png", dpi=120)
    print("Saved reports/confusion_matrix.png")


if __name__ == "__main__":
    main()
