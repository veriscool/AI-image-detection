"""Train the AI image detector.

Example::

    python -m ai_image_detector.train \
        --data-dir ./data \
        --epochs 6 --batch-size 100 --lr 0.001 \
        --out models/ai_image_detector.pth

The data directory must contain ``train/`` and ``test/`` subfolders in
ImageFolder layout (see data.py).
"""

from __future__ import annotations

import argparse
import csv
import os

import torch
import torch.nn as nn
from tqdm import tqdm

from .data import make_loader
from .model import AIImageDetector


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", required=True, help="Folder containing train/ and test/")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out", default="models/ai_image_detector.pth")
    p.add_argument("--resume", help="Optional .pth checkpoint to start from")
    p.add_argument("--history", default="reports/training_history.csv")
    return p.parse_args()


def evaluate(model: nn.Module, loader, device: str) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            _, predicted = torch.max(model(images), 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100.0 * correct / max(total, 1)


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_loader = make_loader(
        os.path.join(args.data_dir, "train"), args.batch_size, shuffle=True
    )
    test_loader = make_loader(
        os.path.join(args.data_dir, "test"), args.batch_size, shuffle=False
    )

    model = AIImageDetector().to(device)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"Resumed from {args.resume}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.history) or ".", exist_ok=True)
    history: list[tuple[int, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for images, labels in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running += loss.item()
        avg = running / len(train_loader)
        history.append((epoch, avg))
        print(f"Epoch {epoch}: train loss {avg:.4f}")

    acc = evaluate(model, test_loader, device)
    print(f"Test accuracy: {acc:.2f}%")

    torch.save(model.state_dict(), args.out)
    print(f"Saved weights to {args.out}")

    with open(args.history, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["epoch", "train_loss"])
        writer.writerows(history)
    print(f"Wrote {args.history}")


if __name__ == "__main__":
    main()
