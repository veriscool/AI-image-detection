"""Classify a single image from the command line.

Example::

    python -m ai_image_detector.predict path/to/image.jpg
    python -m ai_image_detector.predict image.jpg --weights models/ai_image_detector.pth
"""

from __future__ import annotations

import argparse

import torch
from PIL import Image

from .model import CLASS_NAMES, build_transform, load_model


def predict(image_path: str, weights: str = "models/ai_image_detector.pth") -> dict[str, float]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(weights, device)
    image = Image.open(image_path).convert("RGB")
    tensor = build_transform()(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).cpu().flatten().tolist()
    return {name: round(p, 5) for name, p in zip(CLASS_NAMES, probs)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("image")
    p.add_argument("--weights", default="models/ai_image_detector.pth")
    args = p.parse_args()

    scores = predict(args.image, args.weights)
    label = max(scores, key=scores.get)
    print(f"Prediction: {label}")
    for name, score in scores.items():
        print(f"  {name:<12} {score:.3f}")


if __name__ == "__main__":
    main()
