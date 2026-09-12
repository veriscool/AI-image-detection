"""Flask demo for the AI image detector.

Run from the project root::

    pip install -e ".[web]"
    python webapp/app.py

Then open http://127.0.0.1:5000/. Configuration via environment variables:

    AI_DETECTOR_WEIGHTS   path to the .pth file (default: models/ai_image_detector.pth)
    FLASK_DEBUG           set to "1" to enable the reloader/debugger
    PORT                  port to serve on (default: 5000)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from flask import Flask, render_template, request, send_from_directory, url_for
from PIL import Image
from werkzeug.utils import secure_filename

# Make ``src/`` importable when running this file directly without installing.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ai_image_detector.model import CLASS_NAMES, build_transform, load_model  # noqa: E402

WEIGHTS = os.environ.get(
    "AI_DETECTOR_WEIGHTS", str(PROJECT_ROOT / "models" / "ai_image_detector.pth")
)
CONFIDENT_THRESHOLD = 0.75

# One-click preset images for the homepage sample gallery. Optional: if this folder
# isn't present (e.g. a fresh clone before the images are pushed), the gallery is
# simply omitted from the page.
SAMPLE_IMAGES_DIR = PROJECT_ROOT / "test images"
SAMPLE_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
# Maps the "test images/<category>" folder name to the human-readable intended label
# shown under its thumbnails. Falls back to the folder name (title-cased) for any
# category that isn't one of these two.
SAMPLE_CATEGORY_LABELS = {"fake": CLASS_NAMES[0], "real": CLASS_NAMES[1]}

WEBAPP_DIR = Path(__file__).resolve().parent
app = Flask(
    __name__,
    template_folder=str(WEBAPP_DIR / "templates"),
    static_folder=str(WEBAPP_DIR / "static"),
)
app.config["UPLOAD_FOLDER"] = os.path.join(app.root_path, "static", "uploads")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

model = load_model(WEIGHTS, "cpu")
transform = build_transform()


def describe(ai_score: float, human_score: float) -> str:
    if ai_score >= human_score:
        strength = "is" if ai_score > CONFIDENT_THRESHOLD else "may be"
        return f"This image {strength} AI generated."
    strength = "is" if human_score > CONFIDENT_THRESHOLD else "may be"
    return f"This image {strength} human made."


def predict(image: Image.Image) -> tuple[float, float]:
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).cpu().numpy().flatten()
    return round(float(probs[0]), 5), round(float(probs[1]), 5)


def list_sample_images() -> list[dict]:
    """Preset images from ``test images/<category>/*`` for the one-click gallery.

    Each thumbnail is labeled with its intended (ground-truth) class, taken from the
    category folder it lives in (``fake`` -> "AI generated", ``real`` -> "Human made").
    """
    samples: list[dict] = []
    if not SAMPLE_IMAGES_DIR.is_dir():
        return samples

    for category_dir in sorted(SAMPLE_IMAGES_DIR.iterdir()):
        if not category_dir.is_dir():
            continue
        category_label = SAMPLE_CATEGORY_LABELS.get(
            category_dir.name.lower(), category_dir.name.title()
        )
        image_paths = sorted(
            p for p in category_dir.iterdir() if p.suffix.lower() in SAMPLE_IMAGE_EXTENSIONS
        )
        for i, path in enumerate(image_paths, start=1):
            samples.append(
                {
                    "rel_path": f"{category_dir.name}/{path.name}",
                    "label": f"{category_label} {i}",
                }
            )
    return samples


@app.route("/")
def index():
    return render_template("index.html", samples=list_sample_images())


@app.route("/sample-images/<path:relpath>")
def sample_image(relpath):
    return send_from_directory(SAMPLE_IMAGES_DIR, relpath)


@app.route("/result-sample")
def result_sample():
    relpath = request.args.get("path", "")
    valid_paths = {s["rel_path"] for s in list_sample_images()}
    if relpath not in valid_paths:
        return "Unknown sample image", 404

    image = Image.open(SAMPLE_IMAGES_DIR / relpath).convert("RGB")
    ai_score, human_score = predict(image)

    return render_template(
        "result.html",
        ai_score=ai_score,
        human_score=human_score,
        prediction=describe(ai_score, human_score),
        class_names=CLASS_NAMES,
        image_url=url_for("sample_image", relpath=relpath),
    )


@app.route("/result", methods=["POST"])
def result():
    file = request.files.get("image")
    if file is None or file.filename == "":
        return "No image uploaded", 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    image = Image.open(save_path).convert("RGB")
    ai_score, human_score = predict(image)

    return render_template(
        "result.html",
        ai_score=ai_score,
        human_score=human_score,
        prediction=describe(ai_score, human_score),
        class_names=CLASS_NAMES,
        image_url=url_for("static", filename=f"uploads/{filename}"),
    )


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=int(os.environ.get("PORT", 5000)),
        debug=os.environ.get("FLASK_DEBUG") == "1",
    )
