# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A CNN (trained from scratch, no pretrained backbone) that classifies an image as AI-generated
or human-made, served through a small Flask app. Portfolio piece — the code favors clarity
over completeness (no test suite, no lint config).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (PowerShell is the primary shell here)
pip install -e ".[web,train]"   # or: pip install -r requirements.txt
```

## Common commands

```bash
# Run the web demo (imports src/ai_image_detector without installing, via sys.path hack)
python webapp/app.py
# env vars: AI_DETECTOR_WEIGHTS (default models/ai_image_detector.pth), FLASK_DEBUG=1, PORT

# Classify one image from the CLI
python -m ai_image_detector.predict path/to/image.jpg [--weights models/ai_image_detector.pth]
# or, after `pip install -e .`:  ai-detect path/to/image.jpg

# Train (data dir must be ImageFolder layout: <dir>/train|test|validation/{Fake,Real}/*.jpg)
python -m ai_image_detector.train --data-dir ./data --epochs 6 --batch-size 100 --lr 0.001 \
    --out models/ai_image_detector.pth --resume optional/checkpoint.pth

# Evaluate a checkpoint: prints accuracy + confusion matrix, optionally plots it
python -m ai_image_detector.evaluate --data-dir ./data --weights models/ai_image_detector.pth --plot
```

There is no test suite and no configured linter/formatter in this repo.

## Architecture

- **`src/ai_image_detector/model.py` is the single source of truth for the network.** Both
  training and the web app import `AIImageDetector` from here — never redefine the architecture
  elsewhere, or the shipped weights (which are just a state dict) will silently fail to line up
  with the code that loads them.
  - Input: RGB tensor `(N, 3, 250, 250)`, no normalization (`build_transform()` only resizes +
    converts to tensor — matches how the shipped weights were trained). Output: raw logits `(N, 2)`.
  - `conv1(3→6,5x5) → pool → conv2(6→16,5x5) → pool → flatten(16*59*59) → fc1..fc5 → 2 logits`.
  - `CLASS_NAMES = ("AI generated", "Human made")` — index 0/1 matches `ImageFolder`'s alphabetical
    sort of `Fake`/`Real` class folders. Don't reorder without re-checking this.
  - `load_model()` remaps legacy `con1`/`con2` state-dict keys (a typo baked into the original
    notebooks/checkpoint) to `conv1`/`conv2` via `_LEGACY_KEY_MAP`. Any newly trained checkpoint
    uses the correct names already; the remap is a no-op for those and only matters for the
    original shipped `.pth`.
- **`src/ai_image_detector/data.py`** wraps `torchvision.datasets.ImageFolder` + `DataLoader` for
  a single split directory; `train.py`/`evaluate.py` call it once per split (`train`, `test`).
- **`webapp/app.py`** adds `src/` to `sys.path` at runtime instead of requiring `pip install -e .`,
  so it can be run directly with `python webapp/app.py`. It loads the model once at import time
  (module-level global), not per-request.
- The dataset itself is not in the repo (tens of GB); `scripts/filter_images_by_size.py` and
  `scripts/split_dataset.py` are the one-off utilities that built the train/test/validation split
  from raw image folders.
- `notebooks/` holds the exploratory v1 baseline and v2 improved training runs that preceded the
  scripted `train.py`/`evaluate.py` — reference only, not part of the runtime path.
- Repo root still has many loose `.pth` checkpoint files from before the project was restructured
  into this portfolio-ready layout; `.gitignore` excludes all `*.pth` except
  `models/ai_image_detector.pth` (the one shipped/loaded checkpoint) going forward.
