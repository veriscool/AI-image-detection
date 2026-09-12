# AI Image Detector

A convolutional neural network that classifies an image as **AI generated** or
**human made**, trained from scratch in PyTorch and served through a small Flask
web app.

<!-- Add a screenshot or GIF of the running app at docs/screenshots/demo.png -->
<!-- ![Demo](docs/screenshots/demo.png) -->

---

## Highlights

- Custom CNN, built and trained from scratch with no pretrained backbone: two
  convolution/pooling blocks feeding five fully connected layers.
- About 86% test accuracy on a held-out set of real photos vs. AI-generated
  images (81% AI recall, 91% human recall). Full numbers in
  [reports/metrics.md](reports/metrics.md).
- Dataset prep scripts, training loop, evaluation with a confusion matrix, CLI
  inference, and a web demo, all wired together.
- The architecture lives in one place, `src/ai_image_detector/model.py`, and is
  shared by training and serving code, so the weights and the code that loads
  them can't drift apart.

## Results

| Model | Input | Test acc. | AI recall | Human recall | Size |
|---|---|---|---|---|---|
| `ai_image_detector.pth` (shipped) | 250×250 | ~85.8% | ~80.7% | ~90.8% | 26 MB |
| `best_model.pth` (experimental, deeper + dropout) | 500×500 | ~85.0% | ~82.4% | ~87.5% | 228 MB |

The larger model isn't included in the repo: it's about 9x the size for
comparable accuracy. The 78-epoch loss curve is in
[reports/training_history.csv](reports/training_history.csv).

## Project structure

```
.
├── src/ai_image_detector/   # importable package
│   ├── model.py             # AIImageDetector architecture + preprocessing (canonical)
│   ├── data.py              # ImageFolder DataLoader helpers
│   ├── train.py             # training entry point  (python -m ai_image_detector.train)
│   ├── evaluate.py          # test-set metrics + confusion matrix
│   └── predict.py           # single-image CLI     (python -m ai_image_detector.predict)
├── webapp/                  # Flask demo
│   ├── app.py
│   ├── templates/           # index.html, result.html
│   └── static/              # css/, img/, uploads/ (runtime)
├── notebooks/               # exploratory training notebooks (v1 baseline, v2 improved)
├── scripts/                 # one-off dataset preparation utilities
├── models/                  # ai_image_detector.pth (shipped weights)
├── reports/                 # metrics.md, training_history.csv
└── docs/screenshots/        # add app screenshots here
```

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

pip install -e ".[web,train]"
# or:  pip install -r requirements.txt
```

## Run the web app

```bash
python webapp/app.py
```

Open <http://127.0.0.1:5000/>, upload an image, and the model returns the
AI-vs-human probabilities with a doughnut chart. Set `FLASK_DEBUG=1` for the
reloader, or `AI_DETECTOR_WEIGHTS=/path/to/weights.pth` to use a different
checkpoint.

## Classify one image from the CLI

```bash
python -m ai_image_detector.predict path/to/image.jpg
# or, after install:  ai-detect path/to/image.jpg
```

## Train / evaluate

The dataset isn't in this repo; it's tens of GB. Arrange it in
[ImageFolder](https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html)
layout:

```
data/
├── train/{Fake,Real}/*.jpg
├── test/{Fake,Real}/*.jpg
└── validation/{Fake,Real}/*.jpg   # optional
```

```bash
python -m ai_image_detector.train --data-dir ./data --epochs 6 --out models/ai_image_detector.pth
python -m ai_image_detector.evaluate --data-dir ./data --weights models/ai_image_detector.pth --plot
```

`scripts/filter_images_by_size.py` and `scripts/split_dataset.py` built the
train/test/validation split from the raw image folders.

## How it works

1. Images are resized to 250×250 and converted to tensors. No normalization,
   to match how the shipped weights were trained.
2. `conv1 (3→6, 5×5)` → ReLU → maxpool → `conv2 (6→16, 5×5)` → ReLU → maxpool.
3. Flatten to `16·59·59` → FC `120 → 84 → 16 → 8 → 2`.
4. Softmax over the 2 logits gives P(AI) and P(human).

## Limitations

- Trained on a specific mix of AI generators and photo sources; accuracy drops
  on generators or content types the training data didn't cover.
- Not a forensic tool. Treat the output as a heuristic, not proof.

## License

[MIT](LICENSE) — update the copyright holder name to your own.