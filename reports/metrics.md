# Results

Metrics recorded from the training notebooks. Re-generate them at any time with:

```bash
python -m ai_image_detector.evaluate --data-dir ./data --weights models/ai_image_detector.pth --plot
```

## Production model — `models/ai_image_detector.pth`

Architecture: 2 conv/pool blocks + 5 FC layers, 250x250 RGB input (`AIImageDetector`).
Trained with SGD, cross-entropy loss, `lr=0.001`, `batch_size=100`.

| Metric | Value |
|---|---|
| Test accuracy (overall) | ~85.8% |
| Test accuracy — AI generated | ~80.7% |
| Test accuracy — Human made | ~90.8% |

## Experimental larger model — `best_model.pth` (not in repo, ~228 MB)

Architecture: deeper variant with dropout, 500x500 RGB input, mixed-precision
training. Kept for reference; not shipped because it is 9x larger for
comparable accuracy.

| Metric | Value |
|---|---|
| Best validation accuracy | ~87.9% |
| Test accuracy (overall) | ~85.0% |
| Test accuracy — AI generated | ~82.4% |
| Test accuracy — Human made | ~87.5% |

Full 78-epoch train/validation loss curve is in
[`training_history.csv`](training_history.csv).
