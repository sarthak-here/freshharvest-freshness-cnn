# FreshHarvest Freshness Inspection CNN

An image-classification project for automated produce inspection. The model identifies both the produce type and whether the item is fresh or spoiled across 16 balanced classes.

## Business goal

FreshHarvest needs a repeatable inspection process that reduces inconsistent manual decisions on warehouse conveyor lines. This prototype turns a 224 x 224 produce image into:

- produce type: banana, lemon, lulo, mango, orange, strawberry, tamarillo, or tomato;
- freshness status: fresh or spoiled;
- prediction confidence.

The result is intended as a decision-support prototype. A production conveyor deployment would still require camera-specific validation, latency testing, monitoring, and a human-review path for uncertain predictions.

## Dataset

The supplied archive contains 16,000 images, with exactly 1,000 images in each of 16 classes. Folder names use `F_` for fresh and `S_` for spoiled.

The dataset is intentionally excluded from Git because it is provided assignment material. Place or extract it under `data/FRUIT-16K/` before running the notebook.

## Project structure

```text
freshharvest-cnn/
|-- FreshHarvest_CNN.ipynb
|-- train_model.py
|-- requirements.txt
|-- README.md
|-- data/                 # ignored
`-- artifacts/            # metrics and plots; model weights ignored
```

## Run

1. Extract `FreshHarvest_Dataset.zip` so the class folders are under `data/FRUIT-16K/`.
2. Open `FreshHarvest_CNN.ipynb` and run all cells.
3. Alternatively run `python train_model.py --data-dir data/FRUIT-16K --epochs 8`.

The workflow creates deterministic train, validation, and test splits of 70%, 15%, and 15%, trains a compact CNN, selects the checkpoint with the best validation accuracy, and reports final test performance.

