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
|-- FreshHarvest_Transfer_Learning.ipynb
|-- train_model.py
|-- train_transfer.py
|-- requirements.txt
|-- README.md
|-- data/                 # ignored
`-- artifacts/            # metrics and plots; model weights ignored
```

## Run

1. Extract `FreshHarvest_Dataset.zip` so the class folders are under `data/FRUIT-16K/`.
2. Open `FreshHarvest_CNN.ipynb` and run all cells.
3. Alternatively run `python train_model.py --data-dir data/FRUIT-16K --epochs 8`.

The workflow creates deterministic train, validation, and test splits of 70%, 15%, and 15%. It first trains a compact baseline CNN from scratch without transfer learning, data augmentation, batch normalization, dropout, weight decay, or label smoothing. The optimization stage then introduces regularization and hyperparameter tuning, followed by a frozen-backbone ResNet50 transfer-learning experiment. Model selection uses validation accuracy, and the untouched test set is evaluated once per selected model.

## Verified transfer-learning result

The frozen ResNet50 backbone reduced the trainable parameter count to 32,784. Its classification head exceeded the 95% validation target after the first epoch and reached 99.33% validation accuracy after three epochs. The selected checkpoint achieved 98.83% exact 16-class test accuracy and 99.25% fresh-versus-spoiled test accuracy.

Two head configurations were compared:

| Learning rate | Dropout | Epoch 1 validation | Best validation | Test accuracy |
|---:|---:|---:|---:|---:|
| 0.0010 | 0.20 | 97.88% | 99.33% | 98.83% |
| 0.0005 | 0.35 | 96.13% | 98.38% | 98.04% |

The first configuration is the deployment candidate because it converged faster and produced the stronger validation and test results.

For the full chronological record of model choices, unsuccessful experiments, measured metrics, and deployment considerations, see [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

