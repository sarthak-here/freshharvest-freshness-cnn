# FreshHarvest Model Development Log

This document records the complete development path, measured results, and reasons for each modeling decision. It is an experiment journal, not a reconstruction of unrecorded results: every reported metric comes from the saved run artifacts or terminal output.

## 1. Problem definition

FreshHarvest Logistics needs a consistent image-based inspection process for eight produce types: banana, lemon, lulo, mango, orange, strawberry, tamarillo, and tomato. Each item must be classified as fresh or spoiled.

The supplied dataset uses 16 combined classes:

- `F_*`: eight fresh-produce classes;
- `S_*`: eight spoiled-produce classes.

The model therefore solves a 16-class image-classification problem. The predicted class is also converted into two business-facing outputs: produce type and freshness status.

## 2. Dataset audit

- Total images: 16,000
- Image size: 224 x 224 pixels
- Classes: 16
- Images per class: 1,000
- Class balance: perfectly balanced
- Training split: 11,200 images, 700 per class
- Validation split: 2,400 images, 150 per class
- Test split: 2,400 images, 150 per class
- Random seed: 42

The split is deterministic and stratified. The validation set is used for model and epoch selection. The test set is reserved for final evaluation.

The original archive and extracted images are intentionally excluded from Git. This prevents supplied assignment data from being redistributed and keeps the repository manageable.

## 3. Initial compact CNN

The first implementation used a four-stage convolutional network:

1. Two 3 x 3 convolutions with 32 channels, followed by max pooling.
2. Two 3 x 3 convolutions with 64 channels, followed by max pooling.
3. Two 3 x 3 convolutions with 128 channels, followed by max pooling.
4. Two 3 x 3 convolutions with 256 channels, followed by max pooling.
5. Adaptive average pooling and a 16-output linear classifier.

The first completed run included batch normalization, dropout, augmentation, AdamW weight decay, and label smoothing. It produced:

- Best validation accuracy: 98.83%
- Test 16-class accuracy: 98.33%
- Test freshness accuracy: 99.04%

This proved that the dataset could support the required accuracy. However, the next instruction explicitly requested an initial model without transfer learning or regularization, so the workflow was revised instead of presenting this run as the baseline.

## 4. Plain baseline CNN

The baseline was changed to remove:

- transfer learning;
- data augmentation;
- batch normalization;
- dropout;
- weight decay;
- label smoothing.

It used ordinary cross-entropy loss and the Adam optimizer. Validation accuracy was measured across eight epochs to study training duration.

| Epoch | Training accuracy | Validation accuracy |
|---:|---:|---:|
| 1 | 66.16% | 76.96% |
| 2 | 88.35% | 93.08% |
| 3 | 94.02% | 85.25% |
| 4 | 96.62% | 94.58% |
| 5 | 96.58% | 97.71% |
| 6 | 97.51% | 90.96% |
| 7 | 97.74% | 97.00% |
| 8 | 98.21% | 97.92% |

Final baseline results:

- Best validation accuracy: 97.92%
- Test 16-class accuracy: 97.96%

The fluctuations in validation accuracy showed that the plain CNN was capable of high accuracy but was not consistently efficient. It did not meet the later expectation of exceeding 95% in fewer than three epochs.

## 5. Regularization and optimization experiments

The optimization phase evaluated batch normalization, dropout, AdamW weight decay, validation checkpointing, and early-stopping logic.

### Oversized dense-head experiment

The first optimized design retained a 4 x 4 spatial feature map and added a 512-unit dense layer. With learning rate 0.001 and dropout 0.25, validation accuracy was:

| Epoch | Validation accuracy |
|---:|---:|
| 1 | 70.33% |
| 2 | 85.62% |
| 3 | 88.38% |

This design added parameters without converging quickly enough, so the run was stopped rather than spending more compute on a configuration that failed the three-epoch requirement.

### Lightweight regularized-head experiment

The dense head was replaced with global average pooling and a smaller classifier. A higher learning rate was tested, but the first epoch reached only 64.17% validation accuracy. The run was stopped early because it was not on track to beat the required benchmark efficiently.

### Decision

Additional training could have improved these custom CNNs, but that would conflict with the new business requirement to reduce epoch count and compute cost. The project therefore moved to transfer learning.

## 6. ResNet50 transfer learning

ResNet50 pretrained on ImageNet was selected because its visual backbone already contains reusable edge, texture, shape, and object features.

Implementation decisions:

- Load pretrained ResNet50 weights.
- Freeze the convolutional backbone.
- Replace the final layer with dropout and a 16-class linear head.
- Train only 32,784 parameters out of 23,540,816 total parameters.
- Use AdamW with weight decay of 0.0001.
- Save the checkpoint with the best validation accuracy.
- Include early-stopping logic with patience of two epochs.
- Limit training to three epochs.

Two classification-head configurations were compared.

### Configuration A - selected model

- Learning rate: 0.001
- Dropout: 0.20

| Epoch | Training accuracy | Validation accuracy |
|---:|---:|---:|
| 1 | 86.68% | 97.88% |
| 2 | 98.25% | 99.12% |
| 3 | 98.98% | 99.33% |

Final results:

- Exceeded 95% validation accuracy in epoch 1.
- Best validation accuracy: 99.33%
- Test 16-class accuracy: 98.83%
- Test freshness accuracy: 99.25%

### Configuration B

- Learning rate: 0.0005
- Dropout: 0.35

| Epoch | Training accuracy | Validation accuracy |
|---:|---:|---:|
| 1 | 78.99% | 96.13% |
| 2 | 96.26% | 97.92% |
| 3 | 97.54% | 98.38% |

Final results:

- Exceeded 95% validation accuracy in epoch 1.
- Best validation accuracy: 98.38%
- Test 16-class accuracy: 98.04%
- Test freshness accuracy: 98.58%

## 7. Final model selection

Configuration A was selected because it:

- crossed the 95% requirement in one epoch;
- had the highest validation accuracy;
- had the highest exact-class test accuracy;
- had the highest freshness test accuracy;
- required training only a small classification head.

The training and validation curves remain close and improve together across all three epochs, so they do not show meaningful overfitting. Early stopping was available but did not activate because validation accuracy improved every epoch.

The selected model is saved at `artifacts/transfer/best_resnet50_model.pt`.

## 8. Deployment considerations

The measured test performance is strong, but a production conveyor deployment still needs:

- validation using real warehouse cameras and lighting;
- latency and throughput measurements on deployment hardware;
- a confidence threshold for automatic decisions;
- a manual-review queue for uncertain predictions;
- monitoring for seasonal appearance and camera drift;
- special attention to spoiled-class recall because missed spoilage is more costly than an extra manual inspection;
- retraining with new warehouse images when performance changes.

## 9. Repository history

The Git history intentionally preserves the project progression:

1. `8af28d9` - initial CNN project, notebook, evaluation metrics, and plots.
2. `474e691` - trained compact-CNN weights added after confirming the dataset remained ignored.
3. `2b6c094` - corrected baseline and optimized ResNet50 transfer-learning workflow, comparison metrics, notebook, and best checkpoint.
4. The next commit adds this complete development and experiment journal before the repository is published.

