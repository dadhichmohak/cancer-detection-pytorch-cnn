# Histopathologic Cancer Detection using CNNs

A PyTorch convolutional neural network that detects metastatic cancer in small histopathology image patches from lymph node scans. Built for the [Kaggle Histopathologic Cancer Detection](https://www.kaggle.com/c/histopathologic-cancer-detection) challenge and evaluated on **AUC-ROC**.

**Notebook (Google Colab):** [Open the Colab notebook](https://drive.google.com/file/d/1tVeUvpFzw_Gt4FOiKnAQDI4mtXpqfIM7/view?usp=sharing)

---

## Overview

Pathologists examining lymph node tissue for breast cancer metastases must scan very large areas, and small metastases are easy to miss. This project trains a deep CNN to classify image patches automatically, as a step toward faster, more consistent screening support.

**Task:** Binary classification. A patch is labelled positive if the central 32×32 px region contains at least one pixel of tumour tissue.

## Dataset

| Item | Details |
|---|---|
| Source | Kaggle subset of the [PCam](https://github.com/basveeling/pcam) benchmark (derived from Camelyon16) |
| Train images | 220,025 (≈60:40 negative:positive) |
| Test images | 57,458 |
| Patch size | 96 × 96 px, RGB (`.tif`) |
| Metric | Area Under the ROC Curve (AUC) |

Duplicates present in the original PCam data are removed in the Kaggle version.

## Approach

1. **Sampling:** Drew a class-balanced training set of **160,000 patches** (80,000 per class) with a fixed seed for reproducibility.
2. **Data pipeline:** Custom PyTorch `Dataset` reading `.tif` patches with OpenCV; 90/10 train/validation split via `SubsetRandomSampler`.
3. **Augmentation (train only):** random horizontal/vertical flips (p = 0.4), random rotation (±20°), normalisation to [-1, 1].
4. **Model:** 5-block CNN (Conv → BatchNorm → ReLU → MaxPool) with channels 32 → 64 → 128 → 256 → 512, followed by a fully connected head (1024 → 512 → 1) with dropout (0.4) and a sigmoid output. **~6.8M trainable parameters.**
5. **Training:** Binary cross-entropy loss, Adam optimiser (lr = 1.5e-4), batch size 128, 20 epochs on GPU. The checkpoint with the lowest validation loss is saved as `best_model.pt` and used for inference.
6. **Inference:** Predicted probabilities for the test set are written to `submission.csv`.

### Architecture

```
Input (3×96×96)
 └─ Conv(3→32, k3)    → BN → ReLU → MaxPool
 └─ Conv(32→64, k2)   → BN → ReLU → MaxPool
 └─ Conv(64→128, k3)  → BN → ReLU → MaxPool
 └─ Conv(128→256, k3) → BN → ReLU → MaxPool
 └─ Conv(256→512, k3) → BN → ReLU → MaxPool      → 512×3×3
 └─ Flatten → FC 1024 → ReLU → Dropout(0.4)
            → FC 512  → Dropout(0.4)
            → FC 1    → Sigmoid
```

## Results

| Metric | Value |
|---|---|
| Final training loss (epoch 20) | 0.128 |
| Final validation loss (epoch 20) | 0.124 |
| Validation AUC (logged, epoch 20) | 0.980 |
| Test AUC (Kaggle submission) | ≈ 0.95 |

Training and validation loss decrease steadily across all 20 epochs, with no sign of severe overfitting. Dropout, batch norm and augmentation help here.

> **Note on the logged validation AUC:** the notebook accumulates per-batch AUC values across epochs without resetting them, so the logged figure is a running average and understates later-epoch performance. For a cleaner number, compute AUC on the full validation set once per epoch (see *Possible improvements*).

## Getting Started

### Requirements

```bash
pip install torch torchvision numpy pandas scikit-learn opencv-python pillow matplotlib
```

A CUDA-capable GPU is strongly recommended.

### Data

Download the competition data from Kaggle and arrange it as:

```
input/
├── train/                # 220,025 .tif patches
├── test/                 # 57,458 .tif patches
├── train_labels.csv
└── sample_submission.csv
```

### Run

1. Open the [Colab notebook](https://drive.google.com/file/d/1tVeUvpFzw_Gt4FOiKnAQDI4mtXpqfIM7/view?usp=sharing) (or the `.ipynb` file in Jupyter/Kaggle) and switch to a GPU runtime.
2. Set `base_dir` to your data folder.
3. Run all cells. Training saves `best_model.pt`, and the final cells write `submission.csv`.

## Possible Improvements

- Compute validation AUC over the full validation set each epoch instead of per-batch running averages.
- Use dataset-specific channel mean/std for normalisation (values are already computed in the notebook, commented out).
- Test-time augmentation (flips/rotations) and model ensembling.
- Transfer learning with pretrained backbones (ResNet, DenseNet, EfficientNet).
- Learning-rate scheduling and a proper hold-out validation split with a fixed shuffle.
- Weight the loss or crop towards the central 32×32 region that determines the label.
- Add Grad-CAM visualisations for interpretability.

## Limitations

This is a research/learning project on a benchmark dataset and **not a clinical diagnostic tool**. Patches lack wider tissue context, and performance on data from other scanners or hospitals is untested.

## Acknowledgements

- Kaggle Histopathologic Cancer Detection competition and the [PCam](https://github.com/basveeling/pcam) dataset (Veeling et al.), derived from [Camelyon16](https://camelyon16.grand-challenge.org/Data/).
- Based on the Kaggle notebook by [Abhinand](http://kaggle.com/abhinand05). <!-- Edit this line to reflect your own contributions -->

## License

MIT (update as appropriate).
