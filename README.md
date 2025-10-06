# 🎙️ TIMIT Phoneme Classification (PyTorch)

This project implements a deep neural network (DNN) for **phoneme classification** using MFCC feature data (e.g., from the TIMIT dataset).  
It includes dataset preprocessing, a flexible data loader, and a live learning curve plotter.

---

## Overview

This training pipeline demonstrates:
- **Multi-layer fully connected DNN** with BatchNorm + Dropout
- **Automatic GPU/CPU selection**
- **Train/validation split** built into the dataset loader
- **Live Matplotlib visualization** of loss curves
- **Custom postprocessing** of windowed MFCC features

---

## Folder Structure
```
<repo-root>/
├─ Data/
│ ├─ train_11.npy # training input features
│ ├─ train_label_11.npy # corresponding training labels
│ └─ test_11.npy # test input features
├─ driver.py # main training script
└─ README.md
```
---

## Data Format

Each `.npy` file should contain:

| File | Description | Example Shape | Notes |
|------|--------------|---------------|-------|
| `train_11.npy` | Flattened MFCC windows | `(N, 429)` | 11 frames × 39 dims |
| `train_label_11.npy` | Integer phoneme labels | `(N,)` | `int64` class indices |
| `test_11.npy` | Flattened MFCC test data | `(M, 429)` | no labels |

In the dataset loader, each frame (39-dimensional MFCC) is restructured into context windows of size `(2×stride+1)`.

---

## Configuration

You can adjust the hyperparameters in the `__main__` block of `driver.py`:

```python
Epochs = 50
Batch_size = 1024
Val_ratio = 0.05
Learning_rate = 1e-5
stride = 15
isPostProcessing = True
```

> Important:
> If your .npy files were created using 11-frame windows, keep stride=5.
Changing stride without regenerating data will cause shape mismatches.

---

## Model Architecture
A deep feedforward neural network:
```bash
Input → 1024 → 2048 → 3072 → 4096 → 3072 → 2048 → 1024 → 512 → 128 → 39
```

- Each hidden layer uses:

  - Batch Normalization

  - Dropout (0.25)

  - ReLU activation

- Final layer outputs 39 logits (for 39 phoneme classes)

- Loss function: CrossEntropyLoss

- Optimizer: Adam (optionally switch to AdamW)

---

## Live Plotting
The class LiveCurve updates a real-time training/validation loss plot during training:

```python
live = LiveCurve(title="TIMIT DNN Learning Curve", ylabel="MSE")
live.update(train_loss, dev_loss)
```

> Uses matplotlib in interactive mode (plt.ion()).

If you’re running in a headless environment (e.g., SSH or server), disable plotting or use a compatible backend.

---

## How to Run
```bash
python driver.py
```
You’ll see:

- Device info (Using device: cuda or cpu)

- Dataset summary

- Epoch-by-epoch loss & accuracy

- Live updating loss plot

- Final training time

---

## Dataset Loader
TIMITDataset automatically:

- Loads .npy features & labels

- Optionally applies context-window postprocessing

- Splits into training/validation sets

- Returns PyTorch tensors

Example (from script):

```python
train_set = prep_dataLoader("Data/train_11.npy", "Data/train_label_11.npy", mode="train")
val_set   = prep_dataLoader("Data/train_11.npy", "Data/train_label_11.npy", mode="validation")
test_set  = prep_dataLoader("Data/test_11.npy", mode="test", isPostprocess=False)
```

---

## Notes
- CrossEntropyLoss expects class indices (not one-hot labels).

- The dataset normalization step can be extended (e.g., StandardScaler).

- The same script can be used for TIMIT-48 or TIMIT-39 with minimal edits (change output_dim and relabel data).

---

## 11) Licensing & attribution

* This repository is for educational use. Adapt and extend as needed for your coursework or projects.
* **References**:

  1. The code is based [ML2021-Spring HW02](https://github.com/ga642381/ML2021-Spring/blob/main/HW01/HW01.ipynb).
  2. The dataset and resources are from **ntu-ml-2021spring**.

---

