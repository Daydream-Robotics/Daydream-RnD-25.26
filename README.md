# 🧠 Object Detection with Heatmap Regression

This project implements a lightweight object detection pipeline using **centroid-based heatmap regression**. Built with TensorFlow 2.x and Keras, the model predicts class-specific Gaussian heatmaps and offset vectors from multi-scale feature maps.

---

## 📋 Table of Contents

- [Introduction](#introduction)
- [Components](#components)
- [Data Format](#data-format)
- [Model Architecture](#model-architecture)
- [Loss Functions](#loss-functions)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Contributors](#contributors)
- [License](#license)

---

## 🧩 Introduction

This repo offers a minimal yet effective framework for detecting small objects via heatmap regression. It uses a MobileNet-inspired CNN backbone to produce two feature maps (p8 and p16), followed by lightweight heads for heatmap and offset prediction.

---

## 🧱 Components

| File              | Purpose                                                                 |
|-------------------|-------------------------------------------------------------------------|
| `RAWtoTFR.py`     | Converts PNG + JSON annotations into TFRecord format                    |
| `dataloader.py`   | Builds a training pipeline, generates Gaussian heatmaps from TFRecords  |
| `model.py`        | Defines the CNN backbone with bottleneck blocks                         |
| `Loss.py`         | Implements focal and Huber losses for classification and regression     |
| `trainModel.py`   | Orchestrates training: loading data, compiling model, callbacks, etc.   |

---

## 🗂 Data Format

#### Note: It is recommended to create data using Daydream's Dataset Synthesizer

TFRecord examples are generated from:
- `.png` images
- `.json` files containing:
  - `"class"` (e.g. `"RedBall"`)
  - `"projected_cuboid_centroid"` (normalized x, y)

### TFRecord Features:
- `image/encoded`: Raw PNG image bytes
- `objects/classes`: List of class indices
- `objects/xs`: Normalized x coordinates
- `objects/ys`: Normalized y coordinates

### Supported Classes

| ID | Class            |
|----|------------------|
| 0  | RedBall          |
| 1  | BlueBall         |
| 2  | LongGoal         |
| 3  | MiddleGoalTop    |
| 4  | MiddleGoalBottom |

---

## 🏗 Model Architecture

The model backbone is MobileNet-inspired, with inverted bottleneck blocks and residual connections.

### Outputs:
- `p8` → shape: **(40×40×C)**
- `p16` → shape: **(20×20×C)**

### Heads:
- **Heatmap Head**: 5 channels (one per class)
- **Offset Head**: 2 channels (x and y offsets)

---

## 🧠 Loss Functions

| Loss Type     | Description                                                         |
|---------------|---------------------------------------------------------------------|
| **Focal Loss** | Classification loss emphasizing hard examples (α = 0.25, γ = 2.0)   |
| **Huber Loss** | Smooth L1 loss for regression (δ = 2.0)                             |
| **Total Loss** | Weighted sum of focal and huber losses (default λ = 1.0 each)       |

---

## ⚙️ Installation

```bash
git clone https://github.com/your-org/your-repo.git
cd your-repo

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install tensorflow keras
```

---

## 🚀 Usage

### 1. Convert raw data to TFRecords

```bash
python RAWtoTFR.py
```

Make sure the folder defined in `DATA_DIR` contains matching `.png` and `.json` pairs.

### 2. Train the model

Set your TFRecord paths inside `trainModel.py`:
```python
TFRECORD_PATH = ["path/to/train.tfrecord"]
VALIDATION_PATH = ["path/to/val.tfrecord"]
```

Then run:
```bash
python trainModel.py
```

### 3. View logs in TensorBoard

```bash
tensorboard --logdir=logs/fit
```

---

## 🧪 Examples

You can visualize the first parsed TFRecord entry by running:

```bash
python RAWtoTFR.py
```

The script prints out:
- Object class indices
- x/y coordinates

---

## ⚙️ Configuration

Modify the following parameters in `trainModel.py` as needed:

| Parameter         | Purpose                             | Default        |
|------------------|-------------------------------------|----------------|
| `BATCH_SIZE`      | Batch size for training             | 16             |
| `EPOCHS`          | Number of training epochs           | 50             |
| `ALPHA`           | Focal loss alpha                    | 0.25           |
| `GAMMA`           | Focal loss gamma                    | 2.0            |
| `DELTA`           | Huber loss delta                    | 2.0            |
| `LAMBDA_CLS`      | Weight for classification loss      | 1.0            |
| `LAMBDA_OFFSET`   | Weight for offset loss              | 1.0            |

---

## 🛠 Troubleshooting

- **TFRecord not found**: Ensure the paths in `trainModel.py` are correct.
- **No annotations detected**: Check that your JSON includes `"projected_cuboid_centroid"` and `"class"`.
- **Slow training**: Ensure you're using GPU and TensorFlow is properly installed.

---

## 👥 Contributors

- [Alexander Nardi] — Creator
- [UCF7 - Daydream]
- Open to community contributions!

---

## 📄 License
