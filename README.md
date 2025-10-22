# Object Detection with Heatmap Regression

This project implements a lightweight keypoint-based object detection pipeline using TensorFlow 2.x. Objects are detected by regressing Gaussian heatmaps and optional offsets.

## 🔧 Components

- **`RAWtoTFR.py`**: Converts PNG + JSON annotations into TFRecord format.
- **`dataloader.py`**: Builds a heatmap-based dataset pipeline from TFRecords.
- **`model.py`**: Implements an efficient CNN backbone with MobileNet-style bottlenecks.
- **`Loss.py`**: Custom losses (Focal + Huber) for classification and regression.

## 🗂 Data Format

TFRecords are generated from:
- `.png` images
- `.json` files with `"projected_cuboid_centroid"` and `"class"` per object

Each record includes:
- Encoded image
- Object classes
- Normalized x, y positions

## 🏗 Model Architecture

The backbone produces feature maps at two resolutions:
- **p8**: 40×40
- **p16**: 20×20

Future heads can use these maps to predict:
- Heatmaps
- Offsets

## 🧠 Loss Functions

- **Focal Loss**: Penalizes class imbalance in heatmap detection.
- **Huber Loss**: Penalizes L1/L2 errors for regressed offsets.

## 🚀 Training (coming soon)

Create `training.py` to:
1. Load `train.tfrecord`
2. Use `backbone()` for the base model
3. Add prediction heads
4. Compile with `total_loss` from `Loss.py`
5. Train and evaluate

## 📦 Requirements

- TensorFlow 2.x
- Keras
- Python 3.8+

## 🛠 Example Usage

```bash
# Generate TFRecord
python RAWtoTFR.py

# Train model (after training.py is created)
python training.py

