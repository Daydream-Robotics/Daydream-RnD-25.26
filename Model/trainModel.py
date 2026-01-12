import tensorflow as tf
import datetime, os
from keras import layers, models, optimizers, callbacks
from model import backbone
from Loss import HeatmapLoss, OffsetLoss
from metrics import NonAbsolutePeakAccuracy, PeakDetectionAccuracy, HeatmapPrecision, OffsetMAE, OffsetAccuracy
from visualizer import visualize_batch_heatmaps, visualize_single_batch
import matplotlib.pyplot as plt
from dataloader import get_dataset, NUM_CLASSES
from pathlib import Path



# --------------------------------
# CONFIG
# --------------------------------

# Train parameters
BATCH_SIZE = 6
EPOCHS = 100
ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ["/workspace/TensorFlow/Files/Daydream/train.tfrecord"]
VAL_PATH = ["/workspace/TensorFlow/Files/Daydream/val.tfrecord"]
INPUT_SHAPE = (512,512,3)
STEPS_PER_EPOCH = 800
VAL_STEPS_PER_EPOCH = 200

# Loss parameters
ALPHA = 0.25
GAMMA = 2.0
DELTA = 1.0
REDUCTION = "mean" # "mean", "sum", or "none"
LAMBDA_CLS = 2.0
LAMBDA_OFFSET = 0.8


# --------------------------------
# Datasets
# --------------------------------

train_ds = get_dataset(TRAIN_PATH, BATCH_SIZE, shuffle_buffer=256, training=True)
val_ds = get_dataset(VAL_PATH, BATCH_SIZE, shuffle_buffer=256, training=False)

# visualize_batch_heatmaps(
#     train_ds,
#     num_samples=100,
#     output_dir="/home/agnco/TensorFlow/Daydream-RnD-25.26/Model/Visualized"
# )

# --------------------------------
# Model Backbone & Setup
# --------------------------------

backbone_model = backbone(INPUT_SHAPE)

# Set input and output
inputs = backbone_model.input
p8, p16 = backbone_model.output

# Define how to make output heads
def make_head(x, num_classes, name):
    x = layers.Conv2D(64, 3, padding="same", activation="relu", name=f"{name}_conv1")(x)
    x = layers.Conv2D(num_classes, 1, padding="same", activation=None, name=name)(x)
    return x

# Generate Heatmap Heads
p8_heatmap = make_head(p8, NUM_CLASSES, name="p8")
p16_heatmap = make_head(p16, NUM_CLASSES, name="p16")

# Offset Heads
p8_offset = make_head(p8, 2, name="p8_off")
p16_offset = make_head(p16, 2, name="p16_off")

# Define Model

model = models.Model(
    inputs,
    outputs={
        "p8": p8_heatmap,
        "p16": p16_heatmap,
        "p8_off": p8_offset,
        "p16_off": p16_offset
    }
)

# DEBUG: Print actual output names
print("\n🔍 Model output names:")
print(model.output_names)
print("\n🔍 Model outputs:")
for name, output in model.output.items():
    print(f"  {name}: {output}")

# --------------------------------
# Compile Model
# --------------------------------

model.compile(
    optimizer=optimizers.Adam(learning_rate=6e-5, clipvalue=1.0), # og 3e-5
    loss={
        "p8": HeatmapLoss(),
        "p16": HeatmapLoss(),
        "p8_off": OffsetLoss(delta=DELTA),     
        "p16_off": OffsetLoss(delta=DELTA)
    },
    loss_weights={
        "p8": LAMBDA_CLS,
        "p16": LAMBDA_CLS,
        "p8_off": LAMBDA_OFFSET,
        "p16_off": LAMBDA_OFFSET
    },
    metrics={
        "p8": [NonAbsolutePeakAccuracy(2, threshold=0.1, name="acc"), HeatmapPrecision(name='prec')],
        "p16": [NonAbsolutePeakAccuracy(1, threshold=0.1, name="acc"), HeatmapPrecision(name='prec')],
        "p8_off": [OffsetMAE(name='mae'), OffsetAccuracy(threshold=0.3, name='acc')],
        "p16_off": [OffsetMAE(name='mae'), OffsetAccuracy(threshold=0.3, name='acc')]
    }
)




# --------------------------------
# Callbacks
# --------------------------------

# Early stopping
early_stop_cb = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# Save Best Model
checkpoint_cb = callbacks.ModelCheckpoint(
    filepath="best_model.keras",
    monitor="val_p16_off_mae",
    mode="min",
    save_best_only=True
)

# Fix Learning Rate Plateau
reduce_lr_cb = callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-9,
    verbose=1
)

# Tensorboard for Graphing (to call run 'tensorboard --logdir=logs/fit' in terminal)
log_dir = f"logs/fit/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
tensorboard_cb = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


# --------------------------------
# Train Model
# --------------------------------


model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_steps=VAL_STEPS_PER_EPOCH,
    callbacks=[early_stop_cb, reduce_lr_cb, checkpoint_cb, tensorboard_cb],
    verbose=2
)