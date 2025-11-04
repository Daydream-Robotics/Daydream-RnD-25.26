import tensorflow as tf
import datetime, os
from keras import layers, models, optimizers, callbacks
from model import backbone
from Loss import HeatmapLoss, OffsetLoss
from metrics import PeakDetectionAccuracy, HeatmapPrecision, OffsetMAE, OffsetAccuracy
from visualizer import visualize_allclass_heatmaps
import matplotlib.pyplot as plt
from dataloader import get_dataset, NUM_CLASSES



# --------------------------------
# CONFIG
# --------------------------------

# Train parameters
BATCH_SIZE = 16
EPOCHS = 50
TRAIN_PATH = ["/media/agn/BAC8-CC11/Daydream/train.tfrecord"]
VAL_PATH = ["/media/agn/BAC8-CC11/Daydream/val.tfrecord"]
INPUT_SHAPE = (512,512,3)
STEPS_PER_EPOCH = 600
VAL_STEPS_PER_EPOCH = 150

# Loss parameters
ALPHA = 0.25
GAMMA = 2.0
DELTA = 2.0
REDUCTION = "mean" # "mean", "sum", or "none"
LAMBDA_CLS = 1.0
LAMBDA_OFFSET = 1.0


# --------------------------------
# Datasets
# --------------------------------

train_ds = get_dataset(TRAIN_PATH, batch_size=16, shuffle_buffer=256, training=True)
val_ds = get_dataset(VAL_PATH, batch_size=16, shuffle_buffer=256, training=False)

imgs, labels = next(iter(train_ds))
img = imgs[0]
p8_map = labels["p8"][0]
p16_map = labels["p16"][0]

visualize_allclass_heatmaps(img, p8_map, p16_map, )

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
    x = tf.keras.layers.Conv2D(num_classes, 1, padding="same", activation=None, name=name)(x)
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
    optimizer=optimizers.Adam(learning_rate=1e-4, clipnorm=1.0),
    loss={
        "p8": HeatmapLoss(),
        "p16": HeatmapLoss(),
        "p8_off": OffsetLoss(delta=DELTA),     
        "p16_off": OffsetLoss(delta=DELTA)
    },
    loss_weights={
        "p8": 1.0,
        "p16": 1.0,
        "p8_off": 0.5,
        "p16_off": 0.5
    },
    metrics={
        "p8": [PeakDetectionAccuracy(name='acc'), HeatmapPrecision(name='prec')],
        "p16": [PeakDetectionAccuracy(name='acc'), HeatmapPrecision(name='prec')],
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
    monitor="val_loss",
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
    callbacks=[early_stop_cb, reduce_lr_cb, checkpoint_cb, tensorboard_cb]
)