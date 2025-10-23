import tensorflow as tf
import datetime, os
from keras import layers, models, optimizers, callbacks
from model import backbone
from Loss import total_loss
from dataloader import get_dataset

# --------------------------------
# CONFIG
# --------------------------------

# Train parameters
BATCH_SIZE = 16
EPOCHS = 50
TFRECORD_PATH = [""]
VALIDATION_PATH = [""]
INPUT_SHAPE = (320,320,3)

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

train_ds = get_dataset(TFRECORD_PATH, BATCH_SIZE, shuffle_buffer=256, training=True)
val_ds = get_dataset(VALIDATION_PATH, BATCH_SIZE, shuffle_buffer=256, training=False)

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
    x = tf.keras.layers.Conv2D(num_classes, 1, padding="same", activation=None, name=f"{name}_conv2")(x)
    return x

# Generate Heatmap Heads
p8_heatmap = make_head(p8, 5, name="p8_heatmap")
p16_heatmap = make_head(p16, 5, name="p16_heatmap")

# Offset Heads
p8_offset = make_head(p8, 2, name="p8_offset")
p16_offset = make_head(p16, 2, name="p16_offset")

# Define model
model = models.Model(inputs, [p8_heatmap, p16_heatmap, p8_offset, p16_offset])


# --------------------------------
# Custom Loss Wrapper
# --------------------------------

def custom_loss(y_true, y_pred):
    y_p8 = y_true["p8"]
    y_p16 = y_true["p16"]

    pred_cls_p8, pred_cls_p16, pred_off_p8, pred_off_p16 = y_pred

    loss_p8 = total_loss(pred_cls_p8, y_p8, pred_off_p8, tf.zeros_like(pred_off_p8), pred_off_p8, ALPHA, GAMMA, DELTA, REDUCTION, LAMBDA_CLS, LAMBDA_OFFSET)
    loss_p16 = total_loss(pred_cls_p16, y_p16, pred_off_p16, tf.zeros_like(pred_off_p16), pred_off_p16, ALPHA, GAMMA, DELTA, REDUCTION, LAMBDA_CLS, LAMBDA_OFFSET)
    return loss_p8 + loss_p16


# --------------------------------
# Compile Model
# --------------------------------

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss=custom_loss
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
    min_lr=1e-6,
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
    callbacks=[early_stop_cb, reduce_lr_cb, checkpoint_cb, tensorboard_cb]
)