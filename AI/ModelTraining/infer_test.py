
import tensorflow as tf
from keras import models
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# -------------------------
# CONFIG
# -------------------------
# MODEL_PATH = "/workspace/TensorFlow/Daydream-RnD-25.26/WORKING_MODEL_1.keras"
MODEL_PATH = "/workspace/TensorFlow/Daydream-RnD-25.26/256Model1.keras"
IMG_PATH = "/workspace/TensorFlow/Files/Daydream/Photos/Field/000002.png"
CLASS_TARGET = 0      # RedBall
CONF_THRESH = 0.9
P8_STRIDE = 8
P16_STRIDE = 16

# -------------------------
# LOAD MODEL
# -------------------------
logdir = "logs/profile"
model = models.load_model(MODEL_PATH, compile=False)

# -------------------------
# PREPROCESS
# -------------------------
img = Image.open(IMG_PATH).convert("RGB")
img_resized = img.resize((256, 256))
arr = np.array(img_resized) / 255.0
arr = arr.astype(np.float32)[None, ...]

# -------------------------
# INFER
# -------------------------
# Warmup
for _ in range(10):
    preds = model(arr, training=False)
    _ = tf.reduce_sum(preds["p8"]).numpy()

tf.profiler.experimental.start(logdir)

preds = model(arr, training=False)

p8     = preds["p8"][0]
p8_off = preds["p8_off"][0]
p16 = preds["p16"][0]
p16_off = preds["p16_off"][0]

tf.profiler.experimental.stop()

# FIXED: Handle background channel correctly
# p8 and p16 have shape (H, W, NUM_CLASSES + 1) where last channel is background
p8_probs = tf.nn.softmax(p8, axis=-1).numpy()
p16_probs = tf.nn.softmax(p16, axis=-1).numpy()

# Extract only the target class probability (NOT background)
score_p8 = p8_probs[..., CLASS_TARGET]  # (64,64)
score_p16 = p16_probs[..., CLASS_TARGET]  # (32,32)

# Local maxima (3x3) to avoid clusters
score_tf_p8 = tf.convert_to_tensor(score_p8[None, ..., None], dtype=tf.float32)
score_tf_p16 = tf.convert_to_tensor(score_p16[None, ..., None], dtype=tf.float32)
pooled_p8 = tf.nn.max_pool2d(score_tf_p8, ksize=3, strides=1, padding="SAME")[0, ..., 0].numpy()
pooled_p16 = tf.nn.max_pool2d(score_tf_p16, ksize=3, strides=1, padding="SAME")[0, ..., 0].numpy()

is_peak_p8 = (score_p8 == pooled_p8) & (score_p8 >= CONF_THRESH)
is_peak_p16 = (score_p16 == pooled_p16) & (score_p16 >= CONF_THRESH)

dots_p8 = []
dots_p16 = []

# FIXED: Apply offset mapping that matches training
ys_p8, xs_p8 = np.where(is_peak_p8)
ys_p16, xs_p16 = np.where(is_peak_p16)

for gy, gx in zip(ys_p8, xs_p8):
    dx, dy = p8_off[gy, gx]
    # Grid cell center in image space (matches training coord mapping)
    cx = (gx + float(dx)) * P8_STRIDE
    cy = (gy + float(dy)) * P8_STRIDE
    conf = score_p8[gy, gx]
    dots_p8.append((cx, cy, conf))

for gy, gx in zip(ys_p16, xs_p16):
    dx, dy = p16_off[gy, gx]
    cx = (gx + float(dx)) * P16_STRIDE
    cy = (gy + float(dy)) * P16_STRIDE
    conf = score_p16[gy, gx]
    dots_p16.append((cx, cy, conf))

# -------------------------
# VISUALIZE
# -------------------------
plt.figure(figsize=(8,8))
plt.imshow(img_resized)

# p8 detections (red)
if dots_p8:
    xs8 = [d[0] for d in dots_p8]
    ys8 = [d[1] for d in dots_p8]
    plt.scatter(xs8, ys8, s=20, c="red", label=f"p8 ({len(dots_p8)})")

# p16 detections (blue)
if dots_p16:
    xs16 = [d[0] for d in dots_p16]
    ys16 = [d[1] for d in dots_p16]
    plt.scatter(xs16, ys16, s=40, c="blue", marker="x", label=f"p16 ({len(dots_p16)})")

plt.axis("off")
plt.legend()
plt.savefig("infer_out.png", dpi=150)
plt.show()

# Dump raw outputs with confidence scores
with open("preds.txt","w") as f:
    for x, y, conf in dots_p8:
        f.write(f"p8  {x:.2f}, {y:.2f}, conf={conf:.3f}\n")
    for x, y, conf in dots_p16:
        f.write(f"p16 {x:.2f}, {y:.2f}, conf={conf:.3f}\n")

print(f"✅ Saved: infer_out.png")
print(f"📊 P8 detections: {len(dots_p8)}")
print(f"📊 P16 detections: {len(dots_p16)}")