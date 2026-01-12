import tensorflow as tf
from keras import models
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# -------------------------
# CONFIG
# -------------------------
MODEL_PATH = "/workspace/TensorFlow/Daydream-RnD-25.26/best_model.keras"
IMG_PATH = "/workspace/TensorFlow/Files/Daydream/Photos/Field/000002.png"
CLASS_TARGET = 0      # RedBall
CONF_THRESH = 0.9    # *** REAL CONF THRESH ***
P8_STRIDE = 8
P16_STRIDE = 16

# -------------------------
# LOAD MODEL
# -------------------------
model = models.load_model(MODEL_PATH, compile=False)

# -------------------------
# PREPROCESS
# -------------------------
img = Image.open(IMG_PATH).convert("RGB")
img_resized = img.resize((512, 512))
arr = np.array(img_resized) / 255.0
arr = arr.astype(np.float32)[None, ...]

# -------------------------
# INFER
# -------------------------
preds = model.predict(arr)

p8     = preds["p8"][0]      # (64,64,4) logits
p8_off = preds["p8_off"][0]  # (64,64,2)
p16 = preds["p16"][0]
p16_off = preds["p16_off"][0]


# Convert logits -> probs (consistent with your training loss using softmax)
p8_probs = tf.nn.softmax(p8, axis=-1).numpy()
p16_probs = tf.nn.softmax(p16, axis=-1).numpy()

score_p8 = p8_probs[..., CLASS_TARGET]  # (64,64) probability for your target class
score_p16 = p16_probs[..., CLASS_TARGET]

# Local maxima (3x3) to avoid clusters of dots
score_tf_p8 = tf.convert_to_tensor(score_p8[None, ..., None], dtype=tf.float32)  # (1,H,W,1)
score_tf_p16 = tf.convert_to_tensor(score_p16[None, ..., None], dtype=tf.float32)
pooled_p8 = tf.nn.max_pool2d(score_tf_p8, ksize=3, strides=1, padding="SAME")[0, ..., 0].numpy()
pooled_p16 = tf.nn.max_pool2d(score_tf_p16, ksize=3, strides=1, padding="SAME")[0, ..., 0].numpy()

is_peak_p8 = (score_p8 == pooled_p8) & (score_p8 >= CONF_THRESH)
is_peak_p16 = (score_p16 == pooled_p16) & (score_p16 >= CONF_THRESH)

dots_p8 = []
dots_p16 = []
ys_p8, xs_p8 = np.where(is_peak_p8)
ys_p16, xs_p16 = np.where(is_peak_p16)
for gy, gx in zip(ys_p8, xs_p8):
    dx, dy = p8_off[gy, gx]
    cx = (gx + float(dx)) * P8_STRIDE
    cy = (gy + float(dy)) * P8_STRIDE
    dots_p8.append((cx, cy))

for gy,gx in zip(ys_p16, xs_p16):
    dx, dy = p16_off[gy,gx]
    cx = (gx + float(dx)) * P16_STRIDE
    cy = (gy + float(dy)) * P16_STRIDE
    dots_p16.append((cx,cy))

# -------------------------
# VISUALIZE
# -------------------------
plt.figure(figsize=(8,8))
plt.imshow(img_resized)

# p8 detections (red)
xs8 = [d[0] for d in dots_p8]
ys8 = [d[1] for d in dots_p8]
plt.scatter(xs8, ys8, s=20, c="red", label="p8")

# p16 detections (blue)
xs16 = [d[0] for d in dots_p16]
ys16 = [d[1] for d in dots_p16]
plt.scatter(xs16, ys16, s=40, c="blue", marker="x", label="p16")

plt.axis("off")
plt.legend()
plt.savefig("infer_out.png", dpi=150)
plt.show()

# Also dump raw outputs
with open("preds.txt","w") as f:
    for x,y in dots_p8:
        f.write(f"p8  {x:.2f}, {y:.2f}\n")
    for x,y in dots_p16:
        f.write(f"p16 {x:.2f}, {y:.2f}\n")

print("Saved: infer_out.png")