import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# -------------------------
# CONFIG
# -------------------------
MODEL_PATH = "/home/agn/ProgramSpace/TensorFlow/Daydream-RnD-25.26/best_model.keras"
IMG_PATH = "/media/agn/BAC8-CC11/Daydream/Photos/Field/000002.png"
CLASS_TARGET = 0       # RedBall
CONF_THRESH = 0.8      # *** REAL CONF THRESH ***
P8_STRIDE = 8
P16_STRIDE = 16

# -------------------------
# LOAD MODEL
# -------------------------
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

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
p8, p16 = model.predict(arr)

h8, w8 = p8.shape[1], p8.shape[2]

cls_map = p8[0, :, :, :5]    # softmax inputs (logits)
off_map = p8[0, :, :, 5:]    # offsets (dx, dy)

# Apply softmax per cell
cls_probs = tf.nn.softmax(cls_map, axis=-1).numpy()

dots = []

for gy in range(h8):
    for gx in range(w8):

        probs = cls_probs[gy, gx]          # [5 classes]
        bg_prob = probs[0]
        best_class = np.argmax(probs)
        best_conf = probs[best_class]

        # REAL detection rule
        if best_class == 0: continue               # ignore bg
        if best_conf < CONF_THRESH: continue       # require high confidence
        if best_class - 1 != CLASS_TARGET: continue # map: 1=red,2=blue,...

        dx, dy = off_map[gy, gx]
        cx = (gx + dx) * P8_STRIDE
        cy = (gy + dy) * P8_STRIDE

        dots.append((cx, cy))

# -------------------------
# VISUALIZE
# -------------------------
plt.figure(figsize=(8,8))
plt.imshow(img_resized)
xs = [d[0] for d in dots]
ys = [d[1] for d in dots]
plt.scatter(xs, ys, s=20, c="red")
plt.axis("off")
plt.savefig("infer_out.png", dpi=150)
plt.show()

# Also dump raw outputs
with open("preds.txt","w") as f:
    for x,y in dots:
        f.write(f"{x:.2f}, {y:.2f}\n")

print("Saved: infer_out.png")