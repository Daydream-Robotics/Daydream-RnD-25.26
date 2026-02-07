import tensorflow as tf
import json, time, random, sys
from pathlib import Path

# ----------------------------------------
# CONFIG
# ----------------------------------------
DATA_DIR = Path("/home/agnco/Files/NDDS/Null")

OUT_TRAIN = "/home/agnco/Files/NDDS/TFrecords/trainNULL.tfrecord"
OUT_VAL   = "/home/agnco/Files/NDDS/TFrecords/valTestNULL.tfrecord"

SPLIT_RATIO = 0.8

# 2 Classes â†’ must match model order
CLASSES = {
    "RedBall": 0,
    "BlueBall": 1,
}

# ----------------------------------------
# TF Helpers
# ----------------------------------------
def _bytes_feature(b): return tf.train.Feature(bytes_list=tf.train.BytesList(value=[b]))
def _int64_feature(v): return tf.train.Feature(int64_list=tf.train.Int64List(value=v))
def _float_feature(v): return tf.train.Feature(float_list=tf.train.FloatList(value=v))


# ----------------------------------------
# Create a single TF Example from (image, json)
# ----------------------------------------
def make_example(stem: str):
    img_path  = DATA_DIR / f"{stem}.png"
    json_path = DATA_DIR / f"{stem}.json"

    if not img_path.exists() or not json_path.exists():
        print(f"âš  Skipping {stem} (missing image or JSON)")
        return None

    try:
        img_bytes = img_path.read_bytes()
        data = json.loads(json_path.read_text())
    except Exception as e:
        print(f"âŒ Failed reading {stem}: {e}")
        return None

    # Decode PNG shape quickly
    img = tf.io.decode_png(img_bytes, channels=3)
    h, w = img.shape[0], img.shape[1]  # expected 512x512

    classes = []
    xs = []
    ys = []

    for obj in data.get("objects", []):
        cname = obj.get("class")
        if cname not in CLASSES:
            continue

        # Must have projected cuboid & centroid
        projected = obj.get("projected_cuboid")
        centroid  = obj.get("projected_cuboid_centroid")
        if projected is None or centroid is None:
            continue

        cx, cy = centroid

        # ------------------------
        # RULE 1: centroid must be inside frame
        # ------------------------
        if not (0 <= cx < w and 0 <= cy < h):
            continue

        # ------------------------
        # RULE 2: At least 2 vertices must be visible
        # ------------------------
        inside_count = sum(0 <= px < w and 0 <= py < h for (px, py) in projected)
        if inside_count < 2:
            continue

        # Valid object â†’ normalize and store
        classes.append(CLASSES[cname])
        xs.append(cx / w)
        ys.append(cy / h)

    # Build TF Example
    features = {
        "image/encoded": _bytes_feature(img_bytes),
        "image/height":  _int64_feature([h]),
        "image/width":   _int64_feature([w]),
        "objects/count": _int64_feature([len(classes)]),
        "objects/classes": _int64_feature(classes),
        "objects/xs": _float_feature(xs),
        "objects/ys": _float_feature(ys),
    }

    return tf.train.Example(features=tf.train.Features(feature=features))


# ----------------------------------------
# Write TFRecords
# ----------------------------------------
def write_tfrecord(stems, out_path):
    written = 0
    with tf.io.TFRecordWriter(out_path) as writer:
        for i, stem in enumerate(stems):
            example = make_example(stem)
            if example:
                writer.write(example.SerializeToString())
                written += 1

            if i % 20 == 0:
                print(f"Progress: {i}/{len(stems)} (written {written})")

    print(f"âœ” Finished {out_path}: {written} records")
    return written


# ----------------------------------------
# MAIN
# ----------------------------------------
start = time.time()

# List all JSON stems
pairs = sorted(p.stem for p in DATA_DIR.glob("*.json"))
random.shuffle(pairs)

split = int(len(pairs) * SPLIT_RATIO)
train_pairs = pairs[:split]
val_pairs   = pairs[split:]

print(f"Found {len(pairs)} frames â†’ {len(train_pairs)} train / {len(val_pairs)} val")

# Write files
train_written = write_tfrecord(train_pairs, OUT_TRAIN)
val_written   = write_tfrecord(val_pairs, OUT_VAL)

print(f"\nDone in {time.time()-start:.2f}s")
print(f"Train: {train_written}")
print(f"Val:   {val_written}")

# ----------------------------------------
# Verify first record is readable
# ----------------------------------------
tfr = tf.data.TFRecordDataset([OUT_TRAIN])
for raw in tfr.take(1):
    ex = tf.train.Example()
    ex.ParseFromString(raw.numpy())
    f = ex.features.feature
    print("\nVerification sample:")
    print("  classes:", list(f["objects/classes"].int64_list.value))
    print("  xs:", list(f["objects/xs"].float_list.value))
    print("  ys:", list(f["objects/ys"].float_list.value))