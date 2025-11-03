import tensorflow as tf
import json, time, math, random
from pathlib import Path

# CONFIG
DATA_DIR = Path("/home/agn/Datasets/Dataset-Raw")
OUT_TRAIN = "/home/agn/Datasets/train.tfrecord"
OUT_VAL = "/home/agn/Datasets/val.tfrecord"

SPLIT_RATIO = 0.8  # 80% train / 20% val

classes = {
    "RedBall": 0,
    "BlueBall": 1,
    "LongGoal": 2,
    "MiddleGoalTop": 3,
    "MiddleGoalBottom": 4
}

def _bytes_feature(b): return tf.train.Feature(bytes_list=tf.train.BytesList(value=[b]))
def _int64_feature(v): return tf.train.Feature(int64_list=tf.train.Int64List(value=v))
def _float_feature(v): return tf.train.Feature(float_list=tf.train.FloatList(value=v))

# --------------------------------
# Create Tensorflow Example (tf.train.Example)
# --------------------------------
def make_example(stem: str):
    img_path = DATA_DIR / f"{stem}.png"
    json_path = DATA_DIR / f"{stem}.json"

    # Skip if either file missing
    if not img_path.exists() or not json_path.exists():
        print(f"⚠️ Skipping {stem} (missing pair)")
        return None

    img_bytes = img_path.read_bytes()
    label_json = json.loads(json_path.read_text())

    # Get image shape
    img = tf.io.decode_png(img_bytes, channels=3)
    h, w = img.shape[0], img.shape[1]

    cls, xs, ys = [], [], []
    for o in label_json.get("objects", []):
        if "projected_cuboid_centroid" not in o or "class" not in o:
            continue
        if o["class"] not in classes:
            continue
        x, y = o["projected_cuboid_centroid"]
        cls.append(classes[o["class"]])
        xs.append(float(x) / w)
        ys.append(float(y) / h)

    feature = {
        "image/encoded": _bytes_feature(img_bytes),
        "image/height": _int64_feature([h]),
        "image/width": _int64_feature([w]),
        "objects/count": _int64_feature([len(cls)]),
        "objects/classes": _int64_feature(cls),
        "objects/xs": _float_feature(xs),
        "objects/ys": _float_feature(ys)
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


# --------------------------------
# Main
# --------------------------------
start = time.time()

# Collect & shuffle stems
pairs = sorted(p.stem for p in DATA_DIR.glob("*.json"))
random.shuffle(pairs)

# Split 80/20
split_idx = int(len(pairs) * SPLIT_RATIO)
train_pairs = pairs[:split_idx]
val_pairs = pairs[split_idx:]

def write_tfrecord(pairs, out_path):
    count = 0
    with tf.io.TFRecordWriter(out_path) as writer:
        for stem in pairs:
            example = make_example(stem)
            if example is not None:
                writer.write(example.SerializeToString())
                count += 1
    return count

# Write train + val files
train_count = write_tfrecord(train_pairs, OUT_TRAIN)
val_count = write_tfrecord(val_pairs, OUT_VAL)

# Print with verify
elapsed = time.time() - start
print(f"✅ Wrote {train_count} training examples to {OUT_TRAIN}")
print(f"✅ Wrote {val_count} validation examples to {OUT_VAL}")
print(f"⏱️ Total time: {elapsed:.2f}s")

# Verification step (first record from train)
tfr_dataset = tf.data.TFRecordDataset([OUT_TRAIN])
for record in tfr_dataset.take(1):
    example = tf.train.Example()
    example.ParseFromString(record.numpy())
    f = example.features.feature
    print("Classes:", f["objects/classes"].int64_list.value)
    print("Xs:", f["objects/xs"].float_list.value)
    print("Ys:", f["objects/ys"].float_list.value)
