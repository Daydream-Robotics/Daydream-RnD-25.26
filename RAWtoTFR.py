import tensorflow as tf
import json, time, math, random, sys
from pathlib import Path

# CONFIG
DATA_DIR = Path("/media/agn/BAC8-CC11/Daydream/Photos/Field")
OUT_TRAIN = "/media/agn/BAC8-CC11/Daydream/train.tfrecord"
OUT_VAL = "//media/agn/BAC8-CC11/Daydream/val.tfrecord"
SPLIT_RATIO = 0.8

classes = {
    "RedBall": 0,
    "BlueBall": 1,
    "Fillet5": 2,
    "Mirror1_2": 3
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

    # Check existence
    if not img_path.exists() or not json_path.exists():
        print(f"⚠️ Skipping {stem}: missing pair ({img_path.exists()=}, {json_path.exists()=})")
        return None

    # Read files
    try:
        img_bytes = img_path.read_bytes()
        label_json = json.loads(json_path.read_text())
    except Exception as e:
        print(f"❌ Error reading {stem}: {e}")
        return None

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

    print(f"🖼️ {stem}: {len(cls)} objects, size=({w}x{h})")

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
# Write TFRecords
# --------------------------------
def write_tfrecord(pairs, out_path):
    count = 0
    with tf.io.TFRecordWriter(out_path) as writer:
        for i, stem in enumerate(pairs, start=1):
            example = make_example(stem)
            if example is not None:
                writer.write(example.SerializeToString())
                count += 1
            if i % 10 == 0 or i == len(pairs):
                print(f"Progress: {i}/{len(pairs)} ({count} written)")
                sys.stdout.flush()
    return count

# --------------------------------
# Main
# --------------------------------
start = time.time()

pairs = sorted(p.stem for p in DATA_DIR.glob("*.json"))
random.shuffle(pairs)

split_idx = int(len(pairs) * SPLIT_RATIO)
train_pairs = pairs[:split_idx]
val_pairs = pairs[split_idx:]

print(f"🔧 Found {len(pairs)} JSON files → {len(train_pairs)} train / {len(val_pairs)} val")

train_count = write_tfrecord(train_pairs, OUT_TRAIN)
val_count = write_tfrecord(val_pairs, OUT_VAL)

elapsed = time.time() - start
print(f"\n✅ Done in {elapsed:.2f}s")
print(f"✅ Wrote {train_count} training examples → {OUT_TRAIN}")
print(f"✅ Wrote {val_count} validation examples → {OUT_VAL}")

# Verification
tfr_dataset = tf.data.TFRecordDataset([OUT_TRAIN])
for record in tfr_dataset.take(1):
    example = tf.train.Example()
    example.ParseFromString(record.numpy())
    f = example.features.feature
    print("\n🔍 Verification of first record:")
    print("  Classes:", f["objects/classes"].int64_list.value)
    print("  Xs:", f["objects/xs"].float_list.value)
    print("  Ys:", f["objects/ys"].float_list.value)
