import tensorflow as tf
import json, time, math
from pathlib import Path


DATA_DIR = Path("/home/agn/Datasets/Dataset-Raw")
OUT_PATH = "/home/agn/Datasets/train.tfrecord"

# Dictionary of classes
classes = {
    "RedBall": 0,
    "BlueBall": 1,
    "LongGoal": 2,
    "MiddleGoalTop": 3,
    "MiddleGoalBottom": 4
}

# TensorFlow declarations
def _bytes_feature(b): return tf.train.Feature(bytes_list=tf.train.BytesList(value=[b]))
def _int64_feature(v): return tf.train.Feature(int64_list=tf.train.Int64List(value=v))
def _float_feature(v): return tf.train.Feature(float_list=tf.train.FloatList(value=v))


# For each JSON file, convert it to TF Example
def make_example(stem: str):
    img_path = DATA_DIR / f"{stem}.png"
    json_path = DATA_DIR / f"{stem}.json"

    # Ensure both img path and json path exist (skips stuff like _camera_settings and _object_settings)
    if not img_path.exists() or not json_path.exists():
        print(f"⚠️ Skipping {stem} (missing pair)")
        return None

    # Read in data
    img_bytes = img_path.read_bytes()
    label_json = json.loads(json_path.read_text())

    # Get image shape
    img = tf.io.decode_png(img_bytes, channels=3)
    h, w = img.shape[0], img.shape[1]

    # Declaring feature vars
    cls, xs, ys = [], [], []

    # Parse json for "projected_cuboid_centroid" and "class"
    for o in label_json.get("objects", []):
        if "projected_cuboid_centroid" not in o:
            continue
        if "class" not in o:
            continue
        x, y = o["projected_cuboid_centroid"]
        if o["class"] not in classes:
            continue  # skip unknown labels
        cls.append(classes[o["class"]])
        xs.append(float(x) / w) 
        ys.append(float(y) / h)

    # assign features
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

#time it
start = time.time()

# Sort jsons and pngs
pairs = sorted(p.stem for p in DATA_DIR.glob("*.json"))

#Write as TFRecord
count = 0
with tf.io.TFRecordWriter(OUT_PATH) as writer:
    for stem in pairs:
        example = make_example(stem)
        if example is not None:
            writer.write(example.SerializeToString())
            count += 1

elapsed = time.time() - start
print(f"✅ Wrote {count} examples to {OUT_PATH} over {elapsed:.2f} seconds")

filenames = [OUT_PATH]
tfr_dataset = tf.data.TFRecordDataset(filenames)

#Print out first record in the dataset
for record in tfr_dataset.take(1):
    example = tf.train.Example()
    example.ParseFromString(record.numpy())

    # Print selected fields only (we dont need the whole serialized png to be printed)
    f = example.features.feature
    print("Classes:", f["objects/classes"].int64_list.value)
    print("Xs:", f["objects/xs"].float_list.value)
    print("Ys:", f["objects/ys"].float_list.value)
