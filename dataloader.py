import tensorflow as tf
import math

# CONFIG

NUM_CLASSES = 5
P8_HW = (40,40) # Height, Width p8
P16_HW = (20,20) # Height, Width, p16
SIGMA = 1.5 # Gaussian Radius


# --------------------------------
# TFRecord Parser
# DO NOT CALL
# --------------------------------

def _parse_tfrecord(example_proto):
    feature_spec = {
        "image/encoded": tf.io.FixedLenFeature([], tf.string),
        "objects/classes": tf.io.VarLenFeature(tf.int64),
        "objects/xs": tf.io.VarLenFeature(tf.float32),
        "objects/ys": tf.io.VarLenFeature(tf.float32)
    }
    f = tf.io.parse_single_example(example_proto, feature_spec)
    img = tf.image.decode_png(f["image/encoded"], channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    classes = tf.cast(tf.sparse.to_dense(f["objects/classes"]), tf.int32)
    xs = tf.sparse.to_dense(f["objects/xs"])
    ys = tf.sparse.to_dense(f["objects/ys"])
    return img, (classes, xs, ys)


# --------------------------------
# Heatmap Generator
# DO NOT CALL
# --------------------------------

def _generate_gaussian_2d(size, sigma): # Generates a 2D Gaussian Kernel
    ax = tf.range(size[1], dtype=tf.float32)
    ay = tf.range(size[0], dtype=tf.float32)
    xx, yy = tf.meshgrid(ax, ay)
    cx, cy = (size[1]-1)/2.0, (size[0]-1)/2.0
    return tf.exp(-((xx-cx)**2 + (yy-cy)**2) / (2.0*sigma**2))

def _draw_heatmap(classes, xs, ys, out_hw, num_classes=NUM_CLASSES, sigma=SIGMA): # Draws the P8 or P16 Heatmap
    H, W = out_hw
    heatmap = tf.zeros((H, W, num_classes), dtype=tf.float32)
    g_size = int(6 * sigma + 1)
    kernel = _generate_gaussian_2d((g_size, g_size), sigma)

    # Iterates through class list and stamps gaussian for every class
    for i in tf.range(tf.shape(classes)[0]):
        cls = classes[i]
        x = tf.cast(tf.round(xs[i] * (W-1)), tf.int32)
        y = tf.cast(tf.round(ys[i] * (H-1)), tf.int32)

        #top left edge of gaussian
        x0 = tf.maximum(0, x - g_size // 2)
        y0 = tf.maximum(0, y - g_size // 2)
        #bottom right edge of gaussian, clipped to bounds
        x1 = tf.minimum(W, x0 + g_size)
        y1 = tf.minimum(H, y0 + g_size)
        #determine how much of gaussian can be used (ex: might only be able to use 3x10 if on left border)
        kx0 = g_size - (x1 - x0)
        ky0 = g_size - (y1 - y0)
        patch = kernel[ky0:, kx0:]
        patch = patch[:y1 - y0, :x1 - x0]

        # Build coordinate list for all pixels in patch
        yy, xx = tf.meshgrid(tf.range(y0,y1), tf.range(x0,x1), indexing="ij")
        coords = tf.stack([yy,xx, tf.fill(tf.shape(yy), cls)], axis = -1)
        coords = tf.reshape(coords, [-1,3])
        values = tf.reshape(patch, [-1])

        # Stamp gaussian blur
        heatmap = tf.tensor_scatter_nd_max(heatmap, coords, values)
    return heatmap

def _to_heatmaps(img, data): # Create a P8 and P16 resolution heatmap for training
    classes, xs, ys = data
    hm_p8 = _draw_heatmap(classes, xs, ys, out_hw=P8_HW, num_classes=NUM_CLASSES, sigma=SIGMA)
    hm_p16 = _draw_heatmap(classes, xs, ys, out_hw=P16_HW, num_classes=NUM_CLASSES, sigma=SIGMA)
    return img, {"p8": hm_p8, "p16": hm_p16}


# --------------------------------
# Full Dataset Builder
# CALL - from dataloader import get_dataset
# --------------------------------

def get_dataset(tfrecord_paths, batch_size, shuffle_buffer=256, training=True):
    ds = tf.data.TFRecordDataset(tfrecord_paths)
    ds = ds.map(_parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(_to_heatmaps, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        ds = ds.shuffle(shuffle_buffer)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

    