import tensorflow as tf
import os
import math

# CONFIG

IMG_SIZE = (256,256)
NUM_CLASSES = 2 # Red Ball, Blue Ball
SIGMA = 1.5 # Gaussian Radius
LOG_PATH = "/home/agnco/TensorFlow/Daydream-RnD-25.26/debug_stamps.txt"


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
    img = tf.image.resize(img, IMG_SIZE, method="bilinear")
    classes = tf.cast(tf.sparse.to_dense(f["objects/classes"]), tf.int32)
    xs = tf.sparse.to_dense(f["objects/xs"])
    ys = tf.sparse.to_dense(f["objects/ys"])
    return img, (classes, xs, ys)


# --------------------------------
# Heatmap Generator
# DO NOT CALL
# --------------------------------

def _generate_gaussian_2d(size, sigma):
    ax = tf.range(size[1], dtype=tf.float32)
    ay = tf.range(size[0], dtype=tf.float32)
    xx, yy = tf.meshgrid(ax, ay)
    cx = (tf.cast(size[1], tf.float32) - 1.0) / 2.0
    cy = (tf.cast(size[0], tf.float32) - 1.0) / 2.0
    return tf.exp(-((xx-cx)**2 + (yy-cy)**2) / (2.0*sigma**2))

def _draw_heatmap(classes, xs, ys, out_hw, num_classes=NUM_CLASSES, sigma=SIGMA):
    H, W = out_hw
    
    g_size = int(6 * sigma + 1)
    half = g_size // 2
    pad = half
    heatmap = tf.zeros((H + 2*pad, W + 2*pad, num_classes), dtype=tf.float32)
    kernel = _generate_gaussian_2d((g_size, g_size), sigma)

    for i in tf.range(tf.shape(classes)[0]):
        cls = classes[i]
        x_norm = tf.clip_by_value(xs[i], 0.0, 1.0)
        y_norm = tf.clip_by_value(ys[i], 0.0, 1.0)

        # FIXED: Better coordinate mapping
        # Maps [0, 1] to [0, W-1] continuously, then rounds
        gx_float = x_norm * tf.cast(W - 1, tf.float32)
        gy_float = y_norm * tf.cast(H - 1, tf.float32)
        
        gx = tf.cast(tf.round(gx_float), tf.int32)
        gy = tf.cast(tf.round(gy_float), tf.int32)

        # Already in valid range [0, W-1] and [0, H-1]
        gx = tf.clip_by_value(gx, 0, W - 1)
        gy = tf.clip_by_value(gy, 0, H - 1)

        gx_pad = gx + pad
        gy_pad = gy + pad

        x0 = gx_pad - half
        y0 = gy_pad - half
        x1 = gx_pad + half + 1
        y1 = gy_pad + half + 1

        kx0 = tf.maximum(0, -x0)
        ky0 = tf.maximum(0, -y0)
        kx1 = g_size - tf.maximum(0, x1 - (W + 2*pad))
        ky1 = g_size - tf.maximum(0, y1 - (H + 2*pad))

        x0 = tf.maximum(0, x0)
        y0 = tf.maximum(0, y0)
        x1 = tf.minimum(W + 2*pad, x1)
        y1 = tf.minimum(H + 2*pad, y1)

        patch = kernel[ky0:ky1, kx0:kx1]
        ph = tf.shape(patch)[0]
        pw = tf.shape(patch)[1]

        yy = tf.range(y0, y0 + ph, dtype=tf.int32)
        xx = tf.range(x0, x0 + pw, dtype=tf.int32)

        yy = tf.expand_dims(yy, 1)
        xx = tf.expand_dims(xx, 0)

        yy_grid = tf.broadcast_to(yy, [ph, pw])
        xx_grid = tf.broadcast_to(xx, [ph, pw])

        coords = tf.stack([yy_grid, xx_grid, tf.fill([ph, pw], cls)], axis=-1)
        coords = tf.reshape(coords, [-1, 3])
        values = tf.reshape(patch, [-1])

        heatmap = tf.tensor_scatter_nd_max(heatmap, coords, values)

    heatmap = heatmap[pad:pad+H, pad:pad+W, :]
    return heatmap


def _draw_offset_map(xs, ys, out_hw):
    """
    FIXED: Uses same coordinate mapping as heatmap for consistency
    """
    H, W = out_hw
    offsets = tf.zeros((H, W, 2), dtype=tf.float32)

    for i in tf.range(tf.shape(xs)[0]):
        x_norm = tf.clip_by_value(xs[i], 0.0, 1.0)
        y_norm = tf.clip_by_value(ys[i], 0.0, 1.0)
        
        # FIXED: Same mapping as heatmap
        gx_float = x_norm * tf.cast(W - 1, tf.float32)
        gy_float = y_norm * tf.cast(H - 1, tf.float32)
        
        gx = tf.cast(tf.round(gx_float), tf.int32)
        gy = tf.cast(tf.round(gy_float), tf.int32)
        
        gx = tf.clip_by_value(gx, 0, W - 1)
        gy = tf.clip_by_value(gy, 0, H - 1)
        
        # Offset from rounded position
        dx = gx_float - tf.cast(gx, tf.float32)
        dy = gy_float - tf.cast(gy, tf.float32)

        offsets = tf.tensor_scatter_nd_update(
            offsets, [[gy, gx, 0]], [dx]
        )
        offsets = tf.tensor_scatter_nd_update(
            offsets, [[gy, gx, 1]], [dy]
        )

    return offsets


def _to_heatmaps(img, data):
    """
    Given parsed (classes, xs, ys), generates:
      - P8 and P16 heatmaps (NUM_CLASSES channels - NO background)
      - P8 and P16 offset maps (dx, dy)
    """
    classes, xs, ys = data

    h = tf.shape(img)[0]
    w = tf.shape(img)[1]
    p8_hw = (h // 8, w // 8)
    p16_hw = (h // 16, w // 16)
    
    hm_p8 = _draw_heatmap(classes, xs, ys, out_hw=p8_hw,
                          num_classes=NUM_CLASSES, sigma=SIGMA)
    hm_p16 = _draw_heatmap(classes, xs, ys, out_hw=p16_hw,
                           num_classes=NUM_CLASSES, sigma=SIGMA)

    off_p8 = _draw_offset_map(xs, ys, out_hw=p8_hw)
    off_p16 = _draw_offset_map(xs, ys, out_hw=p16_hw)

    return img, {
        "p8": hm_p8,
        "p16": hm_p16,
        "p8_off": off_p8,
        "p16_off": off_p16
    }


# --------------------------------
# Full Dataset Builder
# CALL - from dataloader import get_dataset
# --------------------------------

def get_dataset(tfrecord_paths, batch_size, shuffle_buffer=256, training=True, cache=False):
    """
    FIXED: Better shuffling strategy for large datasets
    
    Args:
        cache: If True, cache parsed records (use only if dataset fits in RAM)
    """
    # Create dataset from files
    ds = tf.data.TFRecordDataset(tfrecord_paths)
    
    # Optional: Shuffle file order for multi-file datasets
    if training and len(tfrecord_paths) > 1:
        ds = ds.shuffle(buffer_size=len(tfrecord_paths), reshuffle_each_iteration=True)
    
    # Parse records
    ds = ds.map(_parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Optional caching (only if your dataset fits in RAM)
    if cache:
        ds = ds.cache()
    
    # IMPORTANT: Shuffle BEFORE repeat for better randomization
    if training:
        ds = ds.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
    
    # Repeat indefinitely
    ds = ds.repeat()
    
    # Generate heatmaps
    ds = ds.map(_to_heatmaps, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Batch and prefetch
    ds = ds.batch(batch_size, drop_remainder=training)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    
    return ds