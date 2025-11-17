import tensorflow as tf
import os
import math

# CONFIG

NUM_CLASSES = 4
P8_HW = (64,64) # Height, Width p8
P16_HW = (32,32) # Height, Width, p16
SIGMA = 1.5 # Gaussian Radius
LOG_PATH = "/home/agn/ProgramSpace/TensorFlow/Daydream-RnD-25.26/debug_stamps.txt"


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
    # Cast to float32 before division to avoid dtype mismatch
    cx = (tf.cast(size[1], tf.float32) - 1.0) / 2.0
    cy = (tf.cast(size[0], tf.float32) - 1.0) / 2.0
    return tf.exp(-((xx-cx)**2 + (yy-cy)**2) / (2.0*sigma**2))

def _draw_heatmap(classes, xs, ys, out_hw, num_classes=NUM_CLASSES, sigma=SIGMA):
    H, W = out_hw
    
    # Gaussian radius (half-width in pixels)
    g_size = int(6 * sigma + 1)
    half = g_size // 2

    # PADDED heatmap to avoid border clipping
    pad = half
    heatmap = tf.zeros((H + 2*pad, W + 2*pad, num_classes), dtype=tf.float32)

    # Full Gaussian kernel
    kernel = _generate_gaussian_2d((g_size, g_size), sigma)

    for i in tf.range(tf.shape(classes)[0]):
        cls = classes[i]
        x_norm = tf.clip_by_value(xs[i], 0.0, 1.0)
        y_norm = tf.clip_by_value(ys[i], 0.0, 1.0)

        # Map normalized coords to grid coordinates
        # Using round for better edge handling
        gx = tf.cast(tf.round(x_norm * tf.cast(W, tf.float32)), tf.int32)
        gy = tf.cast(tf.round(y_norm * tf.cast(H, tf.float32)), tf.int32)

        # Clamp to valid bounds
        gx = tf.clip_by_value(gx, 0, W - 1)
        gy = tf.clip_by_value(gy, 0, H - 1)

        # Convert to padded coordinates
        gx_pad = gx + pad
        gy_pad = gy + pad

        # Compute patch region in padded coords
        x0 = gx_pad - half
        y0 = gy_pad - half
        x1 = gx_pad + half + 1
        y1 = gy_pad + half + 1

        # Crop kernel if it spills out of padded bounds
        kx0 = tf.maximum(0, -x0)
        ky0 = tf.maximum(0, -y0)
        kx1 = g_size - tf.maximum(0, x1 - (W + 2*pad))
        ky1 = g_size - tf.maximum(0, y1 - (H + 2*pad))

        # Clamp image coords
        x0 = tf.maximum(0, x0)
        y0 = tf.maximum(0, y0)
        x1 = tf.minimum(W + 2*pad, x1)
        y1 = tf.minimum(H + 2*pad, y1)

        # Final patch
        patch = kernel[ky0:ky1, kx0:kx1]

        # Meshgrid for scatter
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

    # Crop back to original shape (center region)
    heatmap = heatmap[pad:pad+H, pad:pad+W, :]
    return heatmap


def _draw_offset_map(xs, ys, out_hw):
    """
    Builds a 2-channel offset map (dx, dy) for sub-cell localization.
    Each cell at (gy, gx) where an object center falls stores
    how far the true (x, y) is from that cell center.
    Uses the same coordinate mapping as _draw_heatmap for consistency.
    """
    H, W = out_hw
    offsets = tf.zeros((H, W, 2), dtype=tf.float32)

    for i in tf.range(tf.shape(xs)[0]):
        # Normalize coordinates (same as heatmap)
        x_norm = tf.clip_by_value(xs[i], 0.0, 1.0)
        y_norm = tf.clip_by_value(ys[i], 0.0, 1.0)
        
        # Convert to grid coordinates (same mapping as heatmap)
        gx_float = x_norm * tf.cast(W, tf.float32)
        gy_float = y_norm * tf.cast(H, tf.float32)
        
        # Grid indices (same as heatmap - use round)
        gx = tf.cast(tf.round(gx_float), tf.int32)
        gy = tf.cast(tf.round(gy_float), tf.int32)
        
        # Clip to valid range
        gx = tf.clip_by_value(gx, 0, W - 1)
        gy = tf.clip_by_value(gy, 0, H - 1)
        
        # Calculate offset from the rounded grid position
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
      - P8 and P16 heatmaps
      - P8 and P16 offset maps (dx, dy)
    """
    classes, xs, ys = data

    # Create heatmaps
    hm_p8 = _draw_heatmap(classes, xs, ys, out_hw=P8_HW,
                          num_classes=NUM_CLASSES, sigma=SIGMA)
    hm_p16 = _draw_heatmap(classes, xs, ys, out_hw=P16_HW,
                           num_classes=NUM_CLASSES, sigma=SIGMA)

    # Create offset maps
    off_p8 = _draw_offset_map(xs, ys, out_hw=P8_HW)
    off_p16 = _draw_offset_map(xs, ys, out_hw=P16_HW)

    # Return combined structure
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

def get_dataset(tfrecord_paths, batch_size, shuffle_buffer=256, training=True):
    ds = tf.data.TFRecordDataset(tfrecord_paths)
    ds = ds.map(_parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(_to_heatmaps, num_parallel_calls=tf.data.AUTOTUNE)

    def force_output_order(img, data):
        return img, {
            "p8": data["p8"],             # Heatmap @ 64x64
            "p16": data["p16"],           # Heatmap @ 32x32
            "p8_off": data["p8_off"],     # Offset @ 64x64
            "p16_off": data["p16_off"]    # Offset @ 32x32
        }

    ds = ds.map(force_output_order, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        ds = ds.shuffle(shuffle_buffer)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    ds=ds.repeat()
    return ds