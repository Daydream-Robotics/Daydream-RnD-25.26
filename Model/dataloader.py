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

def _draw_heatmap(classes, xs, ys, out_hw, num_classes=NUM_CLASSES, sigma=SIGMA): # Draws the P8 or P16 Heatmap
    H, W = out_hw
    heatmap = tf.zeros((H, W, num_classes), dtype=tf.float32)
    g_size = int(6 * sigma + 1)
    kernel = _generate_gaussian_2d((g_size, g_size), sigma)

    # Autograph: predeclare vars used in loop body to avoid initialization errors
    cls = tf.constant(0, dtype=tf.int32)
    gx = gy = x0 = y0 = x1 = y1 = kx0 = ky0 = kx1 = ky1 = k_w = k_h = tf.constant(0, dtype=tf.int32)

    # Iterates through class list and stamps gaussian for every class
    for i in tf.range(tf.shape(classes)[0]):
        cls = classes[i]
        x_norm = tf.clip_by_value(xs[i], 0.0, 1.0)
        y_norm = tf.clip_by_value(ys[i], 0.0, 1.0)

        # Grid indices (use floor for consistency with offset logic)
        gx = tf.cast(tf.floor(x_norm * tf.cast(W, tf.float32) + 0.5), tf.int32)
        gy = tf.cast(tf.floor(y_norm * tf.cast(H, tf.float32) + 0.5), tf.int32)

        # Clamp to valid bounds in case x==W or y==H
        gx = tf.clip_by_value(gx, 0, W - 1)
        gy = tf.clip_by_value(gy, 0, H - 1)

        # *CHECK FOR BOUNDS ISSUE
        half = g_size // 2

        # Compute patch coordinates in image space
        x0 = gx - half
        y0 = gy - half
        x1 = gx + half + 1
        y1 = gy + half + 1

        # Compute kernel crop indices (kx0 etc.) based on how much falls outside
        kx0 = tf.maximum(0, -x0)
        ky0 = tf.maximum(0, -y0)
        kx1 = g_size - tf.maximum(0, x1 - W)
        ky1 = g_size - tf.maximum(0, y1 - H)

        # Now clamp the image-space coordinates to bounds
        x0 = tf.maximum(0, x0)
        y0 = tf.maximum(0, y0)
        x1 = tf.minimum(W, x1)
        y1 = tf.minimum(H, y1)

        # Only proceed if window is non-empty
        k_w = x1 - x0
        k_h = y1 - y0
        
        def do_scatter():
            # Crop Gaussian accordingly
            patch = kernel[ky0:ky1, kx0:kx1]
            
            # Get actual patch dimensions after slicing to ensure exact match
            patch_h = tf.shape(patch)[0]
            patch_w = tf.shape(patch)[1]
            
            # Build coordinate list for all pixels in patch - use actual patch dimensions
            yy_range = tf.range(y0, y0 + patch_h, dtype=tf.int32)
            xx_range = tf.range(x0, x0 + patch_w, dtype=tf.int32)
            
            # Create meshgrid using broadcasting to ensure exact dimension match
            yy_expanded = tf.expand_dims(yy_range, 1)  # (patch_h, 1)
            xx_expanded = tf.expand_dims(xx_range, 0)  # (1, patch_w)
            yy_grid = tf.broadcast_to(yy_expanded, [patch_h, patch_w])  # (patch_h, patch_w)
            xx_grid = tf.broadcast_to(xx_expanded, [patch_h, patch_w])  # (patch_h, patch_w)
            
            coords = tf.stack([yy_grid, xx_grid, tf.fill([patch_h, patch_w], cls)], axis=-1)
            coords = tf.reshape(coords, [-1, 3])
            values = tf.reshape(patch, [-1])
            
            # Stamp gaussian blur
            return tf.tensor_scatter_nd_max(heatmap, coords, values)
        
        def skip_scatter():
            return heatmap
        
        # Only scatter if we have valid window dimensions
        heatmap = tf.cond(
            tf.logical_and(k_w > 0, k_h > 0),
            do_scatter,
            skip_scatter
        )
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
        # For consistency: use floor(x_norm * W) like in heatmap
        cx = x_norm * tf.cast(W, tf.float32)
        cy = y_norm * tf.cast(H, tf.float32)
        
        # Grid indices (same as heatmap)
        gx = tf.cast(tf.floor(cx), tf.int32)
        gy = tf.cast(tf.floor(cy), tf.int32)
        
        # Clip to valid range (same as heatmap)
        gx = tf.clip_by_value(gx, 0, W - 1)
        gy = tf.clip_by_value(gy, 0, H - 1)
        
        # Calculate offset from cell CENTER (not corner)
        # Cell center is at (gx + 0.5, gy + 0.5) in grid coordinates
        cell_center_x = tf.cast(gx, tf.float32) + 0.5
        cell_center_y = tf.cast(gy, tf.float32) + 0.5
        
        dx = cx - cell_center_x
        dy = cy - cell_center_y

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


    