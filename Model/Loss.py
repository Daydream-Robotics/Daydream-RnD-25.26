import tensorflow as tf
import keras

# -------
# Focal Softmax With Background Class and Per-Pixel Weights
# Logits: [B,H,W,C] C = 3 (red, blue, bg)
# y_true: [B,H,W,C] one-hot or soft
# fg_mask: [B,H,W] 1 where object exists (any foreground), otherwise 0 (background)
# -------
def _focal_softmax_dense_bg(logits, y_true, fg_mask, alpha_fg=0.75, alpha_bg=0.25, gamma=2.0, neg_weight=1.0, reduction="mean", eps=1e-8):
    y_true = tf.clip_by_value(y_true, 0.0, 1.0)
    den = tf.reduce_sum(y_true, axis=-1, keepdims=True) + eps
    y_true = y_true / den
    
    probs = tf.nn.softmax(logits, axis=-1)
    
    ce = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=logits)
    p_t = tf.reduce_sum(probs * y_true, axis=-1)
    
    focal = tf.pow(1.0 - tf.clip_by_value(p_t, eps, 1.0), gamma)
    
    #alpha balance fg vs bg
    alpha = fg_mask * alpha_fg + (1.0 - fg_mask) * alpha_bg
    
    # extra down weight on bg pixels
    weight = fg_mask * 1.0 + (1.0 - fg_mask) * neg_weight
    
    loss = alpha * focal * ce * weight # [B,H,W]
    
    if reduction == "mean":
        denom = tf.reduce_sum(weight) + eps
        return tf.reduce_sum(loss) / denom
    elif reduction == "sum":
        return tf.reduce_sum(loss)
    else:
        return loss

# --------------------------------
# Focal Softmax Cross Entropy (Class Loss)
# DO NOT CALL
# --------------------------------

def _focal_softmax(logits, y_true, valid_mask, alpha=0.25, gamma=2.0, reduction="mean"):
    """
    Focal loss for *dense* soft labels (heatmaps), not sparse integer labels.
    """

    # Normalize y_true (in case it's not perfectly normalized)
    y_true = tf.clip_by_value(y_true, 0.0, 1.0)
    den = tf.reduce_sum(y_true, axis=-1, keepdims=True) + 1e-8
    y_true = y_true / den

    probs = tf.nn.softmax(logits, axis=-1)

    # Compute standard CE per-pixel
    ce = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=logits)

    # p_t = probability assigned to the true class (from y_true weights)
    p_t = tf.reduce_sum(probs * y_true, axis=-1)

    # Focal modulation
    modulating = tf.pow(1.0 - tf.clip_by_value(p_t, 1e-8, 1.0), gamma)
    loss = alpha * modulating * ce

    # Apply valid mask
    # Ensure valid_mask matches [batch, H, W]
    if len(valid_mask.shape) == 4 and valid_mask.shape[-1] != 1:
        valid_mask = tf.reduce_max(valid_mask, axis=-1)

    # Mask the Numerator
    loss = loss * valid_mask 

    if reduction == "mean":
        num_positives = tf.reduce_sum(valid_mask) + 1e-8
        return tf.reduce_sum(loss) / num_positives
    elif reduction == "sum":
        return tf.reduce_sum(loss)
    else:
        return loss


# --------------------------------
# Huber on Offset (Centroid Loss)
# DO NOT CALL
# --------------------------------
    
def _huber_vector(pred, target, delta):
    err = pred-target
    abs_e = tf.abs(err)
    quad = 0.5 * tf.square(abs_e)
    lin = delta * (abs_e - (0.5 * delta))
    huber = tf.where(abs_e <= delta, quad, lin)
    return tf.reduce_sum(huber, axis=-1)


def _huber(prediction, target, valid_mask, delta=1.0, reduction="mean"):
    #mask positives
    per_cell = _huber_vector(prediction, target, delta)
    loss = per_cell * valid_mask

    if reduction == "mean":
        denom = tf.reduce_sum(valid_mask) + 1e-8
        return tf.reduce_sum(loss) / denom
    elif reduction == "sum":
        return tf.reduce_sum(loss)
    else:
        return loss
    

# --------------------------------
# Compute Loss
# CALL - from Loss import total_loss
# --------------------------------

def total_loss(pred_heatmap=None, y_heatmap=None,
               pred_offset=None, y_offset=None,
               valid_mask=None,
               alpha=0.25, gamma=2.0, delta=2.0,
               reduction="mean",
               lambda_cls=1.0, lambda_offset=1.0):
    """
    Flexible loss function that computes classification loss, offset loss, or both,
    depending on which tensors are provided.
    """

    total = 0.0

    if pred_heatmap is not None and y_heatmap is not None and lambda_cls > 0.0:
        cls_mask = valid_mask if valid_mask is not None else tf.ones_like(y_heatmap[..., 0])
        cls_loss = _focal_softmax(pred_heatmap, y_heatmap, cls_mask, alpha, gamma, reduction)
        total += lambda_cls * cls_loss

    if pred_offset is not None and y_offset is not None and lambda_offset > 0.0:
        off_mask = valid_mask if valid_mask is not None else tf.reduce_max(tf.cast(y_offset > 0, tf.float32), axis=-1)
        offset_loss_val = _huber(pred_offset, y_offset, off_mask, delta, reduction)
        total += lambda_offset * offset_loss_val

    return total

class HeatmapLoss(tf.keras.losses.Loss):
    def __init__(self, alpha_fg=0.75, alpha_bg=0.25, neg_weight=0.25, gamma=2.0, reduction="mean", name="HeatmapLoss"):
        super().__init__(name=name)
        self.alpha_fg = alpha_fg
        self.alpha_bg = alpha_bg
        self.gamma = gamma
        self.neg_weight = neg_weight
        self.reduction_type = reduction

    def call(self, y_true, y_pred):
        # Foreground presence mask
        fg_strength = tf.reduce_max(y_true, axis=-1)
        fg_mask = tf.cast(fg_strength > 0.01, tf.float32)
        
        # Build bg channel from fg heatmaps (assumes no occlusion)
        fg_sum = tf.reduce_sum(y_true, axis=-1, keepdims=True)
        bg = tf.clip_by_value(1.0 - fg_sum, 0.0, 1.0)
        y_true_full = tf.concat([y_true, bg], axis=-1)
        
        return _focal_softmax_dense_bg(
            logits=y_pred,
            y_true = y_true_full,
            fg_mask=fg_mask,
            alpha_fg=self.alpha_fg,
            alpha_bg=self.alpha_bg,
            gamma=self.gamma,
            neg_weight=self.neg_weight,
            reduction=self.reduction_type
        )

class DeprecatedHeatmapLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean", name="HeatmapLoss"):
        super().__init__(name=name)
        self.alpha = alpha
        self.gamma = gamma
        self.reduction_type = reduction

    def call(self, y_true, y_pred):
        valid_mask = tf.reduce_max(y_true, axis=-1)  # [B, H, W]
        valid_mask = tf.cast(valid_mask > 0.01, tf.float32)  # Threshold to get object regions
        
        return total_loss(
            pred_heatmap=y_pred,
            y_heatmap=y_true,
            pred_offset=None,
            y_offset=None,
            valid_mask=valid_mask,  # Use real mask!
            alpha=self.alpha,
            gamma=self.gamma,
            delta=1.0,
            reduction=self.reduction_type,
            lambda_cls=1.0,
            lambda_offset=0.0
        )


class OffsetLoss(tf.keras.losses.Loss):
    def __init__(self, delta=1.0, reduction="mean", name="OffsetLoss"):
        super().__init__(name=name)
        self.delta = delta
        self.reduction_type = reduction

    def call(self, y_true, y_pred):
        # Use your custom _huber function which handles vectors correctly
        mask = tf.cast(tf.reduce_sum(tf.abs(y_true), axis=-1) > 0, tf.float32)
        return _huber(y_pred, y_true, mask, delta=self.delta, reduction=self.reduction_type)