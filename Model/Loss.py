import tensorflow as tf
import keras


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


    if reduction == "mean":
        denom = tf.reduce_sum(valid_mask) + 1e-8
        return tf.reduce_sum(loss) / denom
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


def _huber(prediction, target, valid_mask, delta=2.0, reduction="mean"):
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
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean", name="HeatmapLoss"):
        super().__init__(name=name)
        self.alpha = alpha
        self.gamma = gamma
        self.reduction_type = reduction

    def call(self, y_true, y_pred):
        return total_loss(
            pred_heatmap=y_pred,
            y_heatmap=y_true,
            pred_offset=tf.zeros_like(y_pred),
            y_offset=tf.zeros_like(y_pred),
            valid_mask=tf.ones_like(y_true[..., 0]),  # dummy mask
            alpha=self.alpha,
            gamma=self.gamma,
            delta=1.0,
            reduction=self.reduction_type,
            lambda_cls=1.0,
            lambda_offset=0.0  # disable offset loss
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