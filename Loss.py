import tensorflow as tf
import keras


# --------------------------------
# Focal Softmax Cross Entropy (Class Loss)
# DO NOT CALL
# --------------------------------

def _focal_softmax(logits, y_true, valid_mask, alpha=0.25, gamma=2.0, reduction="mean"):

    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.where(y_true < 0, tf.zeros_like(y_true), y_true), logits=logits) # tf.where says if you have a negative, replace it with 0, else keep the label

    # p_t : softmax prob of true class
    probs = tf.nn.softmax(logits, axis=-1)
    one_hot = tf.one_hot(tf.maximum(y_true, 0), probs.shape[-1], dtype=tf.float32) # [b, a]
    p_t = tf.reduce_sum(probs * one_hot, axis=-1)

    #focal weighting on positives
    modulating = tf.pow(1.0 - tf.clip_by_value(p_t, 1e-8, 1.0), gamma)
    loss = alpha * modulating * ce

    loss = loss * valid_mask

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

def total_loss(pred_heatmap, y_heatmap, pred_offset, y_offset, valid_mask, alpha, gamma, delta, reduction, lambda_cls=1.0, lambda_offset=1.0):
    valid_mask = tf.cast(pred_heatmap >= 0, tf.float32)

    cls_loss = _focal_softmax(pred_heatmap, y_heatmap, valid_mask, alpha, gamma, reduction)
    offset_loss = _huber(pred_offset, y_offset, valid_mask, delta, reduction)

    return (lambda_cls * cls_loss) + (lambda_offset * offset_loss)