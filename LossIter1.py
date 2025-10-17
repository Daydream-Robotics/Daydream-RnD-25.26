import tensorflow as tf
print("Tensorflow ver:", tf.__version__)
from keras import Model, layers
from keras import losses, optimizers, metrics
import datetime

'''
output & y_true structure:
    logits: (batch, x, y, K) - class labels per object
    offset: (batch, x, y, 2) - offset from center for each bounding box corresponding to its logit
    size:   (batch, x, y, 2) - size of bounding box around offset corresponding to its logit
'''



'''
input: (B, A, K); output: scalar loss
inputs: logits - results of prediction, y_true - class labels, alpha & gamma - values to adjust the weighitng of "hard" problems, typically 0.25 and 2, reduction - "mean", "sum" or "none"

valid_mask: y_true holds negative values whenever an object shouldnt be detected in that cell, checking if y_true >= 0 and casting to float every negative space becomes a 0 and cancels, so we dont calculate loss on negative space
ce: standard softmax cross entropy on the labels, tf.where is needed because the CE function cannot accept negative values

p_t - [b, a] where b is an image within the batch, a is a cell within the image, p_t[b,a] = models probability output for true class
    probs = values adding up to one (representing the probabilities that the model thought oject a within batch b was of class k)
    one_hot - selects from y_true which label is supposed to be true (retains that as 1, all others become 0)
    p_t - elementwise multiply the probs * one hot to isolate the prob, then sum over the last dimension, our class dimension, to collapse this into a scalar; this is our models prediction for the correct class
'''
def softmax_focal(logits, y_true, alpha=0.25, gamma=2.0, reduction="mean"):
    # mask: positives only
    valid_mask = tf.cast(y_true >= 0, tf.float32)  # (B, A)

    # CE on positives
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.where(y_true < 0, tf.zeros_like(y_true), y_true), logits=logits) # tf.where says if you have a negative, replace it with 0, else keep the label

    # p_t: softmax prob of the true class
    probs = tf.nn.softmax(logits, axis=-1) # [b, a, k]
    one_hot = tf.one_hot(tf.maximum(y_true, 0), probs.shape[-1], dtype=tf.float32) # [b, a]
    p_t = tf.reduce_sum(probs * one_hot, axis=-1)

    # focal weighting on positives
    modulating = tf.pow(1.0 - tf.clip_by_value(p_t, 1e-8, 1.0), gamma)
    loss = alpha * modulating * ce

    #apply mask
    loss = loss * valid_mask

    if reduction == "mean":
        denom = tf.reduce_sum(valid_mask) + 1e-8
        return tf.reduce_sum(loss) / denom
    elif reduction == "sum":
        return tf.reduce_sum(loss)
    else:
        return loss


'''
input: (B, X, Y, 2); Output: (B, X, Y) where (B, X, Y) = one cell in batch B. Note: X,Y is equivalent to 'A' above
calculates the error, then absolute error, then uses tf.where as an if statement. If abs_e > delta, it calculates loss using lin, else uses quad (IT DOES THIS ON A ELEMENT-WISE BASIS)
returns the sum of the dx,dy (offset) or the width/height error (size); this is the 2 in (B,X,Y,2)
'''
def _huber_vector(pred, target, delta):
    err = pred-target
    abs_e = tf.abs(err)
    quad = 0.5 * tf.square(abs_e)
    lin = delta * (abs_e - (0.5 * delta))
    huber = tf.where(abs_e <= delta, quad, lin)
    return tf.reduce_sum(huber, axis=-1)


'''
input: (B, X, Y, 2); output: scalar loss
pred = model pred, target = target off/size, valid_mask = mask for non-negatives, delta = threshold
Does huber on a cell-by-cell, aka (B, X, Y), basis. Multiplies this by the valid mask, returns as scalar.
'''
def huber(prediction, target, valid_mask, delta=2.0, reduction="mean"):
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
    

'''
Calculates the total loss from detection
Total loss: focal_softmax * l_cls + huber_off * l_off + huber_size * l_size + l2
Note: l2 loss is included as an optimizer
'''
def detection_loss(
        # where A = cell (X,Y) flattened
        logits, # (B, A, K)
        cls_targets, # (B, A) where -1 = ignore, 0->K = class
        pred_offset, # (B, A, 2) - (dx, dy)
        tgt_offset, # (B, A, 2)
        pred_size, # (B, A, 2) - (w, h)
        tgt_size, # (B, A, 2)
        *,
        alpha=0.25,gamma=2.0,delta_offset=2.0, delta_size=2.0, l_cls=1.0, l_sze=1.0, l_off=1.0, l_l2=1e-4, #lambdas for loss adjustment
        l2_vars=None, # e.g model.trainable_variables
        reduction="mean" # "mean", "sum", or "none"
):
    # Returns scalar total loss (if reduction != "none")

    #valid mask
    valid_mask = tf.cast(cls_targets >= 0, tf.float32)

    #classification (focal softmax)
    cls_loss = softmax_focal(logits, cls_targets, alpha=alpha, gamma=gamma, reduction=reduction)

    #offset & size (huber)
    off_loss = huber(pred_offset, tgt_offset, valid_mask, delta=delta_offset, reduction=reduction)
    size_loss = huber(pred_size, tgt_size, valid_mask, delta=delta_size, reduction=reduction)

    #l2 will be included in the layers as tfa.optimizers.AdamW()
    total = l_cls * cls_loss + l_sze * size_loss + l_off * off_loss
    return total, {"cls": cls_loss, "size": size_loss, "off": off_loss}





