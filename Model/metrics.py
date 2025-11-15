import tensorflow as tf

class NonAbsolutePeakAccuracy(tf.keras.metrics.Metric):
    """
    Measures accuracy allowing spatial tolerance (within N grid cells).
    """
    def __init__(self, tolerance=1, threshold=0.1, name='nonabs_peak_acc', **kwargs):
        super().__init__(name=name, **kwargs)
        self.tolerance = tolerance  # Grid cells of tolerance
        self.threshold = threshold
        self.correct = self.add_weight(name='correct', initializer='zeros')
        self.total = self.add_weight(name='total', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Find ground truth peaks
        gt_peaks = tf.reduce_max(y_true, axis=-1)  # [B, H, W]
        peak_mask = tf.cast(gt_peaks > self.threshold, tf.float32)
        
        # Get predicted class at each location
        pred_class = tf.argmax(y_pred, axis=-1)  # [B, H, W]
        true_class = tf.argmax(y_true, axis=-1)  # [B, H, W]
        
        # For each peak, check if ANY nearby prediction matches
        batch_size = tf.shape(y_true)[0]
        height = tf.shape(y_true)[1]
        width = tf.shape(y_true)[2]
        
        # This is simplified - for each peak pixel, we check if class matches
        # within tolerance. For full implementation, you'd need spatial pooling.
        
        # Simple version: apply max pooling to dilate the "correct" regions
        matches = tf.cast(tf.equal(pred_class, true_class), tf.float32)
        
        # Dilate matches by tolerance using max pooling
        if self.tolerance > 0:
            matches_expanded = tf.expand_dims(matches, -1)  # [B, H, W, 1]
            kernel_size = 2 * self.tolerance + 1
            matches_dilated = tf.nn.max_pool2d(
                matches_expanded,
                ksize=kernel_size,
                strides=1,
                padding='SAME'
            )
            matches = tf.squeeze(matches_dilated, -1)
        
        # Count matches at peak locations
        correct = matches * peak_mask
        
        self.correct.assign_add(tf.reduce_sum(correct))
        self.total.assign_add(tf.reduce_sum(peak_mask) + 1e-8)

    def result(self):
        return self.correct / (self.total + 1e-8)

    def reset_state(self):
        self.correct.assign(0.0)
        self.total.assign(0.0)

class PeakDetectionAccuracy(tf.keras.metrics.Metric):
    """
    Measures classification accuracy only at true peak centers.
    """
    def __init__(self, threshold=0.1, name='peak_accuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.correct = self.add_weight(name='correct', initializer='zeros')
        self.total = self.add_weight(name='total', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Find TRUE PEAKS in ground truth (local maxima above threshold)
        # We use max across classes to find ANY object peak
        gt_peaks = tf.reduce_max(y_true, axis=-1)  # [B, H, W]
        
        # Consider only strong peaks as ground truth
        peak_mask = tf.cast(gt_peaks > self.threshold, tf.float32)
        
        # Get predicted and true classes
        pred_class = tf.argmax(y_pred, axis=-1)
        true_class = tf.argmax(y_true, axis=-1)
        
        # Check if prediction matches ground truth
        matches = tf.cast(tf.equal(pred_class, true_class), tf.float32)
        
        # Only count matches at peak locations
        correct = matches * peak_mask
        
        self.correct.assign_add(tf.reduce_sum(correct))
        self.total.assign_add(tf.reduce_sum(peak_mask) + 1e-8)

    def result(self):
        return self.correct / (self.total + 1e-8)

    def reset_state(self):
        self.correct.assign(0.0)
        self.total.assign(0.0)


class HeatmapPrecision(tf.keras.metrics.Metric):
    """
    Precision: Of confident predictions, how many are correct?
    """
    def __init__(self, threshold=0.25, name='heatmap_precision', **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.predicted_positives = self.add_weight(name='pp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert logits to probabilities
        pred_probs = tf.nn.softmax(y_pred, axis=-1)
        
        # Get max probability for each pixel
        max_prob = tf.reduce_max(pred_probs, axis=-1)
        
        # Confident predictions (lower threshold for multi-class)
        pred_positive = tf.cast(max_prob > self.threshold, tf.float32)
        
        # Check if predicted class matches ground truth class
        pred_class = tf.argmax(y_pred, axis=-1)
        true_class = tf.argmax(y_true, axis=-1)
        correct = tf.cast(tf.equal(pred_class, true_class), tf.float32)
        
        # Only count where GT has ANY activation (not just peaks)
        gt_present = tf.cast(tf.reduce_max(y_true, axis=-1) > 0.01, tf.float32)
        
        # True positives: correct AND confident AND gt present
        tp = correct * pred_positive * gt_present
        
        self.true_positives.assign_add(tf.reduce_sum(tp))
        self.predicted_positives.assign_add(tf.reduce_sum(pred_positive) + 1e-8)

    def result(self):
        return self.true_positives / (self.predicted_positives + 1e-8)

    def reset_state(self):
        self.true_positives.assign(0.0)
        self.predicted_positives.assign(0.0)

class OffsetMAE(tf.keras.metrics.Metric):
    """
    Mean Absolute Error for offset predictions, only computed where objects exist.
    """
    def __init__(self, name='offset_mae', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total_error = self.add_weight(name='total_error', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Mask: only compute error where offsets are non-zero (objects exist)
        mask = tf.cast(tf.reduce_sum(tf.abs(y_true), axis=-1) > 0, tf.float32)
        
        # Compute absolute error per channel (dx, dy)
        error = tf.abs(y_pred - y_true)
        
        # Average over channels to get per-pixel error
        per_pixel_error = tf.reduce_mean(error, axis=-1)
        
        # Apply mask and accumulate
        masked_error = per_pixel_error * mask
        
        self.total_error.assign_add(tf.reduce_sum(masked_error))
        self.count.assign_add(tf.reduce_sum(mask) + 1e-8)

    def result(self):
        return self.total_error / (self.count + 1e-8)

    def reset_state(self):
        self.total_error.assign(0.0)
        self.count.assign(0.0)


class OffsetAccuracy(tf.keras.metrics.Metric):
    """
    Percentage of offset predictions within a threshold (e.g., 0.3 pixels).
    """
    def __init__(self, threshold=0.3, name='offset_acc', **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.correct = self.add_weight(name='correct', initializer='zeros')
        self.total = self.add_weight(name='total', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Mask: only compute where offsets exist
        mask = tf.cast(tf.reduce_sum(tf.abs(y_true), axis=-1) > 0, tf.float32)
        
        # Euclidean distance between predicted and true offset
        error = tf.sqrt(tf.reduce_sum(tf.square(y_pred - y_true), axis=-1))
        
        # Count predictions within threshold
        within_threshold = tf.cast(error <= self.threshold, tf.float32)
        
        masked_correct = within_threshold * mask
        
        self.correct.assign_add(tf.reduce_sum(masked_correct))
        self.total.assign_add(tf.reduce_sum(mask) + 1e-8)

    def result(self):
        return self.correct / (self.total + 1e-8)

    def reset_state(self):
        self.correct.assign(0.0)
        self.total.assign(0.0)
