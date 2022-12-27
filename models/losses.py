import tensorflow as tf
from tensorflow.python.ops import array_ops


def focal_loss(target_tensor, prediction_tensor, weights=None, alpha=0.25, gamma=2):
    """Compute focal loss for predictions

    Multi-labels Focal loss formula: FL = -\alpha * (z-p)^\gamma * \log{(p)} -(1-\alpha) * p^\gamma * \log{(1-p)}
    Which :`\alpha` = 0.25, `\gamma` = 2, p = sigmoid(x), z = target_tensor.
    """

    sigmoid_p = tf.nn.sigmoid(prediction_tensor)
    zeros = array_ops.zeros_like(sigmoid_p)
    pos_p_sub = array_ops.where(target_tensor >= sigmoid_p, target_tensor - sigmoid_p, zeros)
    neg_p_sub = array_ops.where(target_tensor > zeros, zeros, sigmoid_p)
    per_entry_cross_ent = -alpha * (pos_p_sub**gamma) * tf.math.log(  # noqa: E226
        tf.clip_by_value(sigmoid_p, 1e-8, 1.0)
    ) - (1-alpha) * (neg_p_sub**gamma) * tf.math.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))  # noqa: E226

    if weights is not None:
        weights = tf.constant(weights, dtype=per_entry_cross_ent.dtype)
        per_entry_cross_ent *= weights

    return tf.reduce_mean(per_entry_cross_ent)
