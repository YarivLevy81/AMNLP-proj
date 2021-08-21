
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow
tf = tensorflow.compat.v1

#https://github.com/hpandana/gradient-accumulation-tf-estimator/blob/master/another-example.py
def train_op_fn(loss, init_lr, num_train_steps, num_warmup_steps, gradient_accumulation_multiplier):
    """Returns the op to optimize the loss."""

    global_step = tf.train.get_global_step()

    # warmup
    learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)

    # Implements linear decay of the learning rate.
    learning_rate = tf.train.polynomial_decay(
        learning_rate,
        global_step,
        num_train_steps,
        end_learning_rate=0.0,
        power=1.0,
        cycle=False)

    global_steps_int = tf.cast(global_step, tf.int32)
    warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

    global_steps_float = tf.cast(global_steps_int, tf.float32)
    warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

    warmup_percent_done = global_steps_float / warmup_steps_float
    warmup_learning_rate = learning_rate * warmup_percent_done

    is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
    learning_rate = ((1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)

    # It is recommended that you use same optimizer for fine tuning
    # (note that the Adam m/v variables are NOT loaded from init_checkpoint.) FIXME
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # accumulate gradients
    tvars = tf.trainable_variables()
    grads = tf.gradients(loss, tvars)
    accum_grads = [tf.Variable(tf.zeros_like(t_var.initialized_value()), trainable=False) for t_var in
                   tvars]

    def apply_accumulated_gradients(accum_grads, grads, tvars):
        accum_op = tf.group([accum_grad.assign_add(grad) for (accum_grad, grad) in zip(accum_grads, grads)])
        with tf.control_dependencies([accum_op]):
            normalized_accum_grads = [1.0 * accum_grad / gradient_accumulation_multiplier for accum_grad in
                                      accum_grads]
            # global_step is not incremented inside optimizer.apply_gradients
            minimize_op = optimizer.apply_gradients(zip(normalized_accum_grads, tvars), global_step=None)
            with tf.control_dependencies([minimize_op]):
                zero_op = tf.group(
                    [accum_grad.assign(tf.zeros_like(accum_grad)) for accum_grad in accum_grads])
        return zero_op

    # Create training operation
    train_op = tf.cond(tf.math.equal(global_step % gradient_accumulation_multiplier, 0),
                       lambda: apply_accumulated_gradients(accum_grads, grads, tvars),
                       lambda: tf.group(
                           [accum_grad.assign_add(grad) for (accum_grad, grad) in zip(accum_grads, grads)])
                       )

    # global_step is incremented here, regardless of the tf.cond branch
    train_op = tf.group(train_op, [tf.assign_add(global_step, 1)])
    return train_op