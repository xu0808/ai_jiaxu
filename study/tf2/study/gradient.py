import numpy as np
import tensorflow as tf


def himmelblau(x):
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


# [1., 0.], [-4, 0.], [4, 0.]
x = tf.constant([4., 0.])

for step in range(30):
    with tf.GradientTape() as tape:
        tape.watch([x])
        y = himmelblau(x)

    grads = tape.gradient(y, [x])[0]
    x -= 0.01 * grads  # 学习率*grads

    print('step {}: x = {}, f(x) = {}'
          .format(step, x.numpy(), y.numpy()))
