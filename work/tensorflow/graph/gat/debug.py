import tensorflow as tf
from tensorflow.python.ops import control_flow_util

control_flow_util.ENABLE_CONTROL_FLOW_V2 = True

print(tf.__version__)
print(tf.executing_eagerly())
