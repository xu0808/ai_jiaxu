# TensorFlow实现卷积
import tensorflow as tf
import numpy as np


def corr2d(x, w, b, pad, strides):
    N, H, W, C = tf.shape(x)
    F, HH, WW, C = tf.shape(w)

    # 填充
    x = tf.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant')
    Hn = 1 + int((H + 2 * pad - HH) / strides[0])
    Wn = 1 + int((W + 2 * pad - WW) / strides[1])
    Y = tf.Variable(tf.zeros((N, Hn, Wn, F), dtype=tf.float32))

    # 一次处理所有样本
    for m in range(F):
        for i in range(Hn):
            for j in range(Wn):
                # 当前第n个样本，第m个输出通道，（i,j）位置的值为：
                # 起止点（i * strides[0]，j * strides[1]）, 高HH, 宽WW
                (h0, w0) = (i * strides[0], j * strides[1])
                (h1, w1) = (h0 + HH, w0 + WW)
                # 当前卷积块（h0, w0, h1, w1）所有通道
                data = x[:, h0:h1, w0:w1, :]
                # 第m个过滤器权重
                filt = w[m, :, :, :]
                # 当前点卷积结果为:点乘+偏值
                Y[:, i, j, m].assign(tf.reduce_sum(tf.multiply(data, filt), axis=(1, 2, 3)) + b[m])

    return Y


inputs_ = np.random.random((10, 28, 28, 3)).astype(np.float32)
w = np.random.random((6, 3, 3, 3)).astype(np.float32)
b = np.random.random((6,)).astype(np.float32)

x = tf.dtypes.cast(tf.constant(inputs_), dtype=tf.float32)
b_tf = tf.dtypes.cast(tf.constant(w), dtype=tf.float32)
b_tf = tf.dtypes.cast(tf.constant(b), dtype=tf.float32)

corr2d(x, w, b, 1, (1, 1))


class Conv2D(tf.keras.layers.Layer):
    def __init__(self, filters=None, kernel_size=(3, 3), stride=(1, 1), pad=0, **kwargs):
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = 0

        super(Conv2D, self).__init__(**kwargs)

    def build(self, input_shape):
        self.w = self.add_weight(name='w',
                                 shape=(self.filters, input_shape[-1], self.kernel_size[0], self.kernel_size[1]),
                                 initializer=tf.random_normal_initializer())
        self.b = self.add_weight(name='b',
                                 shape=(self.filters,),
                                 initializer=tf.random_normal_initializer())

        super(Conv2D, self).build(input_shape)

    def call(self, inputs):
        return self.corr2d(inputs, self.w, self.b, self.pad, self.stride)

    @staticmethod
    def corr2d(x, w, b, pad, strides):
        N, H, W, C = tf.shape(x)
        F, HH, WW, C = tf.shape(w)

        # 填充
        x = tf.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant')
        Hn = 1 + int((H + 2 * pad - HH) / strides[0])
        Wn = 1 + int((W + 2 * pad - WW) / strides[1])
        Y = tf.Variable(tf.zeros((N, Hn, Wn, F), dtype=tf.float32))

        # 一次处理所有样本
        for m in range(F):
            for i in range(Hn):
                for j in range(Wn):
                    # 当前第n个样本，第m个输出通道，（i,j）位置的值为：
                    # 起止点（i * strides[0]，j * strides[1]）, 高HH, 宽WW
                    (h0, w0) = (i * strides[0], j * strides[1])
                    (h1, w1) = (h0 + HH, w0 + WW)
                    # 当前卷积块（h0, w0, h1, w1）所有通道
                    data = x[:, h0:h1, w0:w1, :]
                    # 第m个过滤器权重
                    filt = w[m, :, :, :]
                    # 当前点卷积结果为:点乘+偏值
                    Y[:, i, j, m].assign(tf.reduce_sum(tf.multiply(data, filt), axis=(1, 2, 3)) + b[m])

        return Y


conv2 = Conv2D(filters=6, kernel_size=(3, 3), stride=(1, 1), pad=0)
conv2(x)
