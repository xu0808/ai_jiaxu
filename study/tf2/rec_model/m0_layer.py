#!/usr/bin/env python
# coding: utf-8
# 通用layer

import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Embedding
from tensorflow.keras.regularizers import l2
from tensorflow import random_normal_initializer as init


class Line_layer(Layer):
    def __init__(self, feature_num):
        super().__init__()
        self.b = self.add_weight(name='b', shape=(1,),
                                 initializer=tf.zeros_initializer(), trainable=True)
        self.w = self.add_weight(name='w', shape=(feature_num, 1),
                                 initializer=init(), trainable=True, regularizer=l2(1e-4))

    def call(self, inputs):
        x = tf.matmul(inputs, self.w) + self.b
        return x


class Deep_layer(Layer):
    def __init__(self, hidden_units, output_dim, activation):
        super().__init__()
        self.hidden_layer = [Dense(i, activation=activation) for i in hidden_units]
        # 输出层没有激活函数
        self.output_layer = Dense(output_dim, activation=None)

    def call(self, inputs):
        x = inputs
        for layer in self.hidden_layer:
            x = layer(x)
        output = self.output_layer(x)
        return output

