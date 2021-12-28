#!/usr/bin/env python
# coding: utf-8
# FM（Factorization Machines，因子分解机）
"""
优点：
将二阶交叉特征考虑进来提高模型的表达能力；
引入隐向量，缓解了数据稀疏带来的参数难训练问题；
改进为高阶特征组合时仍为线性复杂度有利于上线应用。

缺点：
虽然考虑了特征的交叉但是表达能力仍然有限不及深度模型。

思考：
当前全部为分类特征时，特征可以直接使用参数向量表达
"""

import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow import random_normal_initializer as init
import tensorflow.keras.backend as K


class FM_layer(tf.keras.layers.Layer):
    def __init__(self, k, w_reg, v_reg, f_n):
        super().__init__()
        self.k = k  # 隐向量vi的维度
        self.w_reg = w_reg  # 权重w的正则项系数
        self.v_reg = v_reg  # 权重v的正则项系数
        self.w0 = self.add_weight(shape=(1,), initializer=tf.zeros_initializer(), trainable=True)
        self.w = self.add_weight(shape=(f_n, 1), initializer=init(), trainable=True, regularizer=l2(self.w_reg))
        self.v = self.add_weight(shape=(f_n, self.k), initializer=init(), trainable=True, regularizer=l2(self.v_reg))

    def call(self, inputs):
        input_dim = K.ndim(inputs)
        if input_dim != 2:
            raise ValueError('Unexpected inputs dimensions %d, expect to be 2' % input_dim)

        # 线性部分 shape:(batch_size, 1)
        linear_part = tf.matmul(inputs, self.w) + self.w0  #
        # 交叉部分 shape:(batch_size, self.k)
        # 交叉部分——第一项
        inter_1 = tf.pow(tf.matmul(inputs, self.v), 2)
        # 交叉部分——第二项
        inter_2 = tf.matmul(tf.pow(inputs, 2), tf.pow(self.v, 2))
        # 交叉结果
        inter_part = 0.5 * tf.reduce_sum(inter_1 - inter_2, axis=-1, keepdims=True)
        # 模型输出
        return tf.nn.sigmoid(linear_part + inter_part)


class FM_Model(tf.keras.Model):
    def __init__(self, k, w_reg=1e-4, v_reg=1e-4, f_n=None):
        super().__init__()
        self.fm = FM_layer(k, w_reg, v_reg, f_n)

    def call(self, inputs, training=None, mask=None):
        output = self.fm(inputs)
        return output
