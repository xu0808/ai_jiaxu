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
from tensorflow.keras.layers import Layer, Dense, Embedding
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from tensorflow import random_normal_initializer as init


class Wide_layer(Layer):
    def __init__(self, feature_num):
        super().__init__()
        self.w0 = self.add_weight(name='w0', shape=(1,),
                                  initializer=tf.zeros_initializer(), trainable=True)
        self.w = self.add_weight(name='w', shape=(feature_num, 1),
                                 initializer=init(), trainable=True, regularizer=l2(1e-4))

    def call(self, inputs):
        x = tf.matmul(inputs, self.w) + self.w0
        return x


class Deep_layer(Layer):
    def __init__(self, hidden_units, output_dim, activation):
        super().__init__()
        self.hidden_layer = [Dense(i, activation=activation) for i in hidden_units]
        self.output_layer = Dense(output_dim, activation=None)

    def call(self, inputs):
        x = inputs
        for layer in self.hidden_layer:
            x = layer(x)
        output = self.output_layer(x)
        return output


class WideDeep(Model):
    def __init__(self, features, hidden_units, output_dim, activation):
        super().__init__()
        self.emb_layer = {}
        # 逐个类别特征初始化embedded层
        for i, feat in enumerate(features[1]):
            self.emb_layer['embed_layer' + str(i)] = Embedding(feat['one_hot_dim'], feat['emb_dim'])

        self.wide = Wide_layer(feature_num=39)
        self.deep = Deep_layer(hidden_units, output_dim, activation)

    def call(self, inputs):
        # dense_inputs: 数值特征，13维
        # sparse_inputs： 类别特征，26维
        # one_hot处理的类别特征(wide侧的输入)
        dense_inputs, sparse_inputs, one_hot_inputs = inputs[:, :13], inputs[:, 13:39], inputs[:, 39:]

        # wide部分
        wide_input = tf.concat([dense_inputs, one_hot_inputs], axis=-1)
        wide_output = self.wide(wide_input)

        # deep部分
        sparse_embed = tf.concat([self.emb_layer['embed_layer' + str(i)](sparse_inputs[:, i]) \
                                  for i in range(sparse_inputs.shape[-1])], axis=-1)
        deep_output = self.deep(sparse_embed)

        output = tf.nn.sigmoid(0.5*(wide_output + deep_output))
        return output