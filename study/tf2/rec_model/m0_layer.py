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


class Embed_Layer(Layer):
    def __init__(self, sparse_features):
        super().__init__()
        self.sparse_dim = len(sparse_features)
        self.emb_layers = {}
        # 逐个类别特征初始化embedded层
        for i, feat in enumerate(sparse_features):
            self.emb_layers['embed_layer' + str(i)] = Embedding(feat['one_hot_dim'], feat['emb_dim'])

    def call(self, sparse_inputs):
        # 将类别特征从one hot dim转换成embed_dim
        emb_layers = [self.emb_layers['embed_layer' + str(i)](sparse_inputs[:, i]) for i in range(self.sparse_dim)]
        sparse_embed = tf.concat(emb_layers, axis=-1)
        return sparse_embed


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

