#!/usr/bin/env python
# coding: utf-8
# DeepFM
"""
优点：
1 两部分联合训练，无需加入人工特征，更易部署；
2 结构简单，复杂度低，两部分共享输入，共享信息，可更精确的训练学习。
缺点：
1 将类别特征对应的稠密向量拼接作为输入，然后对元素进行两两交叉。
这样导致模型无法意识到域的概念，FM与Deep两部分都不会考虑到域，属于同一个域的元素应该对应同样的计算。
"""

import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Embedding
from tensorflow.keras import Model
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
        linear_part = tf.matmul(inputs, self.w) + self.w0
        # 交叉部分 shape:(batch_size, self.k)
        # 交叉部分——第一项
        inter_1 = tf.pow(tf.matmul(inputs, self.v), 2)
        # 交叉部分——第二项
        inter_2 = tf.matmul(tf.pow(inputs, 2), tf.pow(self.v, 2))
        # 交叉结果
        inter_part = 0.5 * tf.reduce_sum(inter_1 - inter_2, axis=-1, keepdims=True)
        # 模型输出
        return tf.nn.sigmoid(linear_part + inter_part)


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


class DeepFM(Model):
    def __init__(self, features, emb_dim, k, w_reg, v_reg, hidden_units, output_dim, activation):
        super().__init__()
        self.dense_features, self.sparse_features = features
        self.dense_dim = len(self.dense_features)
        self.sparse_dim = len(self.sparse_features)
        self.emb_layers = {}
        # 逐个类别特征初始化embedded层
        for i, feat in enumerate(self.sparse_features):
            self.emb_layers['embed_layer' + str(i)] = Embedding(feat['one_hot_dim'], feat['emb_dim'])

        self.fm = FM_layer(k, w_reg, v_reg, f_n=self.dense_dim + emb_dim*self.sparse_dim)
        self.deep = Deep_layer(hidden_units, output_dim, activation)

    def call(self, inputs):
        # 数值特征
        dense_inputs = inputs[:, :self.dense_dim]
        # 类别特征
        sparse_inputs = inputs[:, self.dense_dim:]

        # 将类别特征从one hot dim转换成embed_dim
        emb_layers = [self.emb_layers['embed_layer' + str(i)](sparse_inputs[:, i]) for i in range(self.sparse_dim)]
        sparse_embed = tf.concat(emb_layers, axis=-1)
        # DeepFM两个部分共享输入(数值特征 + 类别特征embedding)
        x = tf.concat([dense_inputs, sparse_embed], axis=-1)

        # fm输出
        fm_output = self.fm(x)
        # deep输出
        deep_output = self.deep(x)
        # 模型整体输出
        output = tf.nn.sigmoid(0.5 * (fm_output + deep_output))
        return output
