#!/usr/bin/env python
# coding: utf-8
# DeepCrossing
"""
简称DCN，是基于Wide&Deep的改进版，把wide侧的LR换成了cross layer，
可显式的构造有限阶特征组合，并且具有较低的复杂度。
优点：
1 引入cross layer显示的构造有限阶特征组合，无需特征工程，可端到端训练；
2 cross layer具有线性复杂度，可累加多层构造高阶特征交互，
  因为类似残差连接的计算方式，累加多层也不会产生梯度消失问题；
3 跟deepfm相同，两个分支共享输入，可更精确的训练学习。
缺点：
1. DCN不会考虑域的概念，属于同一特征的各个元素应同等对待；
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from m0_layer import Embed_layer, Res_layer


class DeepCrossing(Model):
    def __init__(self, features, emb_dim, hidden_units, res_layer_num, activation):
        super().__init__()
        self.dense_features, self.sparse_features = features
        self.dense_dim = len(self.dense_features)
        self.sparse_dim = len(self.sparse_features)
        f_n = self.dense_dim + emb_dim * self.sparse_dim

        # emb层
        self.emb_layers = Embed_layer(self.sparse_features)
        # 残差层
        self.res_layers = [Res_layer(hidden_units, f_n, activation) for _ in range(res_layer_num)]
        # 最终输出
        self.output_layer = Dense(1, activation=activation)

    def call(self, inputs, training=None, mask=None):
        # 数值特征
        dense_inputs = inputs[:, :self.dense_dim]
        # 类别特征
        sparse_inputs = inputs[:, self.dense_dim:]

        # 将类别特征从one hot dim转换成embed_dim
        sparse_embed = self.emb_layers(sparse_inputs)
        # 实际输入(数值特征 + 类别特征embedding)
        x = tf.concat([dense_inputs, sparse_embed], axis=-1)

        # 残差层输出层
        for res_layer in self.res_layers:
            x = res_layer(x)
        # 最终输出
        output = self.output_layer(x)
        return output

