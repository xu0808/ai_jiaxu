#!/usr/bin/env python
# coding: utf-8
# DeepFM
"""
Deep与FM结合的产物，也是Wide&Deep的改进版，只是将LR替换成了FM，提升了模型wide侧提取信息的能力。
优点：
1 两部分联合训练，无需加入人工特征，更易部署；
2 结构简单，复杂度低，两部分共享输入，共享信息，可更精确的训练学习。
缺点：
1 将类别特征对应的稠密向量拼接作为输入，然后对元素进行两两交叉。
这样导致模型无法意识到域的概念，FM与Deep两部分都不会考虑到域，属于同一个域的元素应该对应同样的计算。
"""

import tensorflow as tf
from tensorflow.keras import Model
from m0_layer import Embed_layer, Deep_layer
from m1_fm import FM_layer


class DeepFM(Model):
    def __init__(self, features, emb_dim, k, w_reg, v_reg, hidden_units, output_dim, activation):
        super().__init__()
        self.dense_features, self.sparse_features = features
        self.dense_dim = len(self.dense_features)
        self.sparse_dim = len(self.sparse_features)

        # 逐个类别特征初始化embedded层
        self.emb_layers = Embed_layer(self.sparse_features)

        self.fm = FM_layer(k, w_reg, v_reg, f_n=self.dense_dim + emb_dim*self.sparse_dim)
        self.deep = Deep_layer(hidden_units, output_dim, activation)

    def call(self, inputs):
        # 数值特征
        dense_inputs = inputs[:, :self.dense_dim]
        # 类别特征
        sparse_inputs = inputs[:, self.dense_dim:]

        # 将类别特征从one hot dim转换成embed_dim
        sparse_embed = self.emb_layers(sparse_inputs)
        # DeepFM两个部分共享输入(数值特征 + 类别特征embedding)
        x = tf.concat([dense_inputs, sparse_embed], axis=-1)

        # fm输出
        fm_output = self.fm(x)
        # deep输出
        deep_output = self.deep(x)
        # 模型整体输出
        output = tf.nn.sigmoid(0.5 * (fm_output + deep_output))
        return output
