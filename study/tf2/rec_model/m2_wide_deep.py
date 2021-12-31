#!/usr/bin/env python
# coding: utf-8
# Wide&Deep
"""
优点:
1 结构简单，复杂度低，目前在工业界仍有广泛应用；
2 线性模型与深度模型优势互补，分别提取低阶与高阶特征交互信息，兼顾记忆能力与泛化能力；
3 线性部分为广义线性模型，可灵活替换为其他算法，比如 FM，提升 wide 部分提取信息的能力。
缺点：
1 深度模型可自适应的进行高阶特征交互，但这是隐式的构造特征组合，可解释性差；
2 深度模型仍需要人工特征来提升模型效果，只是需求量没有线性模型大。
"""

import tensorflow as tf
from tensorflow.keras import Model
from m0_layer import Line_layer, Embed_layer, Deep_layer


class WideDeep(Model):
    def __init__(self, wide_feature_num, features, hidden_units, output_dim, activation):
        super().__init__()
        self.dense_features, self.sparse_features = features
        self.dense_dim = len(self.dense_features)
        self.sparse_dim = len(self.sparse_features)

        self.wide = Line_layer(feature_num=wide_feature_num)

        # 逐个类别特征初始化embedded层
        self.emb_layers = Embed_layer(self.sparse_features)
        self.deep = Deep_layer(hidden_units, output_dim, activation)

    def call(self, inputs):
        # 数值特征
        dense_inputs = inputs[:, :self.dense_dim]
        # 类别特征
        sparse_inputs = inputs[:, self.dense_dim:self.dense_dim + self.sparse_dim]
        # one_hot处理的类别特征(wide侧的输入)
        one_hot_inputs = inputs[:, self.dense_dim + self.sparse_dim:]

        # wide部分
        wide_input = tf.concat([dense_inputs, one_hot_inputs], axis=-1)
        wide_output = self.wide(wide_input)

        # deep部分
        # 将类别特征从one hot dim转换成embed_dim
        sparse_embed = self.emb_layers(sparse_inputs)
        deep_output = self.deep(sparse_embed)

        # 模型整体输出
        output = tf.nn.sigmoid(0.5*(wide_output + deep_output))
        return output
