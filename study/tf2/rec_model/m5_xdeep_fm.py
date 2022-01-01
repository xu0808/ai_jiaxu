#!/usr/bin/env python
# coding: utf-8
# xDeepFM
"""
xDeepFM是Wide&Deep的改进版，在此基础上添加了CIN层显式的构造有限阶特征组合。
xDeepFM虽然名字跟DeepFM类似，但是两者相关性不大，DCN才是它的近亲。

优点：
使用vector-wise的方式，通过特征的元素积来进行特征交互，将一个特征域的元素整体考虑，
比bit-wise方式更make sence一些；
缺点：
CIN层的复杂度通常比较大，它并不具有像DCN的cross layer那样线性复杂度，
它的复杂度通常是平方级的，因为需要计算两个特征矩阵中特征的两两交互，这就给模型上线带来压力。
为什么CIN叫压缩感知层？因为每次矩阵W都会将特征两两交互得到的三维矩阵压缩成一维，所以叫做压缩感知。
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Dense
from tensorflow.keras.regularizers import l2

from m0_layer import Line_layer, Embed_layer, Deep_layer


class CIN(Layer):
    def __init__(self, emb_dim, cin_sizes, sparse_dim):
        super(CIN, self).__init__()
        # 向量维度
        self.emb_dim = emb_dim
        # 每层的矩阵个数
        self.cin_sizes = cin_sizes
        # 每层的矩阵个数(包括第0层)
        self.field_nums = [sparse_dim] + self.cin_sizes

        w_shapes = [(1, sparse_dim*self.field_nums[i], self.field_nums[i+1]) for i in range(len(self.cin_sizes))]
        self.cin_w_s = [self.add_weight(shape=shape, initializer=tf.initializers.glorot_uniform(),
                                        regularizer=l2(1e-5), trainable=True) for shape in w_shapes]

    def call(self, inputs):
        # inputs shape: [-1, sparse_dim, emb_dim]
        res_list = [inputs]
        # 最后维切成k份，list: emb_dim * [-1, sparse_dim, 1]
        x_0 = tf.split(inputs, self.emb_dim, axis=-1)
        for i, size in enumerate(self.field_nums[1:]):
            # list: emb_dim * [-1, field_nums[i], 1]
            x_i = tf.split(res_list[-1], self.emb_dim, axis=-1)
            # list: k * [-1, field_num[0], field_num[i]]
            x = tf.matmul(x_0, x_i, transpose_b=True)
            # [emb_dim, -1, sparse_dim*field_nums[i]]
            x = tf.reshape(x, shape=[self.emb_dim, -1, self.field_nums[0]*self.field_nums[i]])
            # [ -1, emb_dim,sparse_dim*field_nums[i]]
            x = tf.transpose(x, [1, 0, 2])
            # [ emb_dim, -1, field_nums[i+1]]
            x = tf.nn.conv1d(input=x, filters=self.cin_w_s[i], stride=1, padding='VALID')
            # [ -1, field_nums[i+1], emb_dim]
            x = tf.transpose(x, [0, 2, 1])
            res_list.append(x)

        # 去掉x_0
        res_list = res_list[1:]
        # [ -1, sum(field_nums), emb_dim]
        res = tf.concat(res_list, axis=1)
        # [ -1, sum(field_nums)]
        output = tf.reduce_sum(res, axis=-1)
        return output


class xDeepFM(Model):
    def __init__(self, features, emb_dim, cin_sizes, hidden_units, out_dim=1, activation='relu', dropout=0.0):
        super(xDeepFM, self).__init__()
        self.dense_features, self.sparse_features = features
        self.dense_dim = len(self.dense_features)
        self.sparse_dim = len(self.sparse_features)
        self.emb_dim = emb_dim
        # 线性层
        self.linear = Line_layer(feature_num=self.dense_dim + self.sparse_dim)
        # emb层
        self.emb_layers = Embed_layer(self.sparse_features)
        # dnn层
        self.deep_layer = Deep_layer(hidden_units, out_dim, activation)
        # cin层
        self.cin_layer = CIN(emb_dim, cin_sizes, self.sparse_dim)
        self.out_layer = Dense(1, activation=None)

    def call(self, inputs, training=None, mask=None):
        # 数值特征
        dense_inputs = inputs[:, :self.dense_dim]
        # 类别特征
        sparse_inputs = inputs[:, self.dense_dim:]

        # 将类别特征从one hot dim转换成embed_dim
        # shape: [-1, sparse_dim * emb_dim]
        sparse_embed = self.emb_layers(sparse_inputs)
        # shape: [-1, sparse_dim, emb_dim]
        emb = tf.reshape(sparse_embed, [-1, self.sparse_dim, self.emb_dim])

        # linear输出
        linear_out = self.linear(inputs)
        # CIN输出
        cin_out = self.cin_layer(emb)
        # dnn输出
        # 实际输入(数值特征 + 类别特征embedding)
        x = tf.concat([dense_inputs, sparse_embed], axis=-1)
        deep_out = self.deep_layer(x)

        # 最终输出
        output = self.out_layer(linear_out + cin_out + deep_out)
        return tf.nn.sigmoid(output)
