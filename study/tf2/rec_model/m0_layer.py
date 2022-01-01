#!/usr/bin/env python
# coding: utf-8
# 通用layer

import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Embedding, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2


class Line_layer(Layer):
    """线性层"""
    def __init__(self, feature_num):
        super().__init__()
        self.w = self.add_weight(name='w', shape=(feature_num, 1), initializer=tf.random_normal_initializer(),
                                 regularizer=l2(1e-4), trainable=True)

        self.b = self.add_weight(name='b', shape=(1,), initializer=tf.zeros_initializer(),
                                 regularizer=l2(1e-4), trainable=True)

    def call(self, inputs):
        x = tf.matmul(inputs, self.w) + self.b
        return x


class Embed_layer(Layer):
    """embedding层"""
    def __init__(self, sparse_features):
        super().__init__()
        self.sparse_dim = len(sparse_features)
        self.emb_layers = {}
        # 逐个类别特征初始化embedded层
        for i, feat in enumerate(sparse_features):
            self.emb_layers['embed_layer' + str(i)] = Embedding(feat['one_hot_dim'], feat['emb_dim'])

    def call(self, sparse_inputs):
        # 将类别特征从one hot dim转换成embed_dim
        embs = [self.emb_layers['embed_layer' + str(i)](sparse_inputs[:, i]) for i in range(self.sparse_dim)]
        # tf.print('embs[0].shape = ', embs[0].shape)
        sparse_embed = tf.concat(embs, axis=-1)
        return sparse_embed


class Deep_layer(Layer):
    """深度神经网络层"""
    def __init__(self, hidden_units, output_dim, activation, dropout=0.0):
        super().__init__()
        self.hidden_layer = [Dense(i, activation=activation) for i in hidden_units]
        # 输出层没有激活函数
        self.output_layer = Dense(output_dim, activation=None)
        self.dropout = Dropout(dropout)

    def call(self, inputs):
        x = inputs
        for layer in self.hidden_layer:
            x = layer(x)
        x = self.dropout(x)
        output = self.output_layer(x)
        return output


class Res_layer(Layer):
    """残差层(深度神经网络层输出 + 输入)"""
    def __init__(self, hidden_units, output_dim, activation):
        super(Res_layer, self).__init__()
        self.deep_layers = Deep_layer(hidden_units, output_dim, activation)

    def call(self, inputs):
        # deep输出
        deep_output = self.deep_layers(inputs)
        return tf.nn.relu(inputs + deep_output)


class Attention_layer(Layer):
    def __init__(self, hidden_units, activation='prelu'):
        super(Attention_layer, self).__init__()
        self.deep_layers = Deep_layer(hidden_units, 1, activation)

    def call(self, inputs, **kwargs):
        # query: [None, k]
        # key:   [None, n, k]
        # value: [None, n, k]
        # mask:  [None, n]
        query, key, value, mask = inputs
        # [None, 1, k]
        query = tf.expand_dims(query, axis=1)
        # [None, n, k]
        query = tf.tile(query, [1, key.shape[1], 1])
        # [None, n, 4*k]
        emb = tf.concat([query, key, query - key, query * key], axis=-1)
        # [None, n, 1]
        deep_output = self.deep_layers(emb)
        # [None, n]
        score = tf.squeeze(deep_output, axis=-1)
        # [None, n]
        padding = tf.ones_like(score) * (-2**32 + 1)
        # [None, n]
        score = tf.where(tf.equal(mask, 0), padding, score)
        score = tf.nn.softmax(score)
        # [None, 1, k]
        output = tf.matmul(tf.expand_dims(score, axis=1), value)
        # [None, k]
        output = tf.squeeze(output, axis=1)
        return output


class Dice_layer(Layer):
    def __init__(self):
        super(Dice_layer, self).__init__()
        self.bn_layer = BatchNormalization()
        self.alpha = self.add_weight(name='alpha', shape=(1,), trainable=True)

    def call(self, inputs):
        x = self.bn_layer(inputs)
        x = tf.nn.sigmoid(x)
        output = x * inputs + (1-x) * self.alpha * inputs
        return output
