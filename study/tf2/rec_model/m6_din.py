#!/usr/bin/env python
# coding: utf-8
# xDeepFM
"""
优点：
1 引入Attention机制，更精准的提取用户兴趣；
2 引入Dice激活函数与，并优化了稀疏场景中的L2正则方式。
缺点：
没有考虑用户点击商品的相对位置信息，后续的DIEN也是针对这点进行了改进。
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, BatchNormalization, PReLU, Dropout
from m0_layer import Embed_layer, Attention_layer, Dice_layer, Deep_layer


class DIN(Model):
    def __init__(self, features, behavior_features, att_hidden_units, ffn_hidden_units,
                 att_activation='prelu', ffn_activation='prelu', maxlen=40, dropout=0.0):
        super(DIN, self).__init__()
        self.dense_features, self.sparse_features = features
        self.dense_dim = len(self.dense_features)
        self.sparse_dim = len(self.sparse_features)
        self.behavior_num = len(behavior_features)
        self.maxlen = maxlen
        self.other_sparse_num = self.sparse_dim - self.behavior_num

        # other sparse embedding
        self.embed_sparse_layers = Embed_layer(self.sparse_features - behavior_features)
        # behavior embedding layers, item id and category id
        self.embed_seq_layers = Embed_layer(behavior_features)

        self.att_layer = Attention_layer(att_hidden_units, att_activation)
        self.bn_layer = BatchNormalization(trainable=True)
        activation = PReLU() if ffn_activation == 'prelu' else Dice_layer()
        self.ffn_deep_layers = Deep_layer(ffn_hidden_units, 1, activation, dropout)

    def call(self, inputs, training=None):
        # dense_inputs:  empty/(None, dense_num)
        # sparse_inputs: empty/(None, other_sparse_num)
        # history_seq:  (None, n, k)
        # candidate_item: (None, k)
        dense_inputs, sparse_inputs, history_seq, candidate_item = inputs

        # dense & sparse inputs embedding
        sparse_embed = self.embed_sparse_layers(sparse_inputs)
        other_feat = tf.concat([sparse_embed, dense_inputs], axis=-1)

        # history_seq & candidate_item embedding
        # (None, n, k)
        seq_embed = self.embed_seq_layers(history_seq)   # ([layer(history_seq[:, :, i])
        # (None, k)
        item_embed = self.embed_seq_layers(candidate_item)

        # one_hot之后第一维是1的token，为填充的0
        # (None, n)
        mask = tf.cast(tf.not_equal(history_seq[:, :, 0], 0), dtype=tf.float32)
        # (None, k)
        att_emb = self.attention_layer([item_embed, seq_embed, seq_embed, mask])

        # 若其他特征不为empty
        if self.dense_len > 0 or self.other_sparse_len > 0:
            emb = tf.concat([att_emb, item_embed, other_feat], axis=-1)
        else:
            emb = tf.concat([att_emb, item_embed], axis=-1)

        emb = self.bn_layer(emb)
        for layer in self.dense_layer:
            emb = layer(emb)

        emb = self.dropout(emb)
        output = self.out_layer(emb)
        # (None, 1)
        return tf.nn.sigmoid(output)
