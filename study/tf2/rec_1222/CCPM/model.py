'''
# Time   : 2021/1/7 17:21
# Author : junchaoli
# File   : model.py
'''

from layer import DNN, CCPM_layer

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding

class CCPM(Model):
    def __init__(self, feature_columns, hidden_units, out_dim=1, activation='relu', dropout=0.0,
                 filters=[4, 4], kernel_width=[6, 5]):
        super(CCPM, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.emb_layers = [Embedding(feat['feat_onehot_dim'], feat['embed_dim'])
                                for i,feat in enumerate(self.sparse_feature_columns)]
        self.dnn_layer = DNN(hidden_units, out_dim, activation, dropout)
        self.ccpm_layer = CCPM_layer(filters, kernel_width)

    def call(self, inputs, training=None, mask=None):
        # dense_inputs:  [None, 13]
        # sparse_inputs: [None, 26]
        dense_inputs, sparse_inputs = inputs[:, :13], inputs[:, 13:]
        sparse_embed = [layer(sparse_inputs[:, i]) for i, layer in enumerate(self.emb_layers)] # 26 * [None, k]
        sparse_embed = tf.transpose(tf.convert_to_tensor(sparse_embed), [1, 0, 2])             # [None, 26, k]

        ccpm_out = self.ccpm_layer(sparse_embed)   # [None, new_field*k]
        x = tf.concat([dense_inputs, ccpm_out], axis=-1)
        output = self.dnn_layer(x)
        return output

