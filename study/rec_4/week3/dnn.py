#!/usr/bin/env python
# coding: utf-8
# DNN基础

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tf_record
import cache_server

batch_size = 200
epochs = 200

feature_num = 4
emb_dim = 4
learning_rate = 0.2
active = tf.nn.relu

keys = ['user_id', 'article_id', 'environment', 'region', 'label']
types = [tf.int64, tf.int64, tf.int64, tf.int64, tf.float32]
# 分批读出每个特征
data_set = tf_record.read('click_log', keys, types, batch_size=batch_size)

config = {
    "train_file": "../data2/train",
    "test_file": "../data2/val",
    "saved_embedding": "../data4/saved_dnn_embedding",
    "max_steps": 10000,
    "train_log_iter": 1000,
    "test_show_step": 1000,
    "last_test_auc": 0.5,

    "saved_checkpoint": "checkpoint",
    "checkpoint_name": "dnn",

    "saved_pb": "../data2/saved_model",

    "input_tensor": ["input_tensor"],
    "output_tensor": ["output_tensor"]
}


class DNN_Model(tf.keras.models.Model):

    def __init__(self):
        super(DNN_Model, self).__init__()
        self.d1 = tf.keras.layers.Dense(32, use_bias=False, activation=active)
        self.d2 = tf.keras.layers.Dense(16, use_bias=False, activation=active )
        self.d3 = tf.keras.layers.Dense(1, use_bias=False, activation=None)

    def call(self, x):
        x_1 = self.d1(x)
        x_2 = self.d2(x_1)
        x_3 = self.d3(x_2)
        out = tf.sigmoid(x_3)
        return out


def train():
    ps = cache_server.CacheServer(vector_dim=emb_dim)
    model = DNN_Model()
    # 1、模型训练
    for epoch in range(epochs):
        print('Start of epoch %d' % epoch)
        batch_num = 0
        # user_id, article_id, environment, region
        for f_1, f_2, f_3, f_4, label in data_set:
            # 初始化和读取向量
            features = [f_1, f_2, f_3, f_4]
            embs = [tf.constant(ps.pull(f.numpy())) for f in features]
            y_true = tf.reshape(label, [200, 1])
            with tf.GradientTape() as tape:

                # 【所有参数计算必须位于tape.watch后】
                tape.watch(embs)
                # 所有特征向量拼接
                inputs = tf.concat(embs, 1)
                logits = model(inputs)
                loss = tf.reduce_mean(
                  tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y_true))
            # 损失计算
            if batch_num % 100 == 0:
                print('epoch = %d, batch_num = %d, loss = %f' % (epoch, batch_num, loss.numpy()))
            # 最小损失推出
            if loss < 0.000001:
                break
                # 最小损失推出
            # if batch_num > 2:
            #     break
            # 梯度下降
            grads = tape.gradient(loss, embs)
            for i in range(4):
                # 更新向量
                embs[i] -= grads[i] * learning_rate
                ps.push(features[i].numpy(), embs[i].numpy())
            batch_num += 1

    print('result -> epoch = %d, batch_num = %d, loss = %f' % (epoch, batch_num, loss.numpy()))
    print('user_id_emb top 5', features[0][0:5])


if __name__ == "__main__":
    train()
