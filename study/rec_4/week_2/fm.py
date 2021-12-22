#!/usr/bin/env python
# coding: utf-8
# FM特征交叉
# 自定义参数实现方式
# 可以对比TensorFlow2.0 实现FM
# https://blog.csdn.net/qq_34333481/article/details/103919923

import tensorflow as tf
import pandas as pd
import os
import tf_record
import reader_2
import cache_server

# 为了保证对齐，需要全量训练
batch_size = 200
epochs = 200

weight_dim = 17  # 16 + 1
learning_rate = 0.05
feature_dim = 4

keys = ['user_id', 'article_id', 'environment', 'region', 'label']
types = [tf.int64, tf.int64, tf.int64, tf.int64, tf.float32]
# 分批读出每个特征
data_set = tf_record.read('click_log', keys, types, batch_size=batch_size)


def fm_fn(inputs):
    """FM的模型方程：LR线性组合 + 交叉项组合 = 1阶特征组合 + 2阶特征组合"""

    weight = tf.reshape(inputs, shape=[-1, feature_dim, weight_dim])

    # batch * 4 * 16, batch * 4 * 1
    cross_weight, linear_weight = tf.split(weight, num_or_size_splits=[weight_dim - 1, 1], axis=2)

    bias = tf.compat.v1.get_variable("bias", [1, ], initializer=tf.zeros_initializer())
    linear_model = tf.nn.bias_add(tf.reduce_sum(linear_weight, axis=1), bias)

    square_sum = tf.square(tf.reduce_sum(cross_weight, axis=1))  #
    summed_square = tf.reduce_sum(tf.square(cross_weight), axis=1)

    cross_model = 0.5 * tf.reduce_sum(tf.subtract(square_sum, summed_square), axis=1, keepdims=True)
    y_pred = cross_model + linear_model
    return y_pred[:, 0]


def train():
    ps = cache_server.CacheServer(vector_dim=weight_dim)
    # 1、模型训练
    for epoch in range(epochs):
        print('Start of epoch %d' % (epoch,))
        batch_num = 0
        # user_id, article_id, environment, region
        for f_1, f_2, f_3, f_4, label in data_set:
            # 初始化和读取向量
            features = [f_1, f_2, f_3, f_4]
            embs = [tf.constant(ps.pull(f.numpy())) for f in features]
            y_true = tf.constant(label, dtype=tf.float32)
            with tf.GradientTape() as tape:

                # 【普通变量，非普通参数，所有参数计算必须位于tape.watch后】
                tape.watch(embs)
                # 所有特征向量拼接
                inputs = tf.concat(embs, 1)
                y_pred = fm_fn(inputs)
                loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred, labels=y_true))
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
    # 2、向量存取
    keys, values = [], []
    for key in ps.cache:
        keys.append(key)
        values.append(ps.cache.get(key))

    emb_df = pd.DataFrame({'key': keys, 'vec': values})
    # 数据文件
    emb_file = os.path.join(reader_2.data_dir, 'fm_emb.csv')
    emb_df.to_csv(emb_file, index=False, sep=',')


if __name__ == "__main__":
    train()
