#!/usr/bin/env python
# coding: utf-8
# 基于矩阵分解的embedding模型

import tensorflow as tf
import pandas as pd
import os
import tf_record
import reader
import cache_server

# 为了保证对齐，需要全量训练
batch_size = 200
vector_dim = 16
learning_rate = 0.5
epochs = 1000


def train():
    ps = cache_server.CacheServer(vector_dim=vector_dim)
    # 1、模型训练
    for epoch in range(epochs):
        print('Start of epoch %d' % (epoch,))
        keys = ['user_id', 'movie_id', 'rating']
        types = [tf.int64, tf.int64, tf.int64]
        # 分批读出每个特征
        data_set = tf_record.read('rating', keys, types, batch_size=batch_size)
        batch_num = 0
        for user_id, movie_id, label in data_set:
            # 初始化和读取向量
            user_id_emb = tf.constant(ps.pull(user_id.numpy()))
            movie_id_emb = tf.constant(ps.pull(movie_id.numpy()))
            y_true = tf.constant(label, dtype=tf.int64)
            with tf.GradientTape() as tape:
                tape.watch([user_id_emb, movie_id_emb])
                y_pre = tf.reduce_mean(user_id_emb * movie_id_emb, axis=1)
                loss = tf.reduce_mean(tf.square(y_pre, y_true))

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
            grads = tape.gradient(loss, [user_id_emb, movie_id_emb])
            user_id_emb -= grads[0] * learning_rate
            movie_id_emb -= grads[1] * learning_rate
            # 更新向量
            ps.push(user_id.numpy(), user_id_emb.numpy())
            ps.push(movie_id.numpy(), movie_id_emb.numpy())
            batch_num += 1

    print('result -> epoch = %d, batch_num = %d, loss = %f' % (epoch, batch_num, loss.numpy()))
    print('user_id_emb top 5', user_id_emb[0:5])
    print('movie_id_emb top 5', movie_id_emb[0:5])
    # 2、向量存取
    keys, values = [], []
    for key in ps.cache:
        keys.append(key)
        values.append(ps.cache.get(key))

    emb_df = pd.DataFrame({'key': keys, 'vec': values})
    # 数据文件
    emb_file = os.path.join(reader.data_dir, 'rating_emb.csv')
    emb_df.to_csv(emb_file, index=False, sep=',')


if __name__ == "__main__":
    train()
