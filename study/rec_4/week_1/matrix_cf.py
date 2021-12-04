# -*- 自动求导机制
import tensorflow as tf
import numpy as np
import tf_record
import params_server
# 基于矩阵分解的embedding模型

batch_size = 200
vector_dim = 16


def train():
    # 初始化向量
    vec_shape = [batch_size, vector_dim]
    user_id_emb = tf.random.normal(vec_shape)
    movie_id_emb = tf.random.normal(vec_shape)
    epochs = 30
    for epoch in range(epochs):
        print('Start of epoch %d' % (epoch,))
        keys = ['user_id', 'movie_id', 'rating']
        types = [tf.int64, tf.int64, tf.int64]
        # 分批读出每个特征
        data_set = tf_record.read('rating', keys, types, batch_size=batch_size)
        batch_num = 0
        for user_id, movie_id, label in data_set:
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
            # 梯度下降
            grads = tape.gradient(loss, [user_id_emb, movie_id_emb])
            user_id_emb -= grads[0] * 0.5
            movie_id_emb -= grads[1] * 0.5
            batch_num += 1

    print('result -> epoch = %d, batch_num = %d, loss = %f' % (epoch, batch_num, loss.numpy()))
    print('user_id_emb top 5', user_id_emb[0:5])
    print('movie_id_emb top 5', movie_id_emb[0:5])


if __name__ == "__main__":
    train()
