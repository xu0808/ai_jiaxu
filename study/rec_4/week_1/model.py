# -*- 自动求导机制
import tensorflow as tf
import numpy as np
import tf_record
import params_server

batch_size = 12875
vector_dim = 16


def matrix_cf():
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
        for user_id, movie_id, label in data_set:
            y_true = tf.constant(label, dtype=tf.int64)
            with tf.GradientTape() as tape:
                tape.watch([user_id_emb, movie_id_emb])
                y_pre = tf.reduce_mean(user_id_emb * movie_id_emb, axis=1)
                loss = tf.reduce_mean(tf.square(y_pre, y_true))
            grads = tape.gradient(loss, [user_id_emb, movie_id_emb])
            user_id_emb -= grads[0] * 0.05
            movie_id_emb -= grads[1] * 0.05
            print('loss = ', loss.numpy())


if __name__ == "__main__":
    matrix_cf()
