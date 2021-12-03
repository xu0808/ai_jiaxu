# -*- 自动求导机制
import tensorflow as tf
import numpy as np
import tf_record
import params_server

batch_size = 12875
vector_dim = 16
# 梯度下降
num_epoch = 10000 # 迭代轮数
# 优化器
optimizer = tf.keras.optimizers.SGD(learning_rate=5e-4)

def matrix_cf():
    vec_shape = [batch_size, vector_dim]
    user_id_emb = tf.random.normal(vec_shape)
    movie_id_emb = tf.random.normal(vec_shape)
    vars = [user_id_emb, movie_id_emb]
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
                y_pre = tf.reduce_mean(user_id_emb * movie_id_emb, axis=1)
                loss = tf.losses.mean_squared_error(y_pre, y_true)
            grads = tape.gradient(loss, vars)
            # user_id_emb -= grads[0] * 0.05
            # movie_id_emb -= grads[1] * 0.05
            # 根据梯度 更新参数
            # optimizer.apply_gradients(grads_and_vars=zip(grads, vars))
            print('loss = ', loss.numpy())


if __name__ == "__main__":
    matrix_cf()
