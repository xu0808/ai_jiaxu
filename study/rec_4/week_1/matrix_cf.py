#!/usr/bin/env python
# coding: utf-8
# 模型
import tensorflow as tf

'''=================================================
@Function -> 用TensorFlow2实现协同过滤矩阵的分解
@Author ：luoji
@Date   ：2021-10-19
=================================================='''

import numpy as np
import tensorflow as tf


def matrix_cf(alike_matix, rank=10, num_epoch=5000, learning_rate=0.001, reg=0.5):
    row, column = len(alike_matix), len(alike_matix[0])
    # 构建y_true
    y_true = tf.constant(alike_matix, dtype=tf.float32)
    # user权重矩阵
    u_init = np.random.random(size=(row, rank))
    u = tf.Variable(shape=(row, rank), initial_value=u_init, dtype=tf.float32)
    # 代表权重矩阵，初始化为0
    v_init = np.random.random(size=(rank, column))
    v = tf.Variable(shape=(rank, column), initial_value=v_init, dtype=tf.float32)

    variables = [u, v]
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    for batch_index in range(num_epoch):
        with tf.GradientTape() as tape:
            y_pre = tf.matmul(u, v)
            # 正则化项
            reg_value = tf.norm(u, ord='euclidean') + tf.norm(v, ord='euclidean')
            loss = tf.reduce_sum(tf.norm(y_true - y_pre, ord='euclidean') + reg * reg_value)
            print("batch %d : loss %f" % (batch_index, loss.numpy()))

        grads = tape.gradient(loss, variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, variables))
    return u, v, tf.matmul(u, v)


if __name__ == "__main__":
    # 把矩阵分解为 M=U*V ，U和V由用户指定秩rank
    alike_matrix = [[1.0, 2.0, 3.0],
                    [4.5, 5.0, 3.1],
                    [1.0, 2.0, 3.0],
                    [4.5, 5.0, 5.1],
                    [1.0, 2.0, 3.0]]
    # reg 减小则num_epoch需增大
    U, V, preMatrix = matrix_cf(alike_matrix, rank=2, reg=0.5, num_epoch=20000)

    print(U)
    print(V)
    print(alike_matrix)
    print(preMatrix)
    print("this difference between alike_matrix and preMatrix is :")
    print(alike_matrix - preMatrix)
    print('loss is :', sum(sum(abs(alike_matrix - preMatrix))))
3
