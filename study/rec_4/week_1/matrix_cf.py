#!/usr/bin/env python
# coding: utf-8
# 基于矩阵分解协同过滤

import numpy as np
import tensorflow as tf


def np_svd(mat):
    u, s, v = np.linalg.svd(mat)
    print('u = ', u)
    print('s = ', s)
    print('v = ', v)
    s_1 = np.column_stack((np.diag(s), np.zeros(s.shape[0])))
    mat_1 = u.dot(s_1.dot(v))
    print('mat = ', mat)
    print('mat_1 = ', mat_1)


def tf_svd(mat):
    mat_tf = tf.constant(mat, dtype=tf.float32)
    s, u, v = tf.linalg.svd(mat_tf)
    print('u = ', u.numpy())
    print('s = ', s.numpy())
    print('v = ', v.numpy())
    s_1 = np.column_stack((np.diag(s.numpy()), np.zeros(s.numpy().shape[0])))
    mat_1 = u.numpy().dot(s_1.dot(v))
    print('mat = ', mat)
    print('mat_1 = ', mat_1)

# if __name__ == "__main__":
#     row, rank, column = 3, 2, 4
#     u_0 = np.random.random(size=(row, rank))
#     v_0 = np.random.random(size=(rank, column))
#     matrix = np.dot(u_0, v_0)
#     # 1、np svd
#     # np_svd(matrix)
#     # 2、tf svd
#     # tf_svd(matrix)


def matrix_svd(alike_matix, rank=10, num_epoch=5000, learning_rate=0.001, reg=0.5):
    row, column = len(alike_matix), len(alike_matix[0])
    y_true = tf.constant(alike_matix, dtype=tf.float32)  # 构建y_true
    U = tf.Variable(shape=(row, rank), initial_value=np.random.random(size=(row, rank)),
                    dtype=tf.float32)  # 构建一个变量U，代表user权重矩阵
    V = tf.Variable(shape=(rank, column), initial_value=np.random.random(size=(rank, column)),
                    dtype=tf.float32)  # 构建一个变量，代表权重矩阵，初始化为0

    variables = [U, V]
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    for batch_index in range(num_epoch):
        with tf.GradientTape() as tape:
            y_pre = tf.matmul(U, V)
            loss = tf.reduce_sum(tf.norm(y_true - y_pre, ord='euclidean')
                                 + reg * (tf.norm(U, ord='euclidean') + tf.norm(V, ord='euclidean')))  # 正则化项
            print("batch %d : loss %f" % (batch_index, loss.numpy()))

        grads = tape.gradient(loss, variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, variables))
    return U, V, tf.matmul(U, V)


if __name__ == "__main__":
    # 把矩阵分解为 M=U*V ，U和V由用户指定秩rank
    alike_matrix = [[1.0, 2.0, 3.0],
                    [4.5, 5.0, 3.1],
                    [1.0, 2.0, 3.0],
                    [4.5, 5.0, 5.1],
                    [1.0, 2.0, 3.0]]
    U, V, preMatrix = matrix_svd(alike_matrix, rank=2, reg=0.5, num_epoch=2000)  # reg 减小则num_epoch需增大

    print(U)
    print(V)
    print(alike_matrix)
    print(preMatrix)
    print("this difference between alike_matrix and preMatrix is :")
    print(alike_matrix - preMatrix)
    print('loss is :', sum(sum(abs(alike_matrix - preMatrix))))
