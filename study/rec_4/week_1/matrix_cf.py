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


if __name__ == "__main__":
    row, rank, column = 3, 2, 4
    u_0 = np.random.random(size=(row, rank))
    v_0 = np.random.random(size=(rank, column))
    matrix = np.dot(u_0, v_0)
    # 1、np svd
    # np_svd(matrix)
    # 2、tf svd
    tf_svd(matrix)
