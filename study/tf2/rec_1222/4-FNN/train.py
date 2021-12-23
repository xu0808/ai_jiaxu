#!/usr/bin/env python
# coding: utf-8

import os
from model import FM, DNN
from utils import create_criteo_dataset

import tensorflow as tf
from tensorflow.keras import optimizers, losses, metrics
from sklearn.metrics import accuracy_score

data_dir = 'D:\\study\\ide\\ai_jiaxu\\study\\tf2\\rec_1222\\Data'
file_path = os.path.join(data_dir, 'train.txt')
k = 8

if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = create_criteo_dataset(file_path, test_size=0.5)
    model = FM(k)
    optimizer = optimizers.SGD(0.01)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.batch(32).prefetch(tf.data.experimental.AUTOTUNE)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_dataset, epochs=200)

    # 评估
    fm_pre = model(X_test)
    fm_pre = [1 if x > 0.5 else 0 for x in fm_pre]

    # **************** Statement 2 of Training *****************#
    # 获取FM训练得到的隐向量
    v = model.variables[2]  # [None, onehot_dim, k]

    X_train = tf.cast(tf.expand_dims(X_train, -1), tf.float32)  # [None, onehot_dim, 1]
    X_train = tf.reshape(tf.multiply(X_train, v), shape=(-1, v.shape[0] * v.shape[1]))  # [None, onehot_dim*k]

    hidden_units = [256, 128, 64]
    model = DNN(hidden_units, 1, 'relu')
    optimizer = optimizers.SGD(0.0001)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.batch(32).prefetch(tf.data.experimental.AUTOTUNE)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_dataset, epochs=50)

    # 评估
    X_test = tf.cast(tf.expand_dims(X_test, -1), tf.float32)
    X_test = tf.reshape(tf.multiply(X_test, v), shape=(-1, v.shape[0] * v.shape[1]))
    fnn_pre = model(X_test)
    fnn_pre = [1 if x > 0.5 else 0 for x in fnn_pre]

    print("FM Accuracy: ", accuracy_score(y_test, fm_pre))
    print("FNN Accuracy: ", accuracy_score(y_test, fnn_pre))
