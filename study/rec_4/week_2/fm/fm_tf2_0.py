#!/usr/bin/env python
# coding: utf-8
# tf2.0实现FM

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Input, Dense, Add, Activation
from tensorflow.keras.regularizers import l1, l2

# keras通用后端函数
K = tf.keras.backend

# 数据文件
data_dir = 'D:\\study\\rec_4\\data\\2\\fm'
data_file = os.path.join(data_dir, 'diabetes.csv')
train_file = os.path.join(data_dir, 'diabetes_train.txt')
test_file = os.path.join(data_dir, 'diabetes_test.txt')


# 自定义FM的二阶交叉层
class FMCrossLayer(Layer):
    # FM的k取4（演示方便）
    def __init__(self, input_dim, output_dim=4):
        self.input_dim = input_dim
        self.output_dim = output_dim
        super(FMCrossLayer, self).__init__()

    # 初始化训练权重
    def build(self, input_shape):
        self.k = self.add_weight(name='k',
                                 shape=(self.input_dim, self.output_dim),
                                 initializer='glorot_uniform',
                                 trainable=True)
        super(FMCrossLayer, self).build(input_shape)

    # 自定义FM的二阶交叉项的计算公式
    def call(self, x):
        inter_1 = K.pow(K.dot(x, self.k), 2)
        inter_2 = K.dot(K.pow(x, 2), K.pow(self.k, 2))
        return K.sum(inter_1 - inter_2, 1, keepdims=True) * 0.5

    # 输出的尺寸大小
    def output_shape(self, input_shape):
        return input_shape[0], self.output_dim


# 实现FM算法
def FM(feature_dim):
    """FM的模型方程：LR线性组合 + 交叉项组合 = 1阶特征组合 + 2阶特征组合"""
    inputs = Input((feature_dim,))
    # 线性回归
    liner = Dense(units=1, bias_regularizer=l2(0.01), kernel_regularizer=l1(0.02))(inputs)
    # FM的二阶交叉项
    cross = FMCrossLayer(feature_dim)(inputs)
    # 获得FM模型（线性回归 + FM的二阶交叉项）
    add = Add()([liner, cross])
    predict = Activation('sigmoid')(add)

    model = Model(inputs=inputs, outputs=predict)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=[tf.keras.metrics.binary_accuracy])
    return model


# 数据预处理
def preprocess(data):
    # 取特征(8个特征)
    feature = np.array(data.iloc[:, :-1])
    # 取标签并转化为 +1，-1
    label = data.iloc[:, -1].map(lambda x: 1 if x == 1 else 0)

    # 将数组按行进行归一化
    # 特征的最大值，特征的最小值
    zmax, zmin = feature.max(axis=0), feature.min(axis=0)
    feature = (feature - zmin) / (zmax - zmin)
    label = np.array(label)
    return feature, label


# 训练FM模型
def train():
    # 1、加载样本
    train_sample = pd.read_csv(train_file)
    test_sample = pd.read_csv(test_file)
    x_train, y_train = preprocess(train_sample)
    x_test, y_test = preprocess(test_sample)

    # 2、定义模型
    model = FM(8)

    # 3、训练模型
    model.fit(x_train, y_train, epochs=100, batch_size=20, validation_data=(x_test, y_test))
    return model


if __name__ == '__main__':
    fm = train()
    fm.summary()
