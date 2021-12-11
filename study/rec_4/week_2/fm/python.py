#!/usr/bin/env python
# coding: utf-8
# Python手动实现FM
# diabetes皮马人糖尿病数据集FM二分类

import numpy as np
import random
import os
from numpy import *
from random import normalvariate  # 正态分布
from datetime import datetime
import pandas as pd
import numpy as np

# 数据文件
data_dir = 'D:\\study\\rec_4\\data\\2\\fm'
data_file = os.path.join(data_dir, 'diabetes.csv')
train_file = os.path.join(data_dir, 'diabetes_train.txt')
test_file = os.path.join(data_dir, 'diabetes_test.txt')


# 数据集切分
def load_data(ratio):
    train_data = []
    test_data = []
    with open(data_file) as txt_data:
        lines = txt_data.readlines()
        for line in lines:
            line_data = line.strip().split(',')
            # 训练集占比
            if random.random() < ratio:
                train_data.append(line_data)
            else:
                test_data.append(line_data)
            np.savetxt(train_file, train_data, delimiter=',', fmt='%s')
            np.savetxt(test_file, test_data, delimiter=',', fmt='%s')
    return train_data, test_data


# 数据预处理
def preprocess(data):
    # 取特征(8个特征)
    feature = np.array(data.iloc[:, :-1])
    # 取标签并转化为 +1，-1
    label = data.iloc[:, -1].map(lambda x: 1 if x == 1 else -1)  

    # 将数组按行进行归一化
    # 特征的最大值，特征的最小值
    zmax, zmin = feature.max(axis=0), feature.min(axis=0)  
    feature = (feature - zmin) / (zmax - zmin)
    label = np.array(label)
    return feature, label


def sigmoid(inx):
    return 1.0 / (1 + np.exp(-inx))


def train(feature, label, k, iter=10, alpha=0.01):
    """
    # 训练FM模型
    :param feature:特征数组
    :param label: 标签数组
    :param k: 向量维数
    :param iter: 迭代次数
    :return: 常数项w_0, 一阶特征系数w, 二阶交叉特征系数v
    """
    # 数组转矩阵
    feature_matrix = mat(feature)
    # 矩阵的行列数，即样本数m和特征数n
    m, n = shape(feature_matrix)

    # 初始化参数
    # 常数项
    w_0 = 0
    # 一阶特征的系数
    w = zeros((n, 1))
    # 辅助向量(n*k)，二阶交叉特征的系数
    v = normalvariate(0, 0.2) * ones((n, k))

    for it in range(iter):
        # 随机优化，每次只使用一个样本
        for t in range(m):
            # 一、前向传播
            x = feature_matrix[t]
            # 二阶项的计算
            # 每个样本(1*n)x(n*k),得到k维向量（FM化简公式大括号内的第一项）
            inter_1 = multiply(x * v, x * v)
            # 二阶交叉项计算，得到k维向量（FM化简公式大括号内的第二项）
            inter_2 = multiply(x, x) * multiply(v, v)
            # 二阶交叉项计算完成（FM化简公式的大括号外累加）
            interaction = sum(inter_1 - inter_2) / 2.

            # FM的全部项之和
            p = w_0 + x * w + interaction

            # 二、反向求导
            # tmp迭代公式的中间变量，便于计算
            y = label[t]
            tmp = 1 - sigmoid(y * p[0, 0])
            # 常数项更新
            w_0 += alpha * tmp * y
            for i in range(n):
                x_0 = x[0, i]
                if x_0 != 0:
                    # 一阶特征的系数更新
                    w[i, 0] += alpha * tmp * y * x_0
                    for j in range(k):
                        # 二阶交叉特征的系数更新
                        v[i, j] += alpha * tmp * y * (x_0 * inter_1[0, j] - v[i, j] * x_0 * x_0)

        # 计算损失函数的值
        if it % 10 == 0:
            l = loss(predict(feature, w_0, w, v), label)
            print('第{}次迭代后的损失为{}'.format(it, l))

    return w_0, w, v


# 损失函数
def loss(y_pre, label):
    m = len(y_pre)
    l = 0.0
    for i in range(m):
        l -= log(sigmoid(y_pre[i] * label))
    return l


# 预测
def predict(feature, w_0, w, v):
    # 数组转矩阵
    feature_matrix = mat(feature)
    m = np.shape(feature_matrix)[0]
    result = []
    for t in range(m):
        x = feature_matrix[t]
        # 每个样本(1*n)x(n*k),得到k维向量（FM化简公式大括号内的第一项）
        inter_1 = multiply(x * v, x * v)
        # 二阶交叉项计算，得到k维向量（FM化简公式大括号内的第二项）
        inter_2 = multiply(x, x) * multiply(v, v)
        # 二阶交叉项计算完成（FM化简公式的大括号外累加）
        interaction = sum(inter_1 - inter_2) / 2.
        # 计算预测的输出
        p = w_0 + x * w + interaction
        pre = sigmoid(p[0, 0])
        result.append(pre)
    return result


# 评估预测的准确性
def accuracy(y_pre, label):
    m = len(y_pre)
    acc_num = 0
    for i in range(m):
        score_pre = y_pre[i]
        y_true = label[i]
        if float(score_pre) < 0.5 and y_true == -1.0:
            acc_num += 1
        elif float(y_pre[i]) >= 0.5 and y_true == 1.0:
            acc_num += 1
        else:
            continue

    return float(acc_num/m)


if __name__ == '__main__':
    # 1、数据集切分
    # train_sample, test_sample = load_data(0.8)
    # 2、加载样本
    train_sample = pd.read_csv(train_file)
    test_sample = pd.read_csv(test_file)
    x_train, y_train = preprocess(train_sample)
    x_test, y_test = preprocess(test_sample)
    date_start = datetime.now()

    print('开始训练')
    bias, weight, vector = train(x_train, y_train, 4, 100, 0.01)
    print('bias = ', bias)
    print('weight = ', weight)
    print('vector = ', vector)
    predict_train_result = predict(mat(x_train), bias, weight, vector)  # 得到训练的准确性
    print("'训练准确性为：%f" % (1 - accuracy(predict_train_result, y_train)))
    print('训练用时为：%s' % (datetime.now() - date_start))

    print("开始测试")
    predict_test_result = predict(mat(x_test), bias, weight, vector)  # 得到训练的准确性
    print("测试准确性为：%f" % (1 - accuracy(predict_test_result, y_test)))
