#!/usr/bin/env python
# coding: utf-8
# Python手动实现FM
# diabetes皮马人糖尿病数据集FM二分类
# FM算法解析及Python代码实现
# https://www.jianshu.com/p/bb2bce9135e4
# FM(因子分解机)模型算法：稀疏数据下的特征二阶组合问题（个性化特征）
# 1、应用矩阵分解思想，引入隐向量构造FM模型方程
# 2、目标函数（损失函数复合FM模型方程）的最优问题：链式求偏导
# 3、SGD优化目标函数

import numpy as np
import random
import os
from numpy import *
from random import normalvariate  # 正态分布
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin
from collections import Counter

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

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
    label = data.iloc[:, -1].map(lambda x: 1 if x == 1 else 0)

    # 将数组按行进行归一化
    # 特征的最大值，特征的最小值
    zmax, zmin = feature.max(axis=0), feature.min(axis=0)
    feature = (feature - zmin) / (zmax - zmin)
    label = np.array(label)
    return feature, label


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 对每一个样本计算损失
def logit(y, y_hat):
    if y_hat == 'nan':
        return 0
    else:
        return np.log(1 + np.exp(-y * y_hat))


def df_logit(y, y_hat):
    return sigmoid(-y * y_hat) * (-y)


class FM(BaseEstimator):
    def __init__(self, k=5, learning_rate=0.01, iternum=2):
        self.w0 = None
        self.w = None
        self.v = None
        self.k = k
        self.alpha = learning_rate
        self.iternum = iternum

    def call(self, x):
        # 每个样本(1*n)x(n*k),得到k维向量（FM化简公式大括号内的第一项）
        inter_1 = (x.dot(self.v)) ** 2
        # 二阶交叉项计算，得到k维向量（FM化简公式大括号内的第二项）
        inter_2 = (x ** 2).dot(self.v ** 2)
        # 二阶交叉项计算完成（FM化简公式的大括号外累加）
        interaction = np.sum(inter_1 - inter_2) / 2.

        y_hat = self.w0 + x.dot(self.w) + interaction
        return y_hat[0]

    def sgd(self, x, y):
        m, n = np.shape(x)
        # 初始化参数
        self.w0 = 0
        self.w = np.random.uniform(size=(n, 1))
        # Vj是第j个特征的隐向量
        # Vjf是第j个特征的隐向量表示中的第f维
        self.v = np.random.uniform(size=(n, self.k))

        for it in range(self.iternum):
            total_loss = 0
            # X[i]是第i个样本
            for i in range(m):
                y_hat = self.call(x=x[i])
                # 计算logit损失函数值
                total_loss += logit(y=y[i], y_hat=y_hat)
                # 计算logit损失函数的外层偏导
                loss = df_logit(y=y[i], y_hat=y_hat)

                # 1、常数项梯度下降
                # 公式中的w0求导，计算复杂度O(1)
                loss_w0 = loss * 1
                self.w0 = self.w0 - self.alpha * loss_w0

                # X[i,j]是第i个样本的第j个特征
                for j in range(n):
                    if x[i, j] != 0:
                        # 公式中的wi求导，计算复杂度O(n)
                        loss_w = loss * x[i, j]
                        self.w[j] = self.w[j] - self.alpha * loss_w
                        # 公式中的vif求导，计算复杂度O(kn)
                        for f in range(self.k):
                            loss_v = loss * (x[i, j] * (x[i].dot(self.v[:, f])) - self.v[j, f] * x[i, j] ** 2)
                            self.v[j, f] = self.v[j, f] - self.alpha * loss_v

            print('iter={}, loss={:.4f}'.format(it + 1, total_loss / m))
        return self

    def predict(self, x):
        # sigmoid阈值设置
        predicts, threshold = [], 0.5
        # 遍历测试集
        for i in range(x.shape[0]):
            # FM的模型方程
            y_hat = self.call(x=x[i])
            predicts.append(-1 if sigmoid(y_hat) < threshold else 1)
        return np.array(predicts)

    def fit(self, x, y):
        if isinstance(x, pd.DataFrame):
            x = np.array(x)
            y = np.array(y)

        return self.sgd(x, y)


if __name__ == "__main__":
    # 1、数据集切分
    # train_sample, test_sample = load_data(0.8)
    # 2、加载样本
    train_sample = pd.read_csv(train_file)
    test_sample = pd.read_csv(test_file)
    x_train, y_train = preprocess(train_sample)
    x_test, y_test = preprocess(test_sample)

    model = FM(k=10, learning_rate=0.001, iternum=2)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_train)
    print('训练集roc: {:.2%}'.format(roc_auc_score(y_train, y_pred)))
    print('混淆矩阵: \n', confusion_matrix(y_train, y_pred))

    y_true = model.predict(x_test)

    print('验证集roc: {:.2%}'.format(roc_auc_score(y_test, y_true)))
    print('混淆矩阵: \n', confusion_matrix(y_test, y_true))

    from sklearn.metrics import mean_squared_error
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import recall_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import accuracy_score

    # 归一化测试集，返回[0,1]区间
    x_test = MinMaxScaler().fit_transform(x_test)
    val_predicts = model.predict(x_test)
    print('FM测试集的分类准确率为: {:.2%}'.format(accuracy_score(y_test, val_predicts)))
    print("FM测试集均方误差mse：{:.2%}".format(mean_squared_error(y_test, val_predicts)))
    print("FM测试集召回率recall：{:.2%}".format(recall_score(y_test, val_predicts)))
    print("FM测试集的精度precision：{:.2%}".format(precision_score(y_test, val_predicts)))

