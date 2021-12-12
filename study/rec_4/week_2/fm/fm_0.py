#!/usr/bin/env python
# coding: utf-8
# Python手动实现FM
# diabetes皮马人糖尿病数据集FM二分类
#
# FM算法解析及Python实现
# https://www.jianshu.com/p/bb2bce9135e4
# FM(因子分解机)模型算法：稀疏数据下的特征二阶组合问题（个性化特征）
# 1、应用矩阵分解思想，引入隐向量构造FM模型方程
# 2、目标函数（损失函数复合FM模型方程）的最优问题：链式求偏导
# 3、SGD优化目标函数

import os
import random
import pandas as pd
import numpy as np
from sklearn.metrics import recall_score, precision_score, accuracy_score, \
     roc_auc_score, confusion_matrix, mean_squared_error
from sklearn.base import BaseEstimator
from sklearn.preprocessing import MinMaxScaler


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
    """二分类输出非线性映射"""
    return 1 / (1 + np.exp(-x))


# 对每一个样本计算损失
def loss(y, y_hat):
    """计算logit损失函数：L = log(1 + e**(y_hat * y))"""
    if y_hat == 'nan':
        return 0
    else:
        return np.log(1 + np.exp(-y * y_hat))


def loss_grad_0(y, y_hat):
    """计算logit损失函数的外层偏导(不含y_hat的一阶偏导)"""
    return sigmoid(-y * y_hat) * (-y)


class FM(BaseEstimator):
    def __init__(self, k=5, learning_rate=0.01, step_num=2):
        self.w0 = None
        self.w = None
        self.v = None
        self.k = k
        self.alpha = learning_rate
        self.step_num = step_num

    def call(self, x):
        """FM的模型方程：LR线性组合 + 交叉项组合 = 1阶特征组合 + 2阶特征组合"""
        # 每个样本(1*n)x(n*k),得到k维向量（FM化简公式大括号内的第一项）
        inter_1 = (x.dot(self.v)) ** 2
        # 二阶交叉项计算，得到k维向量（FM化简公式大括号内的第二项）
        inter_2 = (x ** 2).dot(self.v ** 2)
        # 二阶交叉项计算完成（FM化简公式的大括号外累加）
        interaction = np.sum(inter_1 - inter_2) / 2.

        y_hat = self.w0 + x.dot(self.w)[0] + interaction
        return y_hat

    def fit(self, x, y):
        m, n = np.shape(x)
        # 初始化参数
        self.w0 = 0
        self.w = np.random.uniform(size=(n, 1))
        # Vj是第j个特征的隐向量
        # Vjf是第j个特征的隐向量表示中的第f维
        self.v = np.random.uniform(size=(n, self.k))

        for it in range(self.step_num):
            total_loss = 0
            # X[i]是第i个样本
            for i in range(m):
                x_i, y_true = x[i], y[i]
                y_hat = self.call(x=x_i)
                # 计算logit损失函数值
                total_loss += loss(y=y_true, y_hat=y_hat)
                # 计算损失函数的外层偏导
                grad_0 = loss_grad_0(y=y_true, y_hat=y_hat)

                # 1、常数项梯度下降
                # 公式中的w0求导，计算复杂度O(1)
                # l(y, y_hat)中y_hat展开w0，求关于w0的内层偏导
                grad_w0 = grad_0 * 1
                # 梯度下降更新w0
                self.w0 = self.w0 - self.alpha * grad_w0

                # X[i,j]是第i个样本的第j个特征
                for j in range(n):
                    x_ij = x[i, j]
                    if x_ij != 0:
                        # 公式中的wi求导，计算复杂度O(n)
                        # l(y, y_hat)中y_hat展开y_hat，求关于W[j]的内层偏导
                        grad_w = grad_0 * x_ij
                        # 梯度下降更新W[j]
                        self.w[j] = self.w[j] - self.alpha * grad_w
                        # 公式中的vif求导，计算复杂度O(kn)
                        for f in range(self.k):
                            # l(y, y_hat)中y_hat展开V[j, f]，求关于V[j, f]的内层偏导
                            grad_v = grad_0 * (x_ij * (x_i.dot(self.v[:, f])) - self.v[j, f] * x_ij ** 2)
                            # 梯度下降更新V[j, f]
                            self.v[j, f] = self.v[j, f] - self.alpha * grad_v
            print('iter={}, loss={:.4f}'.format(it + 1, total_loss / m))
        return self

    def predict(self, x):
        # sigmoid阈值设置
        y_pre, threshold = [], 0.5
        # 遍历测试集
        for i in range(x.shape[0]):
            # FM的模型方程
            y_hat = self.call(x=x[i])
            y_pre.append(-1 if sigmoid(y_hat) < threshold else 1)
        return np.array(y_pre)


if __name__ == "__main__":
    # 1、数据集切分
    # train_sample, test_sample = load_data(0.8)
    # 2、加载样本
    train_sample = pd.read_csv(train_file)
    test_sample = pd.read_csv(test_file)
    x_train, y_train = preprocess(train_sample)
    x_test, y_test = preprocess(test_sample)

    # 3、创建模型
    model = FM(k=10, learning_rate=0.05, step_num=200)
    # 4、训练模型
    model.fit(x_train, y_train)
    # 5、指标计算
    # 5-1、训练集
    y_pred = model.predict(x_train)
    print('训练集roc: {:.2%}'.format(roc_auc_score(y_train, y_pred)))
    print('混淆矩阵: \n', confusion_matrix(y_train, y_pred))
    # 5-2、测试集
    y_pre = model.predict(x_test)
    print('测试集roc: {:.2%}'.format(roc_auc_score(y_test, y_pre)))
    print('混淆矩阵: \n', confusion_matrix(y_test, y_pre))

    # 归一化测试集，返回[0,1]区间
    x_test = MinMaxScaler().fit_transform(x_test)
    y_pre = model.predict(x_test)
    print('FM测试集的分类准确率为: {:.2%}'.format(accuracy_score(y_test, y_pre)))
    print("FM测试集均方误差mse：{:.2%}".format(mean_squared_error(y_test, y_pre)))
    print("FM测试集召回率recall：{:.2%}".format(recall_score(y_test, y_pre)))
    print("FM测试集的精度precision：{:.2%}".format(precision_score(y_test, y_pre)))

