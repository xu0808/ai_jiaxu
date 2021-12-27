#!/usr/bin/env python
# coding: utf-8
# 数据读取

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def click_data(test_size=0.3):
    """
    criteo:非常经典的点击率预估比赛数据
    数据集片段，2000条样本
    字段：数值特征：I1-13, 类别特征：C14-C39
    label,I1,I2,I3,I4,I5,I6,I7,I8,I9,I10,I11,I12,I13,C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11,C12,C13,C14,C15,C16,C17,C18,C19,C20,C21,C22,C23,C24,C25,C26
    0,1.0,1,5.0,0.0,1382.0,4.0,15.0,2.0,181.0,1.0,2.0,,2.0,68fd1e64,80e26c9b,fb936136,7b4723c4,25c83c98,7e0ccccf,de7995b8,1f89b562,a73ee510,a8cd5504,b2cb9c98,37c9c164,2824a5f6,1adce6ef,8ba8b39a,891b62e7,e5ba7672,f54016b9,21ddcdc9,b1252a9d,07b5194c,,3a171ecb,c5c50484,e8b83407,9727dd16
    0,2.0,0,44.0,1.0,102.0,8.0,2.0,2.0,4.0,1.0,1.0,,4.0,68fd1e64,f0cf0024,6f67f7e5,41274cd7,25c83c98,fe6b92e5,922afcc0,0b153874,a73ee510,2b53e5fb,4f1b46f3,623049e6,d7020589,b28479f6,e6c5b5cd,c92f3b61,07c540c4,b04e4670,21ddcdc9,5840adea,60f6221e,,3a171ecb,43f13e8b,e8b83407,731c3655
    0,2.0,0,1.0,14.0,767.0,89.0,4.0,2.0,245.0,1.0,3.0,3.0,45.0,287e684f,0a519c5c,02cf9876,c18be181,25c83c98,7e0ccccf,c78204a1,0b153874,a73ee510,3b08e48b,5f5e6091,8fe001f4,aa655a2f,07d13a8f,6dc710ed,36103458,8efede7f,3412118d,,,e587c466,ad3062eb,3a171ecb,3b183c5c,,
    """
    # 原始数据
    data = pd.read_csv('train.txt')
    print('data.shape = ', data.shape)
    # 数值特征列
    dense_cols = ['I' + str(i) for i in range(1, 14)]
    # 类别特征列
    sparse_cols = ['C' + str(i) for i in range(1, 27)]

    # 缺失值填充
    data[dense_cols] = data[dense_cols].fillna(0)
    data[sparse_cols] = data[sparse_cols].fillna('-1')
    # 归一化
    data[dense_cols] = MinMaxScaler().fit_transform(data[dense_cols])
    # one-hot编码
    # 1、Series的整数会按照one-hot进行编码但DataFrame不会
    # 2、特征的维度数量会有所增加

    data_1 = pd.get_dummies(data)
    print('data_1.shape = ', data_1.shape)

    # 数据集划分
    x = data_1.drop(['label'], axis=1).values
    y = data_1['label']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    # 1、点击数据
    click_data()