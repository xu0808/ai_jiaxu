#!/usr/bin/env python
# coding: utf-8
# 模型训练

import data_utils
from m1_fm import FM_Model

import tensorflow as tf
from tensorflow.keras import optimizers


def init_fm():
    """
    1、FM 模型
    """
    # 训练、测试集
    data_train, data_test = data_utils.one_hot_data(test_size=0.5)
    # 特征数
    feature_num = data_train[0].shape[-1]
    fm_model = FM_Model(k=8, w_reg=1e-4, v_reg=1e-4, f_n=feature_num)
    print('Total params = ', feature_num * (8+1) + 1)
    return data_train, data_test, fm_model


if __name__ == '__main__':
    # 1、模型
    # 1-1、FM
    (x_train, y_train), (x_test, y_test), model = init_fm()

    # 2、优化器
    optimizer = optimizers.SGD(0.01)
    # 3、keras标准模型训练
    # 合并后进行batch操作
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_data = train_data.batch(32).prefetch(tf.data.experimental.AUTOTUNE)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_data, epochs=20)

    # 4、评估
    loss, auc = model.evaluate(x_test, y_test)
    print('eval loss = %6f, auc = %6f' % (loss, auc))
    # 5、模型结构
    model.summary()
