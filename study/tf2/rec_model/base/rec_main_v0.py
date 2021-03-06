#!/usr/bin/env python
# coding: utf-8
# 模型训练

import data_utils
from m1_fm import FM_Model
from m2_wide_deep import WideDeep
from m3_deep_fm import DeepFM
from m4_deep_crossing import DCN
from m5_xdeep_fm import xDeepFM

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


def init_wide_deep():
    """
    2、WideDeep 模型
    """
    # 训练、测试集
    features, data_train, data_test = data_utils.emb_data(is_w_d=True)
    # 特征数
    feature_num = data_train[0].shape[-1]
    # 模型参数
    hidden_units = [256, 128, 64]
    output_dim = 1
    activation = 'relu'
    # 类别特征数
    sparse_num = len(features[1])
    # wide部分特征数[dense_inputs, one_hot_inputs]
    w2d_model = WideDeep(feature_num - sparse_num, features, hidden_units, output_dim, activation)
    return data_train, data_test, w2d_model


def init_deep_fm():
    """
    3、DeepFM 模型
    """
    # 训练、测试集
    emb_dim = 10
    features, data_train, data_test = data_utils.emb_data(emb_dim=emb_dim)
    k = 10
    w_reg = 1e-4
    v_reg = 1e-4
    hidden_units = [256, 128, 64]
    output_dim = 1
    activation = 'relu'
    deep_fm_model = DeepFM(features, emb_dim, k, w_reg, v_reg, hidden_units, output_dim, activation)
    return data_train, data_test, deep_fm_model


def init_deep_cross():
    """
    4、DCN 模型
    """
    # 训练、测试集
    emb_dim = 32
    features, data_train, data_test = data_utils.emb_data(emb_dim=emb_dim)
    hidden_units = [256, 256]
    res_layer_num = 4
    activation = 'relu'
    deep_cross_model = DCN(features, emb_dim, hidden_units, res_layer_num, activation)
    return data_train, data_test, deep_cross_model


def init_x_deep_fm():
    """
    5、xDeepFM 模型
    """
    # 训练、测试集
    emb_dim = 10
    features, data_train, data_test = data_utils.emb_data(emb_dim=emb_dim)
    hidden_units = [256, 128, 64]
    dropout = 0.3
    cin_sizes = [128, 128]
    activation = 'relu'
    deep_fm_model = xDeepFM(features, emb_dim, cin_sizes, hidden_units, activation=activation, dropout=dropout)
    return data_train, data_test, deep_fm_model


if __name__ == '__main__':
    # 1、模型
    # 1-1、FM
    # (x_train, y_train), (x_test, y_test), model = init_fm()
    # 1-2、Wide&Deep
    # (x_train, y_train), (x_test, y_test), model = init_wide_deep()
    # 1-3、Wide&Deep
    # (x_train, y_train), (x_test, y_test), model = init_deep_fm()
    # 1-4、DeepCrossing
    # (x_train, y_train), (x_test, y_test), model = init_deep_cross()
    # 1-5、xDeepFM
    (x_train, y_train), (x_test, y_test), model = init_x_deep_fm()

    # 2、优化器
    optimizer = optimizers.SGD(0.01, clipnorm=1)
    # 3、keras标准模型训练
    # 合并后进行batch操作
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_data = train_data.batch(32).prefetch(tf.data.experimental.AUTOTUNE)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_data, epochs=20)
    print('train finish!')

    # 4、评估
    loss, auc = model.evaluate(x_test, y_test)
    print('eval loss = %6f, auc = %6f' % (loss, auc))
    # 5、模型结构
    model.summary()
