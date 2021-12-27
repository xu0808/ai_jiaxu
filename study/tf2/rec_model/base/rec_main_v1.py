#!/usr/bin/env python
# coding: utf-8
# 模型训练

import data_utils
from m1_fm import FM_Model

import tensorflow as tf
from tensorflow.keras import optimizers, losses, metrics
from sklearn.metrics import accuracy_score


def init_fm():
    """
    1、FM 模型
    """
    # 训练、测试集
    data_train, data_test = data_utils.click_data(test_size=0.5)
    # 特征数
    feature_num = data_train[0].shape[-1]
    fm_model = FM_Model(k=8, w_reg=1e-4, v_reg=1e-4, f_n=feature_num)
    return data_train, data_test, fm_model


if __name__ == '__main__':
    # 1、模型
    # 1-1、FM
    (x_train, y_train), (x_test, y_test), model = init_fm()

    # 2、优化器
    optimizer = optimizers.SGD(0.01)
    # 创建摘要文件写入器
    summary_writer = tf.summary.create_file_writer('D:\\study\\data\\tf2_rec_1222')
    # 3、训练
    for i in range(100):
        with tf.GradientTape() as tape:
            y_pre = model(x_train)
            loss = tf.reduce_mean(losses.binary_crossentropy(y_true=y_train, y_pred=y_pre))
            print('Epoch [%d]:loss = ' % i, loss.numpy())
        with summary_writer.as_default():
            tf.summary.scalar('loss', loss, step=i)
        grad = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grad, model.variables))

    # 4、评估
    pre = model(x_test)
    pre = [1 if x > 0.5 else 0 for x in pre]
    print("AUC: ", accuracy_score(y_test, pre))
