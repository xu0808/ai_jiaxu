#!/usr/bin/env python
# coding: utf-8
# 数据处理

from data_process import *
from model import *

# training params
batch_size = 1
nb_epochs = 10
patience = 100
lr = 0.005
ffd_drop = 0.6
attn_drop = 0.6
residual = False
Sparse = False

hid_units = [8]  # numbers of hidden units per each attention head in each layer
n_heads = [8, 1]  # additional entry for the output layer

nonlinearity = tf.nn.elu
optimizer = tf.keras.optimizers.Adam(lr=lr)


def train(model, inputs, bias_mat, lbl_in, msk_in):
    with tf.GradientTape() as tape:
        logits, accuracy, loss = model(inputs=inputs, bias_mat=bias_mat, lbl_in=lbl_in, msk_in=msk_in, training=True)
    # 梯度下降
    grads = tape.gradient(loss, model.trainable_variables)
    # 优化器
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return logits, accuracy, loss


def evaluate(model, inputs, bias_mat, lbl_in, msk_in):
    logits, accuracy, loss = model(inputs=inputs, bias_mat=bias_mat, lbl_in=lbl_in, msk_in=msk_in, training=False)
    return logits, accuracy, loss


if __name__ == "__main__":
    print('Dataset: ' + dataset)
    print('----- Opt. hyperparams -----')
    print('lr: ' + str(lr))
    print('l2_coef: ' + str(l2_coef))
    print('----- Archi. hyperparams -----')
    print('nb. layers: ' + str(len(hid_units)))
    print('nb. units per layer: ' + str(hid_units))
    print('nb. attention heads: ' + str(n_heads))
    print('residual: ' + str(residual))
    print('nonlinearity: ' + str(nonlinearity))

    # 加载数据
    # adj_matrix: 邻接矩阵(sparse matrix) 2708x2708
    # features：节点信息，2708x1433
    # y_train：标签信息
    # train_mask：哪些是训练样本的标志
    adj_matrix, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data()

    # features归一化
    # 返回：features：归一化后的features，，spars包括三个部分，值，坐标和shape
    features, spars = preprocess_features(features)

    nb_nodes = features.shape[0]
    ft_size = features.shape[1]
    nb_classes = y_train.shape[1]
    features = features[np.newaxis]
    y_train = y_train[np.newaxis]
    y_val = y_val[np.newaxis]
    y_test = y_test[np.newaxis]
    train_mask = train_mask[np.newaxis]
    val_mask = val_mask[np.newaxis]
    test_mask = test_mask[np.newaxis]

    print(f'These are the parameters')
    print(f'batch_size: {batch_size}')
    print(f'nb_nodes: {nb_nodes}')
    print(f'ft_size: {ft_size}')
    print(f'nb_classes: {nb_classes}')

    if Sparse:
        # 将边表adj处理成SparseTensor
        biases = preprocess_adj_bias(adj_matrix)

    else:
        # adj_to_bias 返回经过处理的adj，有边的地方变成0，无边的地方变成 -1e9
        adj_matrix = adj_matrix.todense()
        adj_matrix = adj_matrix[np.newaxis]
        biases = adj_to_bias(adj_matrix, [nb_nodes], nhood=1)

    # 定义模型
    # hid_units = [8]，n_heads = [8, 1]
    model = GAT(hid_units, n_heads, nb_classes, nb_nodes, Sparse, ffd_drop=ffd_drop, attn_drop=attn_drop,
                activation=tf.nn.elu, residual=False)
    print('model: ' + str('SpGAT' if Sparse else 'GAT'))
    vlss_mn = np.inf
    vacc_mx = 0.0
    curr_step = 0

    train_loss_avg = 0
    train_acc_avg = 0
    val_loss_avg = 0
    val_acc_avg = 0

    model_number = 0

    # 训练
    for epoch in range(nb_epochs):
        # ##Training Segment# ##
        start = 0
        while start < nb_nodes:
            end = start + batch_size
            bbias = biases if Sparse else biases[start:end]
            _, acc_tr, loss_value_tr = train(model, inputs=features[start:end], bias_mat=bbias,
                                             lbl_in=y_train[start:end], msk_in=train_mask[start:end])
            train_loss_avg += loss_value_tr
            train_acc_avg += acc_tr
            start += batch_size

        # ##Validation Segment# ##
        start = 0
        while start < nb_nodes:
            end = start + batch_size
            bbias = biases if Sparse else biases[start:end]
            _, acc_vl, loss_value_vl = evaluate(model, inputs=features[start:end], bias_mat=bbias,
                                                lbl_in=y_val[start:end], msk_in=val_mask[start:end])
            val_loss_avg += loss_value_vl
            val_acc_avg += acc_vl
            start += batch_size

        step = start // batch_size
        print('Training: loss = %.5f, acc = %.5f | Val: loss = %.5f, acc = %.5f' %
              (train_loss_avg / step, train_acc_avg / step,
               val_loss_avg / step, val_acc_avg / step))

        # ##Early Stopping Segment# ##

        if val_acc_avg / step >= vacc_mx or val_loss_avg / step <= vlss_mn:
            if val_acc_avg / step >= vacc_mx and val_loss_avg / step <= vlss_mn:
                vacc_early_model = val_acc_avg / step
                vlss_early_model = val_loss_avg / step
                working_weights = model.get_weights()
            vacc_mx = np.max((val_acc_avg / step, vacc_mx))
            vlss_mn = np.min((val_loss_avg / step, vlss_mn))
            curr_step = 0
        else:
            curr_step += 1
            if curr_step == patience:
                print('Early stop! Min loss: ', vlss_mn, ', Max accuracy: ', vacc_mx)
                print('Early stop model validation loss: ', vlss_early_model, ', accuracy: ', vacc_early_model)
                model.set_weights(working_weights)
                break

        train_loss_avg = 0
        train_acc_avg = 0
        val_loss_avg = 0
        val_acc_avg = 0

    # ##Testing Segment# ## Outside of the epochs

    ts_step = 0
    ts_size = features.shape[0]
    ts_loss = 0.0
    ts_acc = 0.0
    while ts_step * batch_size < ts_size:

        if Sparse:
            bbias = biases
        else:
            bbias = biases[ts_step * batch_size:(ts_step + 1) * batch_size]

        _, acc_ts, loss_value_ts = evaluate(model,
                                            inputs=features[ts_step * batch_size:(ts_step + 1) * batch_size],
                                            bias_mat=bbias,
                                            lbl_in=y_test[ts_step * batch_size:(ts_step + 1) * batch_size],
                                            msk_in=test_mask[ts_step * batch_size:(ts_step + 1) * batch_size],
                                            training=False)
        ts_loss += loss_value_ts
        ts_acc += acc_ts
        ts_step += 1

    print('Test loss:', ts_loss / ts_step, '; Test accuracy:', ts_acc / ts_step)
    # print('Test loss: %.5f, acc = %.5f | Val: loss = %.5f, acc = %.5f' %
    #                  (train_loss_avg/tr_step, train_acc_avg/tr_step,
    #                  val_loss_avg/vl_step, val_acc_avg/vl_step))
