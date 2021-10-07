#!/usr/bin/env python
# coding: utf-8
# GraphSAGE-Tensorflow 2.0-无监督
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from dataset import PPIData,CoraData
from sample import neg_sampling
from model import GraphSage_unsup as sage

# 模型参数
# 隐藏单元节点数
hidden_dim = 128
# 每阶采样邻居的节点数
sample_sizes = [5, 5]
# 批处理大小
batch_size = 256
# 训练迭代次数
epochs = 1000
# 每个epoch循环的批次数
num_batch_per_epoch = 2

# 负采样
neg_weight = 1.0
neg_size = 20


# 模型训练
def train():
    for e in range(epochs):
        for batch in range(num_batch_per_epoch):
            # 随机取一批节点
            batch_index = np.random.choice(train_index, size=(batch_size,))
            batch_edge = edges[batch_index]

            # 本批节点的邻居节点采样
            batch_samples, batch_a, batch_b, batch_n = neg_sampling(batch_edge, nodes, neighbors_dict, sample_sizes, neg_size)
            # 所有采样节点的特征
            batch_sampling_x = [feature[idxs] for idxs in batch_samples]

            with tf.GradientTape() as tape:
                _ = model(batch_sampling_x, batch_a, batch_b, batch_n)
                loss = model.losses[0]
                # 梯度下降
                grads = tape.gradient(loss, model.trainable_variables)
                # 优化器
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

            print('Epoch {:03d} Batch {:03d} Loss: {:.4f}'.format(e, batch, loss))

        # 由于测试时没有分批所以节点选择不能太大否则内存溢出
        loss_list = [loss] + [evaluate(indexs) for indexs in [train_index, val_index, test_index]]
        evaluate_value.append(loss_list)
        print('Epoch {:03d} train accuracy: {} val accuracy: {} test accuracy:{}'
              .format(e, loss_list[1], loss_list[2], loss_list[3]))

    # 训练过程可视化
    fig, axes = plt.subplots(4, sharex=True, figsize=(12, 8))
    fig.suptitle('Training Metrics')
    names = ['Loss', 'Accuracy', 'Val Acc', 'Test Acc']
    for i in range(len(names)):
        axes[i].set_ylabel(names[i], fontsize=14)
        axes[i].plot(evaluate_value[:][i])

    plt.show()


# 模型评估
def evaluate(index):
    # 随机取一批节点
    batch_index = np.random.choice(index, size=(batch_size,))
    batch_edge = edges[batch_index]

    # 本批节点的邻居节点采样
    batch_samples, batch_a, batch_b, batch_n = neg_sampling(batch_edge, nodes, neighbors_dict, sample_sizes, neg_size)
    # 所有采样节点的特征
    batch_sampling_x = [feature[idxs] for idxs in batch_samples]
    _ = model(batch_sampling_x, batch_a, batch_b, batch_n)
    loss = model.losses[0]
    return loss


if __name__ == '__main__':
    # 1、数据读取
    # 使用Cora数据快速验证模型结构
    # feature, label, edges, neighbors_dict = CoraData().get_data()
    feature, edges, neighbors_dict = PPIData().get_data()
    nodes = np.asarray(list(neighbors_dict.keys()))
    # 2、分割训练、验证、测试集
    data_index = [0, 5000, 5200, 5400]
    [train_index, val_index, test_index] = [np.arange(data_index[i], data_index[i+1]) for i in range(3)]

    # 3、模型定义
    # 特征维度
    input_dim = feature.shape[1]
    model = sage(input_dim, hidden_dim, sample_sizes, neg_weight)

    # 4、优化器
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01, decay=5e-4)

    # 记录过程值，以便最后可视化
    evaluate_value = []
    train()
