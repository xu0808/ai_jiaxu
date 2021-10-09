#!/usr/bin/env python
# coding: utf-8
# GraphSAGE-Tensorflow 2.0-有监督
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from v1.dataset import CoraData
from v1.sample import multihop_sampling
from model import GraphSage

# 模型参数
# 隐藏单元节点数
hidden_dim = 128
# 每阶采样邻居的节点数
sample_sizes = [5, 5]
# 批处理大小
batch_size = 256
# 训练迭代次数
epochs = 4
# 每个epoch循环的批次数
num_batch_per_epoch = 4


# 模型训练
def train():
    for e in range(epochs):
        for batch in range(num_batch_per_epoch):
            # 随机取一批节点
            batch_src_index = np.random.choice(train_index, size=(batch_size,))
            # 本批节点的标签
            batch_src_label = train_label[batch_src_index].astype(float)
            # 本批节点的邻居节点采样
            batch_sampling_list = multihop_sampling(batch_src_index, sample_sizes, neighbors_dict)
            # 所有采样节点的特征
            batch_sampling_x = [feature[idxs] for idxs in batch_sampling_list]

            with tf.GradientTape() as tape:
                # 模型计算
                batch_train_logits = model(batch_sampling_x)
                # 计算损失函数
                loss = loss_object(batch_src_label, batch_train_logits)
                # 梯度下降
                grads = tape.gradient(loss, model.trainable_variables)
                # 优化器
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

            print('Epoch {:03d} Batch {:03d} Loss: {:.4f}'.format(e, batch, loss))

        # 由于测试时没有分批所以节点选择不能太大否则内存溢出
        acc_list = [loss] + [evaluate(indexs) for indexs in [train_index, val_index, test_index]]
        evaluate_value.append(acc_list)
        print('Epoch {:03d} train accuracy: {} val accuracy: {} test accuracy:{}'
              .format(e, acc_list[1], acc_list[2], acc_list[3]))

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
    test_sampling_result = multihop_sampling(index, sample_sizes, neighbors_dict)
    test_x = [feature[idx.astype(np.int32)] for idx in test_sampling_result]
    test_logits = model(test_x)
    test_label = label[index]
    ll = tf.math.equal(tf.math.argmax(test_label, -1), tf.math.argmax(test_logits, -1))
    accuarcy = tf.reduce_mean(tf.cast(ll, dtype=tf.float32))
    return accuarcy


if __name__ == '__main__':
    # 1、数据读取
    feature, label, _, neighbors_dict = CoraData().get_data()

    # 2、分割训练、验证、测试集
    [train_index, val_index, test_index] = [np.arange(i * 500, (i + 1) * 500) for i in range(3)]
    train_label = label[train_index]

    # 3、模型定义
    # 特征维度
    input_dim = feature.shape[1]
    # 分类数
    num_class = label.shape[1]
    model = GraphSage(input_dim=input_dim, hidden_dim=hidden_dim, sample_sizes=sample_sizes, num_class=num_class)

    # 4、损失函数
    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    # 5、优化器
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01, decay=5e-4)

    # 记录过程值，以便最后可视化
    evaluate_value = []
    train()
