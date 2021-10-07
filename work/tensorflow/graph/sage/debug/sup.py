#!/usr/bin/env python3
# coding: utf-8
# sage有监督学习

import numpy as np
import tensorflow as tf
import time
import cora
from itertools import islice
from sklearn.metrics import f1_score
from mini_batch import build_batch_from_nodes as build_batch
from graph_sage import GraphSageSupervised as Sage


# 每阶邻节点采样数
SAMPLE_SIZES = [5, 5]
# 样本维度
INPUT_DIM = 128
# 训练批样本数
BATCH_SIZE = 256
# 模型训练迭代次数
STEPS = 5
# 学习率
LEARNING_RATE = 0.5


def run():
    # 1、加载数据
    # feature, label, neighbors_dict = CoraData().get_data()
    # num_nodes, num_classes = label.shape
    num_nodes, feature, label, num_classes, neighbors_dict = cora.load_cora()

    # 2、定义模型
    model = Sage(feature, INPUT_DIM, len(SAMPLE_SIZES), num_classes)

    all_nodes = np.random.permutation(num_nodes)
    train_nodes = all_nodes[:2048]
    test_nodes = all_nodes[2048:]

    # training
    
    # 构造训练数据集
    # 训练集对应的label
    def training_minibatch(nodes_for_training, labels, batch_size):
        while True:
            # 从 nodes_for_training 中随机采取 batch_size 个样本[4,6,34,98]
            batch_nodes = np.random.choice(nodes_for_training, size=batch_size, replace=False)
            batch_node = build_batch(batch_nodes, neighbors_dict, SAMPLE_SIZES)
            batch_label = labels[batch_nodes]
            yield (batch_node, batch_label)

    minibatch = training_minibatch(train_nodes, label, BATCH_SIZE)
    optimizer = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    times = []
    # 循环调用minibatch_generator,调用次数：TRAINING_STEPS
    for inputs, inputs_labels in islice(minibatch, 0, STEPS):
        start_time = time.time()
        with tf.GradientTape() as tape:
            predicted = model(inputs)
            loss = loss_fn(tf.convert_to_tensor(inputs_labels), predicted)

        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        end_time = time.time()
        times.append(end_time - start_time)
        print("Loss[]:", loss.numpy())

    # testing
    batch = build_batch(test_nodes, neighbors_dict, SAMPLE_SIZES)
    results = model(batch)
    score = f1_score(label[test_nodes], results.numpy().argmax(axis=1), average="micro")
    print("Validation F1: ", score)
    print("Average batch time: ", np.mean(times))


if __name__ == "__main__":
    run()
