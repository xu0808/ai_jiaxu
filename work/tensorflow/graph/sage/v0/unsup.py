#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import time
from itertools import islice
import ppi
from mini_batch import build_batch_from_edges as build_batch
from graph_sage import GraphSageUnsupervised as GraphSage

# 模型参数
SAMPLE_SIZES = [25, 10]
NEG_WEIGHT = 1.0
NEG_SIZE = 20

# 样本维度
INPUT_DIM = 128
# 训练批样本数
BATCH_SIZE = 512
# 模型训练迭代次数
STEPS = 1000
# 学习率
LEARNING_RATE = 0.00001


def mini_batch(neighbors_dict, batch_size, sample_sizes, neg_size):
    # 根据邻居信息构造边
    edges = np.array([(k, v) for k in neighbors_dict for v in neighbors_dict[k]])
    # 节点列表
    nodes = np.array(list(neighbors_dict.keys()))
    while True:
        # 对边进行采样，得到采样后的边：mini_batch_edges
        # edges.shape[0]=10556,产生(0,10556)中的batch_size个数字[2,44,66,]作为edge的索引。
        # 得到边的数组
        mini_batch_edges = edges[np.random.randint(edges.shape[0], size=batch_size), :]
        batch = build_batch(mini_batch_edges, nodes, neighbors_dict, sample_sizes, neg_size)
        yield batch


def run():
    # 数据读取
    num_nodes, features, neighbors_dict = ppi.load_ppi()
    batch = mini_batch(neighbors_dict, BATCH_SIZE, SAMPLE_SIZES, NEG_SIZE)
    model = GraphSage(features, INPUT_DIM, len(SAMPLE_SIZES), NEG_WEIGHT)
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    # training
    times = []
    step = 0
    for minibatch in islice(batch, 0, STEPS):
        start_time = time.time()
        with tf.GradientTape() as tape:
            _ = model(minibatch)
            loss = model.losses[0]

        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        end_time = time.time()
        times.append(end_time - start_time)
        print("step {},Loss {}".format(step, loss.numpy()))
        step += 1
    print("Average batch time: ", np.mean(times))


if __name__ == "__main__":
    run()
