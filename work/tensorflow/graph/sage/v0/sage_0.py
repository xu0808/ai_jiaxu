#!/usr/bin/env python
# coding: utf-8
# GraphSAGE极简版
# 学习聚合函数 GraphSAGE
# https://blog.csdn.net/weixin_44413191/article/details/109008096
"""
相对于DeepWalk, node2vec的改进主要是对基于随机游走的采样策略的改进
node2vec是结合了BFS和DFS的Deepwalk改进的随机游走算法
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# GNN库tf_geometric可以方便地导入数据集，预处理图数据以及搭建图神经网络
import tf_geometric as tfg
from tf_geometric.datasets.ppi import PPIDataset
from tf_geometric.utils.graph_utils import RandomNeighborSampler
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm

# PPI（Protein-protein interaction networks）数据集
# x: 节点的特征，维度为 2708 * 1433，类型为 np.ndarray
# y: 节点的标签，总共包括7个类别，类型为 np.ndarray
# adjacency_dict: 邻接信息，，类型为 dict

# 平均每张图有2372个节点，每个节点有50个特征。
# 每个节点拥有多种标签，标签的种类总共有121种。
# PPI数据集，训练集(20)，验证集(2)，测试集(2)
train_graphs, valid_graphs, test_graphs = PPIDataset().load_data()
# 标签的分类数
num_classes = train_graphs[0].y.shape[1]
# 邻居节点采样数目分别为25和10
num_sampled_neighbors_list = [25, 10]

# 对每张图进行预处理，保存图信息到图缓存字典中
for graph in train_graphs + valid_graphs + test_graphs:
    neighbor_sampler = RandomNeighborSampler(graph.edge_index)
    graph.cache['sampler'] = neighbor_sampler


# 采用两层MaxPool聚合函数来聚合邻居节点蕴含的信息
graph_sages = [
    tfg.layers.MaxPoolGraphSage(units=256, activation=tf.nn.relu, concat=True),
    tfg.layers.MaxPoolGraphSage(units=256, activation=tf.nn.relu, concat=True)
]
# 用Sequential快速创建神经网络
fc = tf.keras.Sequential([
    keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(num_classes)
])


def forward(graph, training=False):
    h = graph.x
    for i, (graph_sage, num) in enumerate(zip(graph_sages, num_sampled_neighbors_list)):
        # 从图信息缓存中采样num个样本
        edge_index, edge_weight = graph.cache['sampler'].sample(k=num)
        # 将采样得到的边、边权、节点的特征向量输入到模型
        h = graph_sage([h, edge_index, edge_weight], training=training)
    h = fc(h, training=training)
    return h


def compute_loss(logits, vars):
    """
    计算模型损失
    数据集属于多标签、多分类任务，选用sigmoid交叉熵函数
    logits 图节点标签的预测结果
    graph.y 图节点的真实标签
    """
    losses = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=logits,
        labels=tf.convert_to_tensor(graph.y, dtype=tf.float32)
    )

    kernel_vars = [var for var in vars if "kernel" in var.name]
    # 防止过拟合，对模型的参数使用 L2 正则化。
    l2_losses = [tf.nn.l2_loss(kernel_var) for kernel_var in kernel_vars]
    return tf.reduce_mean(losses) + tf.add_n(l2_losses) * 1e-5


def calc_f1(y_true, y_pred):
    y_pred[y_pred > 0] = 1
    y_pred[y_pred <= 0] = 0
    return f1_score(y_true, y_pred, average="micro")


# 使用F1 Score评估聚合邻居节点信息分类任务的性能
def evaluate(graphs):
    y_preds = []
    y_true = []

    for graph in graphs:
        y_true.append(graph.y)
        logits = forward(graph)
        y_preds.append(logits.numpy())

    # 预测结果与其对应的labels转换为一维数组
    y_pred = np.concatenate(y_preds, axis=0)
    y = np.concatenate(y_true, axis=0)
    mic = calc_f1(y, y_pred)
    return mic


# 定义优化器
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)

for epoch in tqdm(range(20)):
    for graph in train_graphs:
        with tf.GradientTape() as tape:
            # 前向传播，执行Dropou
            logits = forward(graph, training=True)
            # 计算误差
            loss = compute_loss(logits, tape.watched_variables())

        vars = tape.watched_variables()
        # 计算梯度
        grads = tape.gradient(loss, vars)
        # 优化器进行优化
        optimizer.apply_gradients(zip(grads, vars))

    if epoch % 1 == 0:
        valid_f1_mic = evaluate(valid_graphs)
        test_f1_mic = evaluate(test_graphs)
        print("epoch = {} loss = {} valid_f1_micro = {}".format(epoch, loss, valid_f1_mic))
        print("epoch = {} test_f1_micro = {}".format(epoch, test_f1_mic))


test_f1_mic = evaluate(test_graphs)
print("test_f1_micro = {}".format(test_f1_mic))
