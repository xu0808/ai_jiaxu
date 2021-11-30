#!/usr/bin/python
# -*- coding: utf-8 -*-
# PinSage基础模型
# https://github.com/breadbread1984/PinSage-tf2.0

import numpy as np
import networkx as nx
import tensorflow as tf
from tensorflow.keras.layers import Dense, LeakyReLU, Concatenate


# 卷积操作类
class Convolve(tf.keras.Model):

    def __init__(self, hidden_channels):
        super(Convolve, self).__init__()
        # 聚合前每个邻居节点需经过的dense层
        self.Q = Dense(units=hidden_channels, activation=LeakyReLU())
        # 邻居节点聚合后的emb与目标节点emb拼接后需经过的dense层
        self.W = Dense(units=hidden_channels, activation=LeakyReLU())

    def call(self, inputs):
        # embeddings.shape = (batch, node number, in_channels)
        # 所有节点的Embedding
        embs = inputs[0]
        # weights.shape = (node number, node number)
        # 所有边权重
        weights = inputs[1]
        # neighbor_set.shape = (node number, neighbor number)  ==> (节点数，邻居数)
        # 针对每个节点采样的邻居节点id集合
        neighbor_set = inputs[2]
        # neighbor_embeddings.shape = (batch, node number, neighbor number, in channels)
        gather_nd = tf.gather_nd(tf.transpose(embs, (1, 0, 2)), tf.expand_dims(neighbor_set, axis=-1))
        neighbor_embs = tf.transpose(gather_nd, (2, 0, 1, 3))

        # neighbor_hiddens.shape = (batch, node_number, neighbor number, hidden channels)
        # 所有的邻居Embedding经过第一层dense层
        neighbor_hiddens = self.Q(neighbor_embs)
        # indices.shape = (node_number, neighbor number, 2)
        tf_range = tf.expand_dims(tf.range(tf.shape(neighbor_set)[0]), axis=1)
        node_nums = tf.tile(tf_range, (1, tf.shape(neighbor_set)[1]))
        # 所有邻居节点及其对应的目标节点id
        indices = tf.stack([node_nums, neighbor_set], axis=-1)
        # neighbor weights.shape = (node number, neighbor number)
        # 提取所有要计算的邻居边的权重
        neighbor_weights = tf.gather_nd(weights, indices)
        # neighbor_weights.shape = (1, node number, neighbor number, 1)
        neighbor_weights = tf.expand_dims(tf.expand_dims(neighbor_weights, 0), -1)
        # weighted_sum_hidden.shape = (batch, node_number, hidden channels)
        # 对所有节点的邻居节点Embedding，根据其与目标节点的边的权重计算加权和
        weighted_sum_hidden = tf.math.reduce_sum(neighbor_hiddens * neighbor_weights, axis=2) / (
                tf.math.reduce_sum(neighbor_weights, axis=2) + 1e-6)
        # concated_hidden.shape = (batch, node number, in_channels + hidden channels)
        # 节点的原始Embedding与每个节点的邻居加权和Embedding拼接
        concated_hidden = Concatenate(axis=-1)([embs, weighted_sum_hidden])
        # hidden_new.shape = (batch, node number, hidden_channels)
        # 拼接后的Embedding经过第二层dense层
        hidden_new = self.W(concated_hidden)
        # normalized.shape = (batch, node number, hidden_channels)
        # 结果Embedding规范化
        normalized = hidden_new / (tf.norm(hidden_new, axis=2, keepdims=True) + 1e-6)
        return normalized


class PinSage(tf.keras.Model):

    def __init__(self, hidden_channels, graph=None, edge_weights=None):
        super(PinSage, self).__init__()
        # 创建卷积层(hidden_channels:每层隐藏层输出维度)
        self.convs = list()
        for i in range(len(hidden_channels)):
            self.convs.append(Convolve(hidden_channels[i]))
        # 在原始图上计算PageRank权重
        self.edge_weights = self.pagerank(graph) if graph is not None else edge_weights

    def call(self, inputs):

        # embeddings.shape = (batch, node number, in channels)
        # 所有节点的Embedding
        embeddings = inputs[0]
        # sample_neighbor_num.shape = ()
        # 邻居采样个数
        sample_neighbor_num = inputs[1]
        # sample a fix number of neighbors according to edge weights.
        # neighbor_set.shape = (node num, neighbor num)
        # 根据边的权重对邻居采样
        neighbor_set = tf.random.categorical(self.edge_weights, sample_neighbor_num)

        for conv in self.convs:
            embeddings = conv([embeddings, self.edge_weights, neighbor_set])
        return embeddings

    def pagerank(self, graph, damp_rate=0.2):
        node_num = len(graph.nodes)
        # 节点清单
        node_ids = sorted([id for id in graph.nodes])
        # 节点清单必须是从0开始的完整序列
        assert node_ids == list(range(node_num))
        # 邻接矩阵
        weights = np.zeros((node_num, node_num,), dtype=np.float32)
        for f in graph.nodes:
            for t in list(graph.adj[f]):
                # 边的标记
                weights[f, t] = 1.
        weights = tf.constant(weights)
        # 邻接矩阵正则化(行内平均)
        line_sum = tf.math.reduce_sum(weights, axis=1, keepdims=True) + 1e-6
        normalized = weights / line_sum
        # 均值向量
        damp_ones = tf.ones((node_num,), dtype=tf.float32)
        # shape=(4,)
        damp_avg = damp_ones / tf.constant(node_num, dtype=tf.float32)
        # 添加维度 shape=(1, 4)
        dampping = tf.expand_dims(damp_avg, 0)
        # learning pagerank.
        v = dampping
        while True:
            v_updated = (1 - damp_rate) * tf.linalg.matmul(v, normalized) + damp_rate * dampping
            d = tf.norm(v_updated - v)
            if tf.equal(tf.less(d, 1e-4), True):
                break
            v = v_updated
        # edge weight is pagerank.
        weights = weights * tf.tile(v, (tf.shape(weights)[0], 1))
        line_sum = tf.math.reduce_sum(weights, axis=1, keepdims=True) + 1e-6
        normalized = weights / line_sum
        return normalized


if __name__ == "__main__":
    assert tf.executing_eagerly()
    g = nx.Graph()
    g.add_node(0)
    g.add_node(1)
    g.add_node(2)
    g.add_node(3)
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    g.add_edge(2, 3)
    pinsage = PinSage([10, 10, 10], g)
    print(pinsage.edge_weights)
