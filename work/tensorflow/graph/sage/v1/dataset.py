#!/usr/bin/env python
# coding: utf-8
# 数据加载

import numpy as np
from collections import defaultdict
from sklearn.preprocessing import StandardScaler


def one_hot(label):
    classes = set(label)
    # np.identity(3)
    # array([[1., 0., 0.],
    #        [0., 1., 0.],
    #        [0., 0., 1.]])
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array([classes_dict[c] for c in label], dtype=np.int32)
    return labels_onehot


def normalize(feature):
    normal_features = feature / feature.sum(1).reshape(-1, 1)
    return normal_features


class CoraData:
    """
    数据说明
    # 论文特征和标签
        # cora.content共2708行，每行代表一个样本点，即一篇论文。
        # 一行由三部分组成：
        # 论文编号（raw_data的编号并非0~2708）。
        # 接下来1433列是论文的词向量。
        # 最后一列为论文类别，如 Neural_Networks 。
    # 论文引用关系
        # cora.cites共有5429行，每一行有两个论文编号，
        # 表示第一个论文先写，第二个论文引用第一个论文。
        # 如果将论文看做图中的点，那么这5429行便是点之间的5429条边。
        # 统计所有邻节点

    """
    def __init__(self, data_root='E:\\workspace\\ai_jiaxu\\data\\graph\\cora\\'):
        self.data_root = data_root

    def load_file(self, dataset='cora'):
        print('Loading {} dataset...'.format(dataset))
        content_file = '{}{}.content'.format(self.data_root, dataset)
        idx_features_labels = np.genfromtxt(content_file, dtype=np.dtype(str))
        cites_file = '{}{}.cites'.format(self.data_root, dataset)
        edges = np.genfromtxt(cites_file, dtype=np.int32)
        return idx_features_labels, edges

    def get_data(self):
        print('Process gat ...')
        # （论文编号[0]_词向量_论文类别[-1]）,（原始论文，引用者）
        idx_features_labels, edges_0 = self.load_file()
        # 特征
        feature = idx_features_labels[:, 1:-1].astype(np.float32)
        # 特征正则化
        feature = normalize(feature)
        # 分类名称标签
        y = idx_features_labels[:, -1]
        # 分类onehot标签
        label = one_hot(y)
        # 论文编号字典
        idx_map = {}
        # 计算每个节点的所有邻居节点(k->node, v->neighbors)
        # 所有边中添加点到点本身
        neighbors_dict = {}
        for index, idx in enumerate(idx_features_labels[:, 0]):
            idx_map[idx] = index
            neighbors_dict[index] = [index]

        edges = []
        for edge in edges_0:
            # 论文编号转索引
            [key, value] = [idx_map[str(node)] for node in edge]
            edges.append([key, value])
            neighbors_dict[key].append(value)

        print('Dataset has {} nodes, {} edges, {} features.'
              .format(feature.shape[0], len(edges), feature.shape[1]))
        return feature, label, np.array(edges),  neighbors_dict


class PPIData:
    def __init__(self, data_root='E:\\workspace\\ai_jiaxu\\data\\graph\\ppi\\'):
        self.data_root = data_root

    def load_file(self, dataset='toy-ppi'):
        print('Loading {} dataset...'.format(dataset))
        feat_file = '{}{}-feats.npy'.format(self.data_root, dataset)
        feature = np.load(feat_file).astype(np.float32)
        walk_file = '{}{}-walks.txt'.format(self.data_root, dataset)
        edges = np.genfromtxt(walk_file, dtype=np.int32)
        return feature, edges

    def get_data(self):
        print('Process gat ...')
        feature_0, edges = self.load_file()
        max_node = 0
        # 计算每个节点的所有邻居节点(k->node, v->neighbors)
        neighbors_dict = defaultdict(list)
        for edge in edges:
            [node_a, node_b] = edge
            if node_a > max_node:
                max_node = node_a
            if node_b > max_node:
                max_node = node_b
            neighbors_dict[node_a].append(node_b)

        # 特征标准化(为何不能使用归一化??)
        feature_1 = feature_0[0:max_node+1, :]
        scaler = StandardScaler()
        scaler.fit(feature_1)
        feature = scaler.transform(feature_1)
        # feature shape (9716, 50),edges number 1895817, key size 9684, max_node 9715
        print('feature shape {},edges number {}, key size {}, max_node {}'
              .format(feature.shape, len(edges), len(neighbors_dict), max_node))
        return feature, edges, neighbors_dict


if __name__ == "__main__":
    cora_data = CoraData().get_data()
    # ppi_data = PPIData().get_data()
    print('loader gat')
