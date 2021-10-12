#!/usr/bin/env python
# coding: utf-8
# 数据处理

import tensorflow as tf
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import sys
from collections import defaultdict

dataset = 'cora'  # ["cora", "citeseer", "pubmed"]

node_max = 2000 - 1
datapath = 'E:\\workspace\\ai_jiaxu\\data\\graph\\gat'

"""
 Prepare adjacency matrix by expanding up to a given neighbourhood.
 This will insert loops on every node.
 Finally, the matrix is converted to bias vectors.
 Expected shape: [graph, nodes, nodes]
"""


def adj_to_bias(adj_dict, sizes, nhood=1):
    nb_graphs = adj_dict.shape[0]
    # 随机生成array,shape=(adj.shape,)
    mt = np.empty(adj_dict.shape)
    for g in range(nb_graphs):  # 2708
        mt[g] = np.eye(adj_dict.shape[1])
        for _ in range(nhood):
            mt[g] = np.matmul(mt[g], (adj_dict[g] + np.eye(adj_dict.shape[1])))
        for i in range(sizes[g]):
            for j in range(sizes[g]):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0
    return -1e9 * (1.0 - mt)


###############################################
# This section of code adapted from tkipf/gcn #
###############################################

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data_0():
    """Load gat."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']  # ind.cora.test.index left out
    objects = []
    for i in range(len(names)):
        with open("{}/ind.{}.{}".format(datapath, dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    print(f'This is graph {graph},{len(graph)}')

    test_idx_reorder = parse_index_file("{}/ind.{}.test.index".format(datapath, dataset))

    test_idx_range = np.sort(test_idx_reorder)  # test_idx_range =  [1708,1709,...,2707] 1000 entries

    if dataset == 'citeseer':
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()

    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj_dict = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    print(f'adj.shape: {adj_dict.shape}')
    print(f'features.shape: {features.shape}')

    return adj_dict, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def load_data():
    """限制读取最大节点"""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']  # ind.cora.test.index left out
    objects = []
    for i in range(len(names)):
        with open("{}/ind.{}.{}".format(datapath, dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph_0 = tuple(objects)
    print(f'This is graph_0 {graph_0},{len(graph_0)}')
    # 过滤节点
    graph = defaultdict(list)
    for node, neighbors in graph_0.items():
        print(node, neighbors)
        if node > node_max:
            continue

        graph[node] = [ngb for ngb in neighbors if ngb <= node_max]
    print(f'This is graph {graph},{len(graph)}')
    # 过滤节点

    test_idx_reorder = parse_index_file("{}/ind.{}.test.index".format(datapath, dataset))

    test_idx_range_0 = np.sort(test_idx_reorder)  # test_idx_range =  [1708,1709,...,2707] 1000 entries

    if dataset == 'citeseer':
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range_0 - min(test_idx_range_0), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range_0 - min(test_idx_range_0), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()

    # 过率特征
    features = features[0:node_max+1]
    # test_idx_range = np.array([idx for idx in list(test_idx_range_0) if idx <= node_max])
    #
    # features[test_idx_reorder, :] = features[test_idx_range, :]
    adj_dict = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    # labels[test_idx_reorder, :] = labels[test_idx_range, :]
    labels = labels[0:node_max+1]

    # idx_test = test_idx_range.tolist()
    idx_train = range(500)
    idx_val = range(500, 700)
    idx_test = range(700, 800)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    print(f'adj.shape: {adj_dict.shape}')
    print(f'features.shape: {features.shape}')

    return adj_dict, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def load_random_data(size):
    adj = sp.random(size, size, density=0.002)  # density similar to cora
    features = sp.random(size, 1000, density=0.015)
    int_labels = np.random.randint(7, size=(size))
    labels = np.zeros((size, 7))  # Nx7
    labels[np.arange(size), int_labels] = 1

    train_mask = np.zeros((size,)).astype(bool)
    train_mask[np.arange(size)[0:int(size / 2)]] = 1

    val_mask = np.zeros((size,)).astype(bool)
    val_mask[np.arange(size)[int(size / 2):]] = 1

    test_mask = np.zeros((size,)).astype(bool)
    test_mask[np.arange(size)[int(size / 2):]] = 1

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    # sparse NxN, sparse NxF, norm NxC, ..., norm Nx1, ...
    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        # 返回稀疏矩阵中有值的行列（索引）
        coords = np.vstack((mx.row, mx.col)).transpose()
        # 返回矩阵中的值
        values = mx.data
        # 返回矩阵的shape
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def standardize_data(f, train_mask):
    """Standardize feature matrix and convert to tuple representation"""
    # standardize gat
    f = f.todense()
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = f[:, np.squeeze(np.array(sigma > 0))]
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = (f - mu) / sigma
    return f


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    # 归一化
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def preprocess_adj_bias(adj):
    num_nodes = adj.shape[0]
    adj = adj + sp.eye(num_nodes)  # self-loop
    adj[adj > 0.0] = 1.0
    if not sp.isspmatrix_coo(adj):
        adj = adj.tocoo()
    adj = adj.astype(np.float32)
    indices = np.vstack(
        (adj.col, adj.row)).transpose()  # This is where I made a mistake, I used (adj.row, adj.col) instead

    return tf.SparseTensor(indices=indices, values=adj.data, dense_shape=adj.shape)


if __name__ == "__main__":
    # 加载数据
    # adj_dict:  sparse matrix，边表信息。 2708x2708
    # features：节点信息，2708x1433
    # y_train：标签信息
    # train_mask：哪些是训练样本的标志
    adj_dict, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data()
