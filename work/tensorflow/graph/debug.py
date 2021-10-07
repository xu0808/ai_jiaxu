#!/usr/bin/env python
# coding: utf-8
# 调试脚本

import networkx as nx
import config
import numpy as np
# import tf_geometric as tfg
import tensorflow as tf


if __name__ == "__main__":
    # G = nx.read_edgelist(config.wiki_edge_file, create_using=nx.DiGraph(),
    #                      nodetype=None, data=[('weight', int)])
    #
    # t, v, test = tfg.datasets.ppi.PPIDataset().load_data()
    # #
    # # i = 1
    # # for node in G.nodes():
    # #     print(node)
    # #     for nbr in G.neighbors(node):
    # #         print(G[node][nbr].get('weight', 1.0))
    # #         break
    # #
    # #     break
    # #     # unnormalized_probs = [G[node][nbr].get('weight', 1.0) for nbr in G.neighbors(node)]
    # #     # print(unnormalized_probs)
    #
    # shuffle_indices = np.random.permutation(np.arange(G.number_of_edges()))
    # print(shuffle_indices)
    # labels = ['a','b','c']
    # classes = set(labels)
    # np.identity(3)
    # array([[1., 0., 0.],
    #        [0., 1., 0.],
    #        [0., 0., 1.]])


    # array = [1,0,1]
    # p = array.pop()
    # print([1] + [2,3])

    # x = tf.constant([1, 4])
    # y = tf.constant([2, 5])
    # z = tf.constant([3, 6])
    # a = tf.stack([x, y, z])
    # a
    # print(a)
    # b = tf.stack([x, y, z], axis=1)
    # b
    # print(b)
    # c = tf.stack([x, y, z], axis=-1)
    # c
    # print(c)

    classes = set([5,0,1,3,1,3])
    # np.identity(3)
    # array([[1., 0., 0.],
    #        [0., 1., 0.],
    #        [0., 0., 1.]])
    classes_dict = {c: i for i, c in enumerate(classes)}
    print(classes_dict)

