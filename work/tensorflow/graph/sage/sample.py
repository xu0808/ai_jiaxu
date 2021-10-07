#!/usr/bin/env python
# coding: utf-8
# 节点采样
import numpy as np
from functools import reduce


def sampling(nodes, sample_size, neighbors_dict):
    """根据源节点采样指定数量的邻居节点，注意使用的是有放回的采样；
       某个节点的邻居节点数量少于采样数量时，采样结果出现重复的节点
    param nodes {list, ndarray} -- 源节点列表
    param sample_size {int} -- 需要采样的节点数
    param neighbors_dict {dict} -- 节点到其邻居节点的映射表
    return np.ndarray -- 采样结果构成的列表
    """
    results = []
    for node_id in nodes:
        # 从节点的邻居中进行有放回地进行采样
        res = np.random.choice(neighbors_dict[node_id], size=(sample_size,))
        results.append(res)
    return np.asarray(results).flatten()


def multihop_sampling(nodes, sample_sizes, neighbors_dict):
    """根据源节点进行多阶采样
    param nodes {list, np.ndarray} -- 源节点id
    param sample_sizes {list of int} -- 每阶需要采样的个数
    param neighbors_dict {dict} -- 节点到其邻居节点的映射
    return [list of ndarray] -- 每阶采样的总结果
    """
    # 源节点集合
    sampling_result = [nodes]  # 首先包含源节点
    # k阶采样
    # 先对源节点进行1阶采样 在与源节点距离为1的节点中采样hopk_num个节点；
    # 再对源节点进行2阶采样，即对源节点的所有1阶邻居进行1阶采样
    for k, hopk_num in enumerate(sample_sizes):
        pre_nodes = sampling_result[k]
        # 针对上一阶节点采样hopk_num个
        hopk_result = sampling(pre_nodes, hopk_num, neighbors_dict)
        sampling_result.append(hopk_result)

    # 每阶采样的总结果
    return sampling_result


def neighbors(nodes, neigh_dict):
    """
    节点集合的所有邻节点(去重)
    """
    return np.unique(np.concatenate([neigh_dict[n] for n in nodes]))


# 无监督学习时：通过边的信息进行负采样
def neg_sampling(batch_edge, nodes, neighbors_dict, sample_sizes, neg_size):
    # batch_a边的起点集合，batch_b边的终点集合
    batch_a, batch_b = batch_edge.transpose()  # 数组转置
    neighbors_batch_a = neighbors(batch_a, neighbors_dict)
    neighbors_batch_b = neighbors(batch_b, neighbors_dict)
    # 所有的负采样的样本(连续排除)
    possible_negs = reduce(np.setdiff1d, (nodes, batch_a, neighbors_batch_a, batch_b, neighbors_batch_b))
    # 负采样样本
    batch_n = np.random.choice(possible_negs, min(neg_size, len(possible_negs)), replace=False)
    # 本批节点
    batch_node = np.unique(np.concatenate((batch_a, batch_b, batch_n)))
    # 正负节点一起采样
    batch_samples = multihop_sampling(batch_node, sample_sizes, neighbors_dict)
    return batch_samples, batch_a, batch_b, batch_n

