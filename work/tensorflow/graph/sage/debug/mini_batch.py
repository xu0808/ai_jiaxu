#!/usr/bin/env python3
import numpy as np
import collections
from functools import reduce

# 数据模型
fields_node = ['src_nodes', 'neighbors_index', 'nodes_index',  'adj_mats']
Batch_node = collections.namedtuple('Batch_node', fields_node)
fields_edge = ['src_nodes', 'neighbors_index', 'nodes_index',  'adj_mats', 'batch_a', 'batch_b', 'batch_n']
Batch_edge = collections.namedtuple("Batch_edge", fields_edge)


# 无监督学习时：通过边的信息进行负采样
def build_batch_from_edges(batch_edge, nodes, neighbors_dict, sample_sizes, neg_size):
    # batch_a边的起点集合，batch_b边的终点集合
    batch_a, batch_b = batch_edge.transpose()  # 数组转置
    neighbors_batch_a = neighbors(batch_a, neighbors_dict)
    neighbors_batch_b = neighbors(batch_b, neighbors_dict)
    # 所有的负采样的样本(连续排除)
    # reduce(fun,seq): sequence连续使用function
    # np.setdiff1d(arr1,arr2)：返回存在arr1中不存在于arr2中的元素(去重)。
    possible_negs = reduce(np.setdiff1d, (nodes, batch_a, neighbors_batch_a, batch_b, neighbors_batch_b))
    # 负采样样本
    batch_n = np.random.choice(possible_negs, min(neg_size, len(possible_negs)), replace=False)
    # 本批节点
    batch_node = np.unique(np.concatenate((batch_a, batch_b, batch_n)))
    # order does matter, in the model, use tf.gather on this
    batch_a_index = np.searchsorted(batch_node, batch_a)
    # order does matter, in the model, use tf.gather on this
    batch_b_index = np.searchsorted(batch_node, batch_b)
    # order does not matter, in the model, use tf.boolean_mask on this
    # 测试一维数组的每个元素是否也存在于第二个数组中。
    # 返回一个与ar1长度相同的布尔数组，该数组为true，其中ar1的元素位于ar2中，否则为False。
    batch_n_index = np.in1d(batch_node, batch_n)
    # 正负节点一起采样
    batch = build_batch_from_nodes(batch_node, neighbors_dict, sample_sizes)
    return Batch_edge(batch.src_nodes, batch.neighbors_index, batch.nodes_index, batch.adj_mats, batch_a_index, batch_b_index, batch_n_index)


def build_batch_from_nodes(batch_node, neigh_dict, sample_sizes):
    nodes_list = [batch_node]
    nodes_index = []
    neighbors_index = []
    adj_mats = []

    max_node_id = max(list(neigh_dict.keys()))
    for sample_size in reversed(sample_sizes):
        sample_nodes, node_index, neighbor_index, adj_mat = adj_matrix(nodes_list[-1], neigh_dict, sample_size, max_node_id)
        nodes_list.append(sample_nodes)
        neighbors_index.append(neighbor_index)
        nodes_index.append(node_index)
        adj_mats.append(adj_mat)

    # 源节点只取最后一层
    src_nodes = nodes_list[-1]
    # 创建Batch对象
    return Batch_node(src_nodes, neighbors_index, nodes_index,  adj_mats)


def adj_matrix(batch_node, neighbors_dict, sample_size, max_node_id):
    """
    邻阶矩阵
    """
    # 随机从边节点中采样出sample_size个节点，若边个数小于sample_size全部采样
    def sample(ns):
        # ns=[9,100,200,102] 某个节点的边节点数组
        return np.random.choice(ns, min(len(ns), sample_size), replace=False)

    # 生成len=max_node_id + 1的全零vector,将采样的节点的位置标识为 1.
    def vectorize(ns):
        v = np.zeros(max_node_id + 1, dtype=np.float32)
        v[ns] = 1
        return v

    # sample neighbors
    sample_node = [sample(neighbors_dict[n]) for n in batch_node]
    # 生成vector(将采样的节点的位置标识为 1.其余位置=0)
    sample_node_mark = np.stack([vectorize(nodes) for nodes in sample_node])
    # 节点是否被采样标记
    sample_array = np.any(sample_node_mark.astype(np.bool), axis=0)
    # 被采样过节点在当前样本中是否被采样
    # 维度降低为（max_node_id + 1）-> 被采样过节点数
    sample_mask = sample_node_mark[:, sample_array]
    # 每个样本采样总和
    sample_mask_sum = np.sum(sample_mask, axis=1, keepdims=True)
    # 邻阶矩阵
    adj_mat = sample_mask / sample_mask_sum
    # 采样到的节点的序号(维度缩减)
    neighbor = np.arange(sample_array.size)[sample_array]

    # 所有节点
    sample_nodes = np.union1d(batch_node, neighbor)
    neighbor_index = np.searchsorted(sample_nodes, neighbor)
    node_index = np.searchsorted(sample_nodes, batch_node)
    return sample_nodes, node_index, neighbor_index, adj_mat


def neighbors(nodes, neigh_dict):
    """
    节点集合的所有邻节点(去重)
    """
    return np.unique(np.concatenate([neigh_dict[n] for n in nodes]))
