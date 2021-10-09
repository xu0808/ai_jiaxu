#!/usr/bin/env python
# coding: utf-8
# 原始行为日志和商品信息数据处理

import pandas as pd
import numpy as np
import time
import networkx as nx
from walker import RandomWalker
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict

# 数据文件配置
data_dir = '../data'
# 行为日志
action_file = '%s/action_head.csv' % data_dir
# 商品信息
product_file = '%s/jdata_product.csv' % data_dir
# 图结构
graph_file = '%s_1/graph.csv' % data_dir
# 特征数据
feature_file = '%s_1/feature.csv' % data_dir
# 采样所得所有组合
sample_pairs_file = '%s_1/sample_pairs.csv' % data_dir


p = 0.25
q = 2
num_walks = 10
walk_length = 10
window_size = 5


def agg_session(action_data, use_type=None):
    """
    用户聚合、时间排序
    """
    if use_type is None:
        use_type = [1, 2, 3, 5]
    action_data = action_data[action_data['type'].isin(use_type)]
    action_data = action_data.sort_values(by=['user_id', 'action_time'], ascending=True)
    group_action_data = action_data.groupby('user_id').agg(list)
    session_list = group_action_data.apply(split_session, axis=1)
    return session_list.to_numpy()


def split_session(data, time_cut=30, cut_type=2):
    """
    按照切分时长和行为类型划分会话序列
    """
    sku_list = data['sku_id']
    time_list = data['action_time']
    type_list = data['type']
    session = []
    tmp_session = []
    sku_size = len(sku_list)
    for i, item in enumerate(sku_list):
        # 分割点（分割类型、序列末尾、间隔超时）
        if type_list[i] == cut_type or i == sku_size - 1 \
                or ((time_list[i+1] - time_list[i]).seconds/60 > time_cut):
            tmp_session.append(item)
            session.append(tmp_session)
            tmp_session = []
        else:
            tmp_session.append(item)
    return session


def sample_pairs(walks):
    all_pairs = []
    for walk in walks:
        for i in range(len(walk)):
            # 当前位置前后窗口范围
            for j in range(i - window_size, i + window_size + 1):
                # 保证j在数组之内且不为当前位置
                if i == j or j < 0 or j >= len(walk):
                    continue
                else:
                    all_pairs.append([walk[i], walk[j]])
    return np.array(all_pairs, dtype=np.int32)


def create_graph():
    """
    从用户行为表生成图
    """
    # 1、读取行为数据
    print('read action gat start!')
    start_time = time.time()
    action_df = pd.read_csv(action_file, parse_dates=['action_time']).drop('module_id', axis=1).dropna()
    # 所有商品（去重）
    sku_unique = action_df['sku_id'].unique()
    sku_df = pd.DataFrame({'sku_id': list(sku_unique)})
    sku_lbe = LabelEncoder()
    sku_df['sku_id'] = sku_lbe.fit_transform(sku_df['sku_id'])
    action_df['sku_id'] = sku_lbe.transform(action_df['sku_id'])
    print('split session seq end!  %.2f s' % (time.time() - start_time))

    # 2、会话拆分
    print('split session seq start!')
    start_time = time.time()
    sessions_list = agg_session(action_df, use_type=[1, 2, 3, 5])
    session_list = []
    for item_list in sessions_list:
        for s in item_list:
            if len(s) > 1:
                session_list.append(s)

    print('split session seq end!  %.2f s' % (time.time() - start_time))

    # 3、生成图
    # 统计边
    print('create graph start!')
    start_time = time.time()
    node_pair = defaultdict(int)
    for s in session_list:
        for i in range(1, len(s)):
            node_pair[(s[i - 1], s[i])] += 1

    in_node, out_node, weight = [], [], []
    for edge in node_pair.keys():
        in_node.append(edge[0])
        out_node.append(edge[1])
        weight.append(node_pair.get(edge))

    graph_df = pd.DataFrame({'in_node': in_node, 'out_node': out_node, 'weight': weight})
    graph_df.to_csv(graph_file, sep=' ', index=False, header=False)
    print('create graph  end!  %.2f s' % (time.time() - start_time))

    # 4、边信息
    print('add side info start!')
    start_time = time.time()
    product_df = pd.read_csv(product_file).drop('market_time', axis=1).dropna()
    sku_df['sku_id'] = sku_lbe.inverse_transform(sku_df['sku_id'])
    print("sku nums: " + str(sku_df.count()))
    sku_side_info = pd.merge(sku_df, product_df, on='sku_id', how='left').fillna(0)
    # id2index
    for feat in product_df.columns:
        if feat != 'sku_id':
            lbe = LabelEncoder()
            sku_side_info[feat] = lbe.fit_transform(sku_side_info[feat])
        else:
            sku_side_info[feat] = sku_lbe.transform(sku_side_info[feat])
    print('add side info end!  %.2f s' % (time.time() - start_time))
    sku_side_info = sku_side_info.sort_values(by=['sku_id'], ascending=True)
    sku_side_info.to_csv(feature_file, index=False, header=False, sep='\t')


def sample():
    # 1、读取图
    print('read graph start!')
    start_time = time.time()
    G = nx.read_edgelist(graph_file, create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])
    print('read graph end!  %.2f s' % (time.time() - start_time))

    # 2、创建迁移概率别名表
    print('create alias table start!')
    start_time = time.time()
    walker = RandomWalker(G, p=p, q=q)
    walker.preprocess_transition_probs()
    print('create alias table end!  %.2f s' % (time.time() - start_time))

    # 3、创建迁移概率别名表
    print('sample walk start!')
    start_time = time.time()
    sample_0 = walker.simulate_walks(num_walks=num_walks, walk_length=walk_length, workers=4, verbose=1)
    # 过滤小于3个序列
    walks = list(filter(lambda x: len(x) > 2, sample_0))
    print('sample walk e end!  %.2f s' % (time.time() - start_time))

    # 4、样本内所有窗口内组合记录
    all_pairs = sample_pairs(walks)  # # , window_size)

    np.savetxt(sample_pairs_file, X=all_pairs, fmt="%d", delimiter=" ")


if __name__ == '__main__':
    # 1、从用户行为表生成图
    # create_graph()

    # 2、生成采样序列
    sample()

