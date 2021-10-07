#!/usr/bin/env python
# coding: utf-8
# 图采样

import itertools
import random
from joblib import Parallel, delayed
from alias import alias_sample, normalized_alias_table
from utils import partition_num


class RandomWalker:
    def __init__(self, G, p=1, q=1):
        """
        随机游走
        t(上一节点), v(当前节点), x(采样节点)
        p,q参数控制深度优先和广度优先
        """
        self.G = G
        self.p = p
        self.q = q
        self.alias_nodes = None
        self.alias_edges = None

    def neighbors_weight(self, v):
        """ 邻居节点权重 """
        return {x: self.G[v][x].get('weight', 1.0) for x in self.G.neighbors(v)}

    def edge_prob(self, t, x, weight):
        """ 边的实际游走概率 """
        # 1、返回上一节点(d_tx == 0)
        if x == t:
            return weight / self.p
        # 2、返回上一节点邻节点(d_tx == 1)
        if self.G.has_edge(x, t):
            return weight
        # 3、远处游走(d_tx > 1)
        return weight / self.q

    def preprocess_transition_probs(self):
        """ 转移概率别名表 """
        # 节点转移概率别名表
        alias_nodes = {}
        for node in self.G.nodes():
            # 邻居节点的权重
            weigh_dict = self.neighbors_weight(node)
            # 转移概率别名表
            alias_nodes[node] = normalized_alias_table(list(weigh_dict.values()))

        # 边转移概率别名表
        alias_edges = {}
        for edge in self.G.edges():
            [t, v] = edge
            # 邻居节点的权重
            weigh_dict = self.neighbors_weight(v)
            probs = []
            for x, weight in weigh_dict.items():
                probs.append(self.edge_prob(t, x, weight))
            alias_edges[edge] = normalized_alias_table(probs)
        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

    def deep_walk(self, walk_length, start_node):
        walk = [start_node]
        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(self.G.neighbors(cur))
            if len(cur_nbrs) > 0:
                # 完全随机
                walk.append(random.choice(cur_nbrs))
            else:
                break
        return walk

    def node2vec(self, walk_length, start_node):
        walk = [start_node]
        while len(walk) < walk_length:
            v = walk[-1]
            nbrs = list(self.G.neighbors(v))
            if len(nbrs) > 0:
                if len(walk) == 1:
                    alias = self.alias_nodes[v]
                else:
                    t = walk[-2]
                    alias = self.alias_edges[(t, v)]
                walk.append(nbrs[alias_sample(alias[0], alias[1])])
            else:
                break
        return walk

    def simulate_walks(self, num_walks, walk_length, workers=1, verbose=0):
        """ 多线程随机游走 """
        nodes = list(self.G.nodes())
        results = Parallel(n_jobs=workers, verbose=verbose, )(
            delayed(self._simulate_walks)(nodes, num, walk_length)
            for num in partition_num(num_walks, workers))

        walks = list(itertools.chain(*results))
        return walks

    def _simulate_walks(self, nodes, num_walks, walk_length,):
        walks = []
        for _ in range(num_walks):
            random.shuffle(nodes)
            for v in nodes:
                # deep walk
                if self.p == 1 and self.q == 1:
                    walks.append(self.deep_walk(walk_length=walk_length, start_node=v))
                # node2vec
                else:
                    walks.append(self.node2vec(walk_length=walk_length, start_node=v))
        return walks
