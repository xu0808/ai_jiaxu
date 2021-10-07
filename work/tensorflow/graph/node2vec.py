#!/usr/bin/env python
# coding: utf-8
# Node2vec
# DeepWalk、Node2vec、LINE
# https://www.cnblogs.com/Lee-yl/p/12670515.html
"""
相对于DeepWalk, node2vec的改进主要是对基于随机游走的采样策略的改进
node2vec是结合了BFS和DFS的Deepwalk改进的随机游走算法
"""

from joblib import Parallel, delayed
from gensim.models import Word2Vec
import networkx as nx
import config
import itertools
import random
from alias import alias_sample, create_alias_table


class RandomWalker:
    def __init__(self, G, p=1, q=1):
        """
        param p: 控制重复访问刚刚访问过的顶点的概率。
        param q: 控制着游走是向外还是向内
                 若q<1随机游走倾向于访问接近t的顶点(偏向BFS)。
                 若q>1倾向于访问远离t的顶点(偏向DFS)。
        """
        self.G = G
        self.p = p
        self.q = q

    # 并行分区分配规则
    def part_num(self, num, workers):
        # 每一分区取模数个
        parts = [num // workers] * workers
        rod = num % workers
        # 余数单独一个分区
        return parts if rod == 0 else parts + [rod]

    def preprocess_transition_probs(self):
        """
        迁移概率别名表
        alias_nodes 首节点转移概率的别名表
        alias_edges 上一次访问顶点t当前访问顶点为v时到下一个顶点x转移概率的alias表。
        """
        alias_nodes = {}
        # 所有节点
        for node in self.G.nodes():
            # 所有边权重
            probs_0 = [self.G[node][nbr].get('weight', 1.0) for nbr in G.neighbors(node)]
            # 所有边权重总和
            probs_sum = sum(probs_0)
            # 所有边转移概率
            probs = [float(u_prob)/probs_sum for u_prob in probs_0]
            # probs, alias
            alias_nodes[node] = create_alias_table(probs)

        alias_edges = {}
        # 所有边
        for edge in G.edges():
            # 上一次访问顶点t,当前访问顶点为v
            t, v = edge[0], edge[1]
            probs_0 = []
            # 所有邻节点
            for x in G.neighbors(v):
                # 边权重
                weight = G[v][x].get('weight', 1.0)  # w_vx
                # 不同类型节点的边权重
                # 1、返回t的权重
                if x == t:  # d_tx == 0
                    prob_0 = weight / self.p
                # 2、返回t邻节点的权重
                elif G.has_edge(x, t):  # d_tx == 1
                    prob_0 = weight
                # 3、返回他和t邻节点以外的权重
                else:  # d_tx > 1
                    prob_0 = weight / self.q
                probs_0.append(prob_0)

            # 所有边权重总和
            probs_sum = sum(probs_0)
            # 所有边转移概率
            probs = [float(u_prob)/probs_sum for u_prob in probs_0]
            # probs, alias
            alias_edges[edge] = create_alias_table(probs)

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

    # 随机游走，walk_length为游走长度，start_node为开始节点
    def deepwalk_walk(self, walk_length, start_node):
        walk = [start_node]
        while len(walk) < walk_length:
            # 序列最后一个
            cur = walk[-1]
            # 所有邻节点
            cur_nbrs = list(self.G.neighbors(cur))
            # 随机取邻节点，不存在则直接退出
            if len(cur_nbrs) > 0:
                walk.append(random.choice(cur_nbrs))
            else:
                break
        return walk

    # 随机游走，walk_length为游走长度，start_node为开始节点
    def node2vec_walk(self, walk_length, start_node):
        walk = [start_node]
        while len(walk) < walk_length:
            # 序列最后一个
            cur = walk[-1]
            # 所有邻节点
            cur_nbrs = list(self.G.neighbors(cur))
            if len(cur_nbrs) > 0:
                # 取采样别名表
                # 是否首节点
                if len(walk) == 1:
                    alias_table = self.alias_nodes[cur]
                else:
                    alias_table = self.alias_edges[(walk[-2], cur)]
                # 随机采样结果
                sample_index = alias_sample(alias_table[0], alias_table[1])
                # 序列添加采样节点
                walk.append(cur_nbrs[sample_index])
            else:
                break

        return walk

    # 产生图所有节点的num_walks个随机序列
    def _simulate_walks(self, nodes, num_walks, walk_length):
        walks = []
        for _ in range(num_walks):
            random.shuffle(nodes)
            for v in nodes:
                # 序列只有1个节点时，直接使用deepwalk
                if self.p == 1 and self.q == 1:
                    walks.append(self.deepwalk_walk(walk_length=walk_length, start_node=v))
                # 当序列多于2个节点时，使用node2vec_walk
                else:
                    walks.append(self.node2vec_walk(walk_length=walk_length, start_node=v))
        return walks

    # 并行执行_simulate_walks，并将结果合并
    # num_walks为产生多少个随机游走序列
    # walk_length为游走序列长度
    def simulate_walks(self, num_walks, walk_length, workers=1, verbose=0):
        G = self.G
        nodes = list(G.nodes())
        results = Parallel(n_jobs=workers, verbose=verbose)(
            delayed(self._simulate_walks)(nodes, num, walk_length) for num in
            self.part_num(num_walks, workers))

        walks = list(itertools.chain(*results))
        return walks


class Node2Vec:
    def __init__(self, graph, walk_length, num_walks, p=1.0, q=1.0, workers=1):
        self.graph = graph
        self._embeddings = {}
        # 采样
        self.walker = RandomWalker(graph, p=p, q=q, )

        # 构造别名表
        print("Preprocess transition probs...")
        self.walker.preprocess_transition_probs()

        # 采样
        self.sentences = self.walker.simulate_walks(
            num_walks=num_walks, walk_length=walk_length, workers=workers, verbose=1)

    def train(self, embed_size=128, window_size=5, workers=3, iter=5, **kwargs):
        kwargs["sentences"] = self.sentences
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["size"] = embed_size
        kwargs["sg"] = 1
        kwargs["hs"] = 0  # node2vec not use Hierarchical Softmax
        kwargs["workers"] = workers
        kwargs["window"] = window_size
        kwargs["iter"] = iter

        # Word2vec
        print("Learning embedding vectors...")
        model = Word2Vec(**kwargs)
        print("Learning embedding vectors done!")
        self.w2v_model = model
        return model

    def get_embeddings(self,):
        if self.w2v_model is None:
            print("model not train")
            return {}

        self._embeddings = {}
        for word in self.graph.nodes():
            self._embeddings[word] = self.w2v_model.wv[word]

        return self._embeddings


if __name__ == "__main__":
    G = nx.read_edgelist(config.wiki_edge_file,
                         create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])
    model = Node2Vec(G, walk_length=10, num_walks=80, p=0.25, q=4, workers=1)
    model.train(window_size=5, iter=3)
    embeddings = model.get_embeddings()
    print('embeddings:')
    print(embeddings)