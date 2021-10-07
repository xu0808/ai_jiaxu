#!/usr/bin/env python
# coding: utf-8
# DeepWalk
# DeepWalk、Node2vec、LINE
# https://www.cnblogs.com/Lee-yl/p/12670515.html
"""
Random Walk：一种可重复访问已访问节点的深度优先遍历算法。
给定当前访问起始节点，从其邻居中随机采样节点作为下一个访问节点，
重复此过程，直到访问序列长度满足预设条件。
Word2vec：接着利用skip-gram模型进行向量学习。
"""

from joblib import Parallel, delayed
import itertools
import random
from gensim.models import Word2Vec
import networkx as nx
import config


class RandomWalker:
    """
    给定当前访问起始节点，从其邻居中随机采样节点作为下一个访问节点，
    重复此过程，直到访问序列长度满足预设条件。
    """
    def __init__(self, G):
        self.G = G

    # 并行分区分配规则
    def part_num(self, num, workers):
        # 每一分区取模数个
        parts = [num // workers] * workers
        rod = num % workers
        # 余数单独一个分区
        return parts if rod == 0 else parts + [rod]

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

    # 产生图所有节点的num_walks个随机序列
    def _simulate_walks(self, nodes, num_walks, walk_length):
        walks = []
        for _ in range(num_walks):
            # 随机打撒节点
            random.shuffle(nodes)
            for v in nodes:
                walks.append(self.deepwalk_walk(walk_length, start_node=v))
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


class DeepWalk:
    def __init__(self, graph, walk_length, num_walks, workers=1):
        self.graph = graph
        self.w2v_model = None
        self._embeddings = {}
        self.walker = RandomWalker(graph)
        self.sentences = self.walker.simulate_walks(
            num_walks=num_walks, walk_length=walk_length, workers=workers, verbose=1)

    def train(self, embed_size=128, window_size=5, workers=3, iter=5, **kwargs):
        kwargs["sentences"] = self.sentences
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["size"] = embed_size
        kwargs["sg"] = 1  # skip gram
        kwargs["hs"] = 1  # deepwalk use Hierarchical Softmax
        kwargs["workers"] = workers
        kwargs["window"] = window_size
        kwargs["iter"] = iter
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
    model = DeepWalk(G, walk_length=10, num_walks=80, workers=1)
    model.train(window_size=5, iter=3)
    embeddings = model.get_embeddings()
    print('embeddings:')
    print(embeddings)