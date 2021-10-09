#!/usr/bin/env python3

import numpy as np
from collections import defaultdict

datapath = 'E:\\workspace\\ai_jiaxu\\data\\graph\\cora'


def load_cora():
    node_num = 2708
    num_feats = 1433
    feature = np.zeros((node_num, num_feats), dtype=np.float32)
    label = np.empty((node_num, 1), dtype=np.int64)
    node_map = {}
    label_map = {}
    with open(datapath + "/cora.content") as fp:
        for i, line in enumerate(fp):
            info = line.strip().split()
            feature[i, :] = list(map(float, info[1:-1]))
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            label[i] = label_map[info[-1]]

    neigh_dict = defaultdict(set)
    with open(datapath + "/cora.cites") as fp:
        for line in fp:
            info = line.strip().split()
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            neigh_dict[paper1].add(paper2)
            neigh_dict[paper2].add(paper1)

    neigh_dict = {k: np.array(list(v)) for k, v in neigh_dict.items()}
    
    return node_num, feature, label, len(label_map), neigh_dict


def debug():
    node_num, feature, label, class_num, neigh_dict = load_cora()
    # node_num = 2708, class_num = 7, feature shape = (2708, 1433), label shape = (2708, 1)
    print('node_num = {}, class_num = {}, feature shape = {}, label shape = {}'
          .format(node_num, class_num, feature.shape, label.shape))


if __name__ == "__main__":
    debug()
