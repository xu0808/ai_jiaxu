#!/usr/bin/env python3

import numpy as np
from collections import defaultdict
from sklearn.preprocessing import StandardScaler

datapath = 'E:\\workspace\\ai_jiaxu\\data\\graph\\ppi'


def load_ppi():
    node_num = 14755
    num_feats = 50

    feat_data = np.load(datapath + "/toy-ppi-feats.npy")
    feat_data = feat_data.astype(np.float32)

    scaler = StandardScaler()
    scaler.fit(feat_data)
    feature = scaler.transform(feat_data)

    neigh_dict = defaultdict(set)
    with open(datapath + "/toy-ppi-walks.txt") as fp:
        for line in fp:
            info = line.strip().split()
            item1 = int(info[0])
            item2 = int(info[1])
            neigh_dict[item1].add(item2)
            neigh_dict[item2].add(item1)

    neigh_dict = {k: np.array(list(v)) for k, v in neigh_dict.items()}
 
    return node_num, feature, neigh_dict


def debug():
    node_num, feature, neigh_dict = load_ppi()
    # node_num = 14755, feature shape = (14755, 50)
    print('node_num = {}, feature shape = {}'.format(node_num, feature.shape))


if __name__ == "__main__":
    debug()
