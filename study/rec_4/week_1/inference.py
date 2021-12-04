#!/usr/bin/env python
# coding: utf-8
# 推理服务

import pandas as pd
import numpy as np
import os
import params_server
import reader
# pip install faiss-cpu
import faiss
np.printoptions()

if __name__ == "__main__":
    # 读取向量
    emb_file = os.path.join(reader.data_dir, 'rating_emb.csv')
    emb_df = pd.read_csv(emb_file)
    print(emb_df[0:5])
    keys, values = emb_df.values[:, 0], emb_df.values[:, 1]
    vecs = []
    for value in list(values):
        value = value[1:-1].replace('\n', '')
        vec = []
        for v in value.split(' '):
            # 小数点、负数、科学计数法
            if v.replace('.', '').replace('e', '').replace('-', '').isnumeric():
                vec.append(float(v))
        vecs.append(vec)

    # 近邻查找
    index = faiss.IndexFlatL2(len(vecs[0]))
    emb = np.array(vecs).astype(np.float32)
    index.add(emb)
    # d top5的距离数组，i top5的索引数组
    # 基础验证最近的为自己距离为0
    d, i = index.search(emb, 5)
    sim_dic = {}
    for index in range(i.shape[0]):
        # 第一个为自己，后面为近邻
        sim_dic[keys[i[index][0]]] = keys[i[index][1:]]

    print('test')



