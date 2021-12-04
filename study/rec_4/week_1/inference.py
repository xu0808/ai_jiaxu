#!/usr/bin/env python
# coding: utf-8
# 推理服务

import tensorflow as tf
import pandas as pd
import numpy as np
import os
import params_server
import tf_record
import reader
# pip install faiss-cpu
import faiss

np.printoptions()

if __name__ == "__main__":
    # 1、读取向量
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
    emb_dict = {k: v for (k, v) in zip(keys, vecs)}

    # 分离用户和电影向量
    keys = ['user_id', 'movie_id', 'rating']
    types = [tf.int64, tf.int64, tf.int64]
    # 分批读出每个特征
    data_set = tf_record.read('rating', keys, types, batch_size=200)
    user_key, user_emb = [], []
    movie_key, movie_emb = [], []
    for user_id, movie_id, _ in data_set:
        for key in user_id.numpy():
            if key in user_key:
                continue
            else:
                user_key.append(key)
                user_emb.append(emb_dict[key])
        for key in movie_id.numpy():
            if key in movie_key:
                continue
            else:
                movie_key.append(key)
                movie_emb.append(emb_dict[key])

    # 2、近邻查找
    # 2-1、i2i
    emb_i2i = np.array(movie_emb).astype(np.float32)
    index_i2i = faiss.IndexFlatL2(emb_i2i.shape[1])
    index_i2i.add(emb_i2i)
    # d top5的距离数组，i top5的索引数组
    # 基础验证最近的为自己距离为0
    d, i = index_i2i.search(emb_i2i, 6)
    sim_dic_i2i = {}
    for index in range(i.shape[0]):
        top5 = np.array(movie_key)[i[index]]
        # 第一个为自己，后面为近邻
        sim_dic_i2i[top5[0]] = top5[1:]
    # 2-2、u2i
    index_u2i = faiss.IndexFlatL2(emb_i2i.shape[1])
    index_u2i.add(emb_i2i)
    emb_u2i = np.array(user_emb).astype(np.float32)
    # 为每个用户查找5个电影
    d, i = index_u2i.search(emb_u2i, 5)
    sim_dic_u2i = {}
    for index in range(i.shape[0]):
        top5 = np.array(movie_key)[i[index]]
        sim_dic_u2i[user_key[index]] = top5
    print('i2i and u2i finish!')
