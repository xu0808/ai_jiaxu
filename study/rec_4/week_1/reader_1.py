#!/usr/bin/env python
# coding: utf-8
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import tf_record

data_dir = 'D:\\study\\rec_4\\data\\1'


def read_rating():
    # 数据文件
    ratings_file = os.path.join(data_dir, 'ratings.dat')
    # 不能直接使用'::'分割
    rating_df = pd.read_csv(ratings_file, encoding="utf-8", header=None, sep=':')
    print('rating_df:', rating_df)
    # 取出数组
    rating_0 = rating_df.values.astype(np.int32)
    print('rating_0.shape:', rating_0.shape)
    # 取出user_id， movie_id， rating
    rating = rating_0[:, [0, 2, 4]]
    print('rating.shape:', rating.shape)
    print('rating:', rating)
    # 只取出用户小于2000的记录
    rating_1 = rating[rating[:, 0] < 2000]
    print('rating_1.shape:', rating_1.shape)
    print('rating_1:', rating_1)
    # hash离散化
    # 作用：a.更好的稀疏表示；b. 具有一定的特征压缩；c.能够方便模型的部署
    rating_hash = []
    for i in range(rating_1.shape[0]):
        hash_line = [hash('user_id=%d' % rating_1[i, 0]), hash('movie_id=%d' % rating_1[i, 1]), rating_1[i, 2]]
        rating_hash.append(hash_line)
    print('rating_hash.shape: (%d, %d)' % (len(rating_hash), len(rating_hash[0])))
    print('rating_hash last 5:', rating_hash[-5:])
    return rating_hash


def write_recod():
    ratings_data = read_rating()
    keys = ['user_id', 'movie_id', 'rating']
    types = ['int64', 'int64', 'int64']
    tf_record.write('rating', keys, types, ratings_data)


def read_recod():
    keys = ['user_id', 'movie_id', 'rating']
    types = [tf.int64, tf.int64, tf.int64]
    # 分批读出每个特征
    data_set = tf_record.read('rating', keys, types)
    data_total = 0
    batch_num = 0
    for user_id, movie_id, rating in data_set:
        if batch_num == 0:
            print('user_id = ', user_id)
            print('movie_id = ', movie_id)
            print('rating = ', rating)
            batch_size = user_id.shape[0]
        batch_num += 1
        data_total += user_id.shape[0]

    # 样本257488，每批200，共1288批
    print('data_set batch_size = ', batch_size)
    print('data_set batch_num = ', batch_num)
    print('data_set data_total = ', data_total)


if __name__ == '__main__':
    # read_rating()
    # write_recod()
    read_recod()

