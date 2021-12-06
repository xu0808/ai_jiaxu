#!/usr/bin/env python
# coding: utf-8
import os
import pandas as pd
import numpy as np

data_dir = 'D:\\study\\rec_4\\data\\1'
tf_record_dir = os.path.join(data_dir, 'tf_record')


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


if __name__ == '__main__':
    read_rating()
