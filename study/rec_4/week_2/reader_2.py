#!/usr/bin/env python
# coding: utf-8
# 数据读取

import os
import pandas as pd
import numpy as np
import random
from collections import defaultdict
import tensorflow as tf
import tf_record

data_dir = 'D:\\study\\rec_4\\data\\2\\data'
features = 'user_id,article_id,environment,region'.split(',')
label = 'label'


def read_log():
    # 数据文件
    articles_file = os.path.join(data_dir, 'articles.csv')
    article_df = pd.read_csv(articles_file, encoding='utf-8')
    print('article_df:', article_df[:5])
    categories = article_df.values[:, 1]
    print('categories:', categories[0:5])

    click_log_file = os.path.join(data_dir, 'click_log.csv')
    click_log_df = pd.read_csv(click_log_file, encoding='utf-8')
    print('click_log_df:', click_log_df[:5])
    click_log = click_log_df.values[:, 0:3]
    print('click_log:', click_log[0:5])

    # 样本统计
    user_item_dict = defaultdict(list)
    item_user_dict = defaultdict(list)
    user_item_ts = defaultdict(int)
    for i in range(click_log.shape[0]):
        [user_id, item_id, ts] = click_log[i]
        user_item_dict[user_id].append(item_id)
        item_user_dict[item_id].append(user_id)
        user_item_ts['%d_%d' % (user_id, item_id)] = ts

    return user_item_dict, item_user_dict, user_item_ts, categories


def feature():
    # 数据文件
    # user_id,article_id,timestamp,environment,device_group,os,country,region,referrer_type
    click_log_file = os.path.join(data_dir, 'click_log.csv')
    click_log_df = pd.read_csv(click_log_file, encoding='utf-8')
    # 只取user_id,article_id,environment,region
    click_log_0 = click_log_df.values.astype(np.int32)
    print('click_log_0.shape:', click_log_0.shape)
    # 只取前2千条行为记录
    click_log_1 = click_log_0[0:2000, [0, 1, 3, 7]]
    print('click_log_1.shape:', click_log_1.shape)
    print('click_log_1:\n', click_log_1[-5:])

    # 文章清单
    article_list = list(np.unique(click_log_1[:, 1]))
    samples = []
    for i in range(click_log_1.shape[0]):
        log = list(click_log_1[i])
        samples.append(log + [1])
        # 每个样本取10个负样本
        for article_id in random.sample(article_list, 10):
            samples.append([log[0], article_id, log[2], log[3], 0])
    print('samples:\n', samples[:5])
    # user_id,article_id,environment,region
    sample_hash = []
    for i in range(len(samples)):
        hash_line = [hash('%s=%d' % (name, samples[i][index])) for index, name in enumerate(features)]
        sample_hash.append(hash_line + [samples[i][4]])
    print('sample_hash:\n', sample_hash[:5])
    return sample_hash


def write_recod():
    feature_hash = feature()
    keys = features + [label]
    types = ['int64']*4 + ['float']
    tf_record.write('click_log', keys, types, feature_hash)


def read_recod():
    keys = features + [label]
    types = [tf.int64]*4 + [tf.float32]
    # 分批读出每个特征
    data_set = tf_record.read('click_log', keys, types)
    data_total = 0
    batch_num = 0
    for user_id, article_id, _, _, y_true in data_set:
        if batch_num == 0:
            print('user_id = ', user_id)
            print('article_id = ', article_id)
            print('y_true = ', y_true)
            batch_size = user_id.shape[0]
        batch_num += 1
        data_total += user_id.shape[0]

    # 样本257488，每批200，共1288批
    print('data_set batch_size = ', batch_size)
    print('data_set batch_num = ', batch_num)
    print('data_set data_total = ', data_total)


if __name__ == '__main__':
    # read_log()
    # feature()
    # write_recod()
    read_recod()
