#!/usr/bin/env python
# coding: utf-8
# 数据读取

import os
import pandas as pd
from collections import defaultdict

data_dir = 'D:\\study\\rec_4\\data\\2\\data'
tf_record_dir = os.path.join(data_dir, 'tf_record')


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


if __name__ == '__main__':
    read_log()