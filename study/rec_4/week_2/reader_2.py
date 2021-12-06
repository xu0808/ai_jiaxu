#!/usr/bin/env python
# coding: utf-8
import os
import pandas as pd
import numpy as np

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
    # 添加分类列
    click_log = np.c_[click_log, [categories[i] for i in click_log[:, 1]]]
    print('click_log:', click_log[0:5])


if __name__ == '__main__':
    read_log()
