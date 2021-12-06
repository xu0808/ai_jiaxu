#!/usr/bin/env python
# coding: utf-8
# ItemCF的主要思想是：给用户推荐之前喜欢物品的相似物品。模式：i2i
""" 文章与文章之间的相似性矩阵计算
    点击时间权重，其中的参数可以调节，点击时间相近权重大，相远权重小
    两篇文章的类别的权重，其中类别相同权重大
"""

import math
import traceback
import numpy as np
import pandas as pd
import warnings
from collections import defaultdict
import json

import reader_2


#


def item_cf_sim(user_item_dict, user_item_ts, categories, item_user_dict, top_n=20):
    """
     parma user_item_dict 用户点击文章列表字典
     parma user_item_ts   用户点击文章时间字典
     parma categories     文章分类列表
     parma item_user_dict  文章读者用户列表字典
     parma top_n 相似top数
     return  i2i_sim 物品相似度列表
    """
    # 计算物品相似度
    i2i_sim = defaultdict(dict)
    # 一、逐个用户计算物品相似度
    for user, items in user_item_dict.items():
        # 在基于商品的协同过滤优化的时候可以考虑时间因素
        for i in items:
            for j in items:
                if i == j:
                    continue
                # 1、点击时间权重，其中的参数可以调节，点击时间相近权重大，相远权重小
                ts_i = user_item_ts['%d_%d' % (user, i)]
                ts_j = user_item_ts['%d_%d' % (user, j)]
                ts_weight = np.exp(0.7 ** np.abs(ts_i - ts_j))

                # 2、两篇文章的类别的权重，其中类别相同权重大
                type_weight = 1.0 if categories[i] == categories[j] else 0.7
                # 3、考虑多种因素的权重计算最终的文章之间的相似度
                i2i_sim[i].setdefault(j, 0)
                i2i_sim[i][j] += ts_weight * type_weight / math.log(len(items) + 1)

    # 二、逐个用户计算物品相似度
    for i, sims in i2i_sim.items():
        i_count = len(item_user_dict[i])
        tmp_sim = {}
        for j, w in sims.items():
            tmp_sim[j] = w / math.sqrt(i_count * len(item_user_dict[j]))
        i2i_sim[i] = sorted(tmp_sim.items(), key=lambda kv: (kv[1], kv[0]))[:top_n]


if __name__ == '__main__':
    # 行为日志读取
    user_item_dict, item_user_dict, user_item_ts, categories = reader_2.read_log()
    # item_cf相似物品计算
    item_cf_sim(user_item_dict, user_item_ts, categories, item_user_dict, top_n=20)
