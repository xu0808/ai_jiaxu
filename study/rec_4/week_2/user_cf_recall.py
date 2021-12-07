#!/usr/bin/env python
# coding: utf-8
# UserCF的主要思想是：给用户推荐与其相似的用户喜欢的的物品。模式：u2u2i
"""
   用户与用户之间的相似性矩阵计算
   用户活跃度，两个用户活跃度的均值越大权重越大
"""
import math
from collections import defaultdict
import reader_2


def user_cf_sim(user_item_dict, user_item_ts, item_user_dict, top_n=20):
    """
     parma user_item_dict 用户点击文章列表字典
     parma user_item_ts   用户点击文章时间字典
     parma categories     文章分类列表
     parma item_user_dict  文章读者用户列表字典
     parma top_n 相似top数
     return  i2i_sim 物品相似度列表
    """
    # 一、逐个物品计算用户相似度
    # 计算用户相似度
    u2u_sim_0 = defaultdict(dict)
    for item, users in item_user_dict.items():
        for i in users:
            num_i = len(user_item_dict[i])
            num_user = len(users)
            for j in users:
                if i == j:
                    continue
                u2u_sim_0[i].setdefault(j, 0)
                # 1、用户平均活跃度作为活跃度的权重，这里的式子也可以改善
                num_j = len(user_item_dict[j])
                act_weight = 0.1 * 0.5 * (num_i + num_j)
                # 2、考虑多种因素的权重计算最终的用户之间的相似度
                u2u_sim_0[i][j] += round(act_weight / math.log(num_user + 1), 4)

    # 二、合并用户相似度
    u2u_sim = {}
    for i, sims in u2u_sim_0.items():
        i_count = len(user_item_dict[i])
        tmp_sim = {}
        for j, w in sims.items():
            tmp_sim[j] = round(w/math.sqrt(i_count * len(user_item_dict[j])), 4)
        # 取相似商品评分topN
        u2u_sim[i] = sorted(tmp_sim.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:top_n]
        print(i, u2u_sim[i])


if __name__ == '__main__':
    # 行为日志读取
    user_item_dict, item_user_dict, user_item_ts, _ = reader_2.read_log()
    # item_cf相似物品计算
    user_cf_sim(user_item_dict, user_item_ts, item_user_dict, top_n=20)
