#!/usr/bin/env python
# coding: utf-8
# 基于item_cf模型

from collections import defaultdict
import math
import numpy as np


class ItemCF:
    def __init__(self, user_item_dict, item_hot_list, sim_item_topK, topN, i2i_sim=None):
        """
        Item-based collaborative filtering
        :param user_item_dict: A dict. {user1: [(item1, score),...], user2: ...}
        :param item_hot_list: A list. The popular movies list.
        :param sim_item_topK: A scalar. Choose topK items for calculate.
        :param topN: A scalar. The number of recommender list.
        :param i2i_sim: dict. If None, the model should calculate similarity matrix.
        """

    def __get_item_sim(self):
        """
        calculate item similarity weight matrix
        :return: i2i_sim
        """
        i2i_sim = dict()
        item_cnt = defaultdict(int)
        for user, items in self.user_item_dict.items():
            for i, score_i in items:
                item_cnt[i] += 1
                i2i_sim.setdefault(i, {})
                for j, score_j in items:
                    if i == j:
                        continue
                    i2i_sim[i].setdefault(j, 0)
                    i2i_sim[i][j] += 1 / math.log(len(items) + 1)
        for i, related_items in i2i_sim.items():
            for j, wij in related_items.items():
                i2i_sim[i][j] = wij / math.sqrt(item_cnt[i] * item_cnt[j])
        return i2i_sim

    def recommend(self, user_id):
        """
        recommend one user
        :param user_id: user's ID
        :return:
        """
        item_rank = dict()
        user_hist_items = self.user_item_dict[user_id]
        for i, score_i in user_hist_items:
            for j, wij in sorted(self.i2i_sim[i].items(), key=lambda x: x[1], reverse=True)[:self.sim_item_topK]:
                if j in user_hist_items:
                    continue

                item_rank.setdefault(j, 0)
                item_rank[j] += 1 * wij

        if len(item_rank) < self.topN:
            for i, item in enumerate(self.item_hot_list):
                if item in item_rank:
                    continue
                item_rank[item] = - i - 1  # rank score < 0
                if len(item_rank) == self.topN:
                    break
        item_rank = sorted(item_rank.items(), key=lambda x: x[1], reverse=True)[:self.topN]

        return [i for i, score in item_rank]

    def evaluate_model(model, test):
        """
        evaluate model
        :param model: model of CF
        :param test: dict.
        :return: hit rate, ndcg
        """
        hit, ndcg = 0, 0
        for user_id, item_id in test.items():
            item_rank = model.recommend(user_id)
            if item_id in item_rank:
                hit += 1
                ndcg += 1 / np.log2(item_rank.index(item_id) + 2)
        return hit / len(test), ndcg / len(test)
