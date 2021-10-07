#!/usr/bin/python
# -*- coding: utf-8 -*-
# 随机游走模型
# 1、Alias采样算法
# https://www.cnblogs.com/Lee-yl/p/12749070.html
"""
1、制作表：（create_alias_table函数，O(N)）
（1）概率分布 area_ratio 的每个概率乘上N
（2）small,large分别存放小于1和大于1的事件下标。
（3）每次从small,large中各取一个，将大的补充到小的之中，小的出队列，
    再看大的减去补给之后，如果大于1，继续放入large中，如果等于1，则也出去，
    如果小于1则放入small中。
（4）获取accept、alias
    accept存放第i列对应的事件i矩形的面积百分比;
    alias存放第i列不是事件i的另外一个事件的标号;
    上例中accept=[2/3,1,1/3,1/3],alias=[2,0,1,1],
    这里alias[1]的0是默认值,也可默认置为-1避免和事件0冲突；

2、采样：（alias_sample函数,O(1)）
  随机采样1~N 之间的整数i，决定落在哪一列，
  随机采样0~1之间的一个概率值，
  如果小于accept[i]，则采样i，
  如果大于accept[i]，则采样alias[i]。
"""

import numpy as np


def normalized_alias_table(probs):
    """ 正则化后的别名表 """
    prob_sum = sum(probs)
    normalized_probs = [float(prob) / prob_sum for prob in probs]
    return create_alias_table(normalized_probs)


def create_alias_table(area_ratio):
    """
    创建别名
    area_ratio[i]代表事件i出现的概率
    :param area_ratio: sum(area_ratio)=1
    :return: accept,alias
    """
    n = len(area_ratio)
    # accept存放事件i百分比
    # alias存放非事件i的标号
    accept, alias = [0] * n, [0] * n
    # 分别存放小于1和大于1的事件下标
    smaller, larger = [], []

    # ------（1）概率 * n -----
    area_ratio_n = np.array(area_ratio) * n

    # ------（2）获取small 、large -----
    for i, prob in enumerate(area_ratio_n):
        accept[i] = prob
        # 刚好为1的直接跳过
        if prob == 1.0:
            continue
        if prob < 1.0:
            smaller.append(i)
        else:
            larger.append(i)

    # ------（3）修改柱状图 ----- （4）获取accept和alias -----
    while smaller and larger:
        # 分别取出最后一个下标
        small_idx, large_idx = smaller.pop(), larger.pop()
        alias[small_idx] = large_idx
        accept[large_idx] = accept[large_idx] - (1 - accept[small_idx])
        if accept[large_idx] < 1.0:
            smaller.append(large_idx)
        else:
            larger.append(large_idx)

    return accept, alias


def alias_sample(accept, alias):
    """
    基于别名表的随机采样
    :param accept:
    :param alias:
    :return: sample index
    """
    n = len(accept)
    i = int(np.random.random() * n)
    r = np.random.random()
    if r < accept[i]:
        return i
    else:
        return alias[i]


if __name__ == '__main__':
    probs = [0.2, 0.3, 0.1, 0.2, 0.2]
    # Construct the table
    accept_res, alias_res = create_alias_table(probs)
    print(accept_res)
    print(alias_res)
    print(alias_sample(accept_res, alias_res))
