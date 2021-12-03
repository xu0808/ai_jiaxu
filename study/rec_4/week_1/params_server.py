#!/usr/bin/env python
# coding: utf-8
# 参数服务器

import numpy as np


# 单例模式
class PS:
    def __init__(self, vector_dim):
        np.random.seed(2020)
        self.params = dict()
        self.dim = vector_dim
        print("params_server inited...")

    def pull(self, keys):
        values = []
        # 这里传进来的数据是[batch, feature_len]
        for key in keys:
            value = self.params.get(key, None)
            if value is None:
                # 用到的时候才随机产生
                value = np.random.rand(self.dim)
                self.params[key] = value
            values.append(value)
        return np.asarray(values, dtype='float32')

    def push(self, keys, values):
        for i in range(len(keys)):
            self.params[keys[i]] = values[i]
