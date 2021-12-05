#!/usr/bin/env python
# coding: utf-8
# 参数服务器

# pip install cacheout
# 缓存库
import cacheout
import numpy as np
import time


# 单例模式
class CacheServer:
    def __init__(self, vector_dim):
        np.random.seed(2020)
        self.cache = cacheout.Cache(maxsize=100000, ttl=0, timer=time.time, default=None)
        self.dim = vector_dim
        print("params_server inited...")

    def pull(self, keys):
        values = []
        cache_dict = self.cache.get_many(keys)
        # 判断是否包含不存在的key
        new_cache_dict = {}
        for key in keys:
            if key not in cache_dict:
                print('value is None -> ', key)
                # 用到的时候才随机产生
                value = np.random.rand(self.dim)
                cache_dict[key] = value
                new_cache_dict[key] = value

            values.append(cache_dict[key])
        if len(new_cache_dict) > 0:
            self.cache.set_many(new_cache_dict)

        return np.asarray(values, dtype='float32')

    def push(self, keys, values):
        cache_dict = {k: v for (k, v) in zip(keys, values)}
        self.cache.set_many(cache_dict)


if __name__ == "__main__":
    # 默认不过期
    ps = CacheServer(vector_dim=2)
    ps.cache.set('-6990346318830052955', {'data': {}}, ttl=1)
    assert ps.cache.get('-6990346318830052955') == {'data': {}}
    time.sleep(1)
    assert ps.cache.get('-6990346318830052955') is None

    # 设置
    ps.cache.set_many({'a': 1, 'b': 2, 'c': 3})
    print(ps.cache.get_many(['a', 'a', 'b', 'd']))