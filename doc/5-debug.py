import sys
import os
import pandas as pd
import numpy as np

data_dir = 'D:\\study\\rec_4\\data'
tf_record_dir = 'D:\\study\\rec_4\\tf_record'

# 打印Python版本
print("Python version: ", sys.version)
print("Python version info: ", sys.version_info)


def read_rating():
    # 数据文件
    ratings_file = os.path.join(data_dir, 'ratings.dat')
    # 不能直接使用'::'分割
    rating_df = pd.read_csv(ratings_file, encoding="utf-8", header=None, sep=':')
    print('rating_df:', rating_df)
    # 取出数组
    rating_0 = rating_df.values.astype(np.int32)
    print('rating_0.shape:', rating_0.shape)
    # 取出user， movie， rating
    rating = rating_0[:, [0, 2, 4]]
    print('rating.shape:', rating.shape)
    print('rating:', rating)
    # 只取出用户小于2000的记录
    rating_1 = rating[rating[:, 0] < 2000]
    print('rating_1.shape:', rating_1.shape)
    print('rating_1:', rating_1)


if __name__ == '__main__':
    read_rating()