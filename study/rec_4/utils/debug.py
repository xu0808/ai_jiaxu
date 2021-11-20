import os
import pandas as pd
import numpy as np

data_dir = 'D:\\study\\rec_4\\data'
tf_record_dir = 'D:\\study\\rec_4\\tf_record'


# 数据文件
ratings_file = os.path.join(data_dir, 'ratings.dat')
# 不能直接使用'::'分割
rating_df = pd.read_csv(ratings_file, encoding="utf-8", header=None, sep=':')
print('rating_df:', rating_df)
rating_0 = rating_df.values.astype(np.int32)
print('rating_0.shape:', rating_0.shape)
rating = rating_0[:, [0, 2, 4]]
print('rating.shape:', rating.shape)
print('rating:', rating)
