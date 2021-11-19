import os
import pandas as pd


data_dir = 'E:\\study\\deepshare\\rec_4\\data'
tf_record_dir = 'E:\\study\\deepshare\\rec_4\\tf_record'


# 数据文件
ratings_file = os.path.join(data_dir, 'ratings.dat')
rating_df = pd.read_csv(ratings_file, encoding="utf-8", header=None)
print(rating_df)
