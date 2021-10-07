# 机器翻译-tfrecord存储

# 导入包
import os
import pandas as pd
import tensorflow as tf

# 数据文件
data_dir = 'D:\\Users\\jiaxu\\data\\deepshare\\transformer'
news_file = os.path.join(data_dir, 'news-commentary-v14.en-zh.tsv')

# win10下需要写入同一目录？？？！！！
# 将数据集存储为TFRecord文件
tfrecord_files = [os.path.join(data_dir, '{}.tfrecord'.format(name)) for name in ['train', 'valid']]

df = pd.read_csv(news_file, error_bad_lines=False, sep='\t', header=None)
print(df.head())
# datas = [df.iloc[:280000], df.iloc[280000]]
# 使用50000个样子验证代码
datas = [df.iloc[:50000], df.iloc[50000:60000]]


def featureData(encode):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str.encode(encode)]))


for i in [0, 1]:
    with tf.io.TFRecordWriter(tfrecord_files[i]) as writer:
        for en, zh in datas[i].values:
            try:
                # 建立 tf.train.Feature 字典
                feature = {'en': featureData(en), 'zh': featureData(zh)}
                # 通过字典建立 Example
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                # 将Example序列化并写入 TFRecord 文件
                writer.write(example.SerializeToString())
            except:
                pass
