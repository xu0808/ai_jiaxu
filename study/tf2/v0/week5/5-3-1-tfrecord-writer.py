# 写入tfrecordfile(每个文件只取500个)
# 特别注意：
# 当前下载数据集本身已经图片格式了，课程中原始数据为涂鸦点集故需要画图生成图片
import os
import numpy as np
import tensorflow as tf
import cv2

data_dir = 'D:\\Users\\jiaxu\\data\\deepshare'
quick_draw_dir = os.path.join(data_dir, 'quick_draw')
img_num = 10000
img_size = 28
img_size_1 = 128


# 将数据集存储为TFRecord文件
train_tfrecord = os.path.join(quick_draw_dir, 'train.tfrecords')
test_tfrecord = os.path.join(quick_draw_dir, 'test.tfrecords')

with tf.io.TFRecordWriter(train_tfrecord) as writer:
    for nzp in os.listdir(quick_draw_dir):
        if nzp.startswith('train_batch'):
            data = np.load(os.path.join(quick_draw_dir, nzp))
            imgs = data['gat'][:img_num]
            imgs = np.reshape(imgs, [-1, img_size, img_size])
            labels = data['labels'][:img_num].tolist()

            for img, label in zip(imgs, labels):
                img = cv2.resize(img, (img_size_1, img_size_1)).tostring()
                # 建立 tf.train.Feature 字典
                feature = {
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img])),  # 图片是一个 Bytes 对象
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))  # 标签是一个 Int 对象
                }
                example = tf.train.Example(features=tf.train.Features(feature=feature))  # 通过字典建立 Example
                writer.write(example.SerializeToString())  # 将Example序列化并写入 TFRecord 文件


with tf.io.TFRecordWriter(test_tfrecord) as writer:
    for nzp in os.listdir(quick_draw_dir):
        if nzp == 'test.npz':
            data = np.load(os.path.join(quick_draw_dir, nzp))
            imgs = data['gat'][:img_num]
            imgs = np.reshape(imgs, [-1, img_size, img_size])
            labels = data['labels'][:img_num].tolist()

            for img, label in zip(imgs, labels):
                img = cv2.resize(img, (img_size_1, img_size_1)).tostring()
                # 建立 tf.train.Feature 字典
                feature = {
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img])),  # 图片是一个 Bytes 对象
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))  # 标签是一个 Int 对象
                }
                example = tf.train.Example(features=tf.train.Features(feature=feature))  # 通过字典建立 Example
                writer.write(example.SerializeToString())  # 将Example序列化并写入 TFRecord 文件
