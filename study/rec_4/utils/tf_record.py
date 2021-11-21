#!/usr/bin/env python
# coding: utf-8
# tfRecordFile文件读写
import os
import tensorflow as tf
import numpy as np
import reader


def bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        # BytesList won't unpack a string from an EagerTensor.
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def write(file_name, keys, types, feature_data):
    """
    数据写入tf_record
    Args:
        file_name 数据写入文件名称
        keys 每个特征名称
        types 每个特征类型（bytes、float、int64）
        feature_data 特征数据（二维数组，逐行写如，每列对应key和type）
    """
    # 数据文件
    tf_dir = os.path.join(reader.tf_record_dir, file_name)
    with tf.io.TFRecordWriter(tf_dir) as writer:
        # 需要逐行写入
        for line in feature_data:
            # 建立tf.train.Feature字典
            feature = {}
            for j in range(len(keys)):
                feature[keys[j]] = eval('%s_feature' % types[j])(line[j])
            # 通过字典建立 Example
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            # 将Example序列化
            serialized = example.SerializeToString()
            # 写入TFRecord文件
            writer.write(serialized)
        writer.close()


def read(file_name, keys, types, batch_size=20):
    """
    数据写入tf_record
    Args:
        file_name 数据文件名称
        keys 每个特征名称
        types 每个特征类型（如：tf.int64）
        batch_size 批大小
    """
    # 定义Feature结构，告诉解码器每个Feature的类型是什么
    feature_desc = {}
    for i in range(len(keys)):
        feature_desc[keys[i]] = tf.io.FixedLenFeature([], types[i])

    # tf.train.Example解码,独立取出每个字段
    def _parse_example(example_string):
        result = tf.io.parse_single_example(example_string, feature_desc)
        return [result[key] for key in keys]

    tf_dir = os.path.join(reader.tf_record_dir, file_name)
    tf_dataset_0 = tf.data.TFRecordDataset(tf_dir)
    tf_dataset = tf_dataset_0.map(_parse_example)
    tf_dataset = tf_dataset.batch(batch_size)
    # 高效批处理
    tf_dataset = tf_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return tf_dataset


def write_recod():
    ratings_data = reader.read_rating()
    keys = ['user_id', 'movie_id', 'rating']
    types = ['int64', 'int64', 'int64']
    write('rating', keys, types, ratings_data)


def read_recod():
    keys = ['user_id', 'movie_id', 'rating']
    types = [tf.int64, tf.int64, tf.int64]
    # 分批读出每个特征
    data_set = read('rating', keys, types)
    size = 0
    for user_id, movie_id, rating in data_set:
        if size == 0:
            print('user_id = ', user_id)
            print('movie_id = ', movie_id)
            print('rating = ', rating)
        size += 1
    print('data_set size = ', size)


if __name__ == '__main__':
    # write_recod()
    read_recod()



