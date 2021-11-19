# tfRecordFile文件写入
import tensorflow as tf

print(tf.__version__)
data_dir = 'D:\\study\\deepshare\\rec_4\\data'
tf_record_dir = 'D:\\study\\deepshare\\rec_4\\tf_record'


def bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        # BytesList won't unpack a string from an EagerTensor.
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

