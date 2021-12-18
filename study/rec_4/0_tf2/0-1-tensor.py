# -*- tensor 的使用
import tensorflow as tf

print(tf.__version__)


def tensor():
    # 一、模型参数
    # Rank 0
    mammal = tf.Variable("Elephant", tf.string)
    tf.print(tf.rank(mammal))
    tf.print(tf.shape(mammal))
    # print(tf.rank(mammal))

    # Rank 1
    mystr = tf.Variable(["Hello"], tf.string)
    tf.print(tf.rank(mystr))
    tf.print(tf.shape(mystr))

    # Rank 2
    mymat = tf.Variable([[7], [11]], tf.int16)
    tf.print(tf.rank(mymat))
    tf.print(tf.shape(mymat))

    # 二、特征参数
    # 创建张量
    a = tf.constant([1, 2, 3], dtype=tf.int16)
    tf.print(tf.rank(a))
    tf.print(tf.shape(a))
    b = tf.zeros((2, 2), dtype=tf.int16)
    tf.print(tf.rank(b))
    tf.print(tf.shape(b))
    # reshape
    rank_three_tensor = tf.ones([3, 4, 5])
    tf.print(tf.rank(rank_three_tensor))
    tf.print(tf.shape(rank_three_tensor))
    matrix = tf.reshape(rank_three_tensor, [6, 10])
    tf.print(tf.rank(matrix))
    tf.print(tf.shape(matrix))


def operate():
    # 一、数据处理
    # tf.strings
    # 字符切割
    a = tf.strings.bytes_split('hello')
    print("tf.strings.bytes_split('hello')", a)
    # 单词切割
    b = tf.strings.split('hello world')
    print("tf.strings.split('hello world')", b)
    # string hash
    c = tf.strings.to_hash_bucket(['hello', 'world'], num_buckets=10)
    print("tf.strings.to_hash_bucket(['hello', 'world']", c)

    # tf.debugging
    # tf自带debug函数
    a = tf.random.uniform((10, 10))
    tf.debugging.assert_equal(x=a.shape, y=(10, 10))
    # # 错误示范
    # tf.debugging.assert_equal(x=a.shape, y=(20, 10))

    # tf.random
    a = tf.random.uniform(shape=(10, 5), minval=0, maxval=10)
    print(a)

    # 二、数学运算
    # tf.math
    a = tf.constant([[1, 2], [3, 4]])
    b = tf.constant([[5, 6], [7, 8]])
    tf.print(tf.math.add(a, b))
    tf.print(tf.math.subtract(a, b))
    tf.print(tf.math.multiply(a, b))
    tf.print(tf.math.divide(a, b))

    # tf.dtypes
    x = tf.constant([1.8, 2.2], dtype=tf.float32)
    x1 = tf.dtypes.cast(x, tf.int32)
    tf.print(x1)


if __name__ == '__main__':
    # tensor()
    operate()
