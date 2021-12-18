# -*- 创建模型的三种方式
# 1、序列方式
# 2、函数方式
# 3、子类方式
# 完美：
# 样本普通np数组即可
# model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
# model.fit(data, labels, epochs=10, batch_size=32)

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

tf.print(tf.__version__)
tf.print(tf.test.is_gpu_available())

optimizer = tf.keras.optimizers.RMSprop(0.001)
loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
metrics = ['accuracy']


def seq():
    """
    # 1、序列方式
    序列层组成模型
    注意：样本可以是普通np数组
    """
    # 第一种Sequential
    model = tf.keras.Sequential()
    model.add(layers.Dense(64, activation='relu'))  # 第一层
    model.add(layers.Dense(64, activation='relu'))  # 第二层
    model.add(layers.Dense(10))  # 第三层
    # 。。。。。。。
    # 第二种Sequential
    model = tf.keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(32,)),  # 第一层
        layers.Dense(64, activation='relu'),  # 第二层
        layers.Dense(10)  # 第三层
        # 。。。。。
    ])

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    data = np.random.random((1000, 32))
    labels = np.random.random((1000, 10))
    model.fit(data, labels, epochs=10, batch_size=32)


def func():
    """
    # 2、函数方式
    Model定义时指定输入、输出
    """
    # 1、单一输入、输出
    inputs = tf.keras.Input(shape=(32,))
    x = layers.Dense(64, activation='relu')(inputs)  # 第一层
    x = layers.Dense(64, activation='relu')(x)  # 第二层
    predictions = layers.Dense(10)(x)  # 第三层
    model = tf.keras.Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    data = np.random.random((1000, 32))
    labels = np.random.random((1000, 10))
    model.fit(data, labels, batch_size=32, epochs=5)

    # 2、多输入、输出
    inputs1 = tf.keras.Input(shape=(32,))  # 输入1
    inputs2 = tf.keras.Input(shape=(32,))  # 输入2
    x1 = layers.Dense(64, activation='relu')(inputs1)  # 第一层
    x2 = layers.Dense(64, activation='relu')(inputs2)  # 第一层
    x = tf.concat([x1, x2], axis=-1)
    x = layers.Dense(64, activation='relu')(x)  # 第二层
    predictions = layers.Dense(10)(x)  # 第三层
    model = tf.keras.Model(inputs=[inputs1, inputs2], outputs=predictions)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    data1 = np.random.random((1000, 32))
    data2 = np.random.random((1000, 32))
    labels = np.random.random((1000, 10))
    model.fit((data1, data2), labels, batch_size=32, epochs=5)


class MyModel(tf.keras.Model):

    def __init__(self, num_classes=10):
        super(MyModel, self).__init__(name='my_model')
        self.num_classes = num_classes
        # 定义自己需要的层
        self.dense_1 = layers.Dense(32, activation='relu') #
        self.dense_2 = layers.Dense(num_classes)

    def call(self, inputs):
        # 定义前向传播
        # 使用在 (in `__init__`)定义的层
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        return x


def sub():
    model = MyModel(num_classes=10)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    data = np.random.random((1000, 32))
    labels = np.random.random((1000, 10))
    # Trains for 5 epochs.
    model.fit(data, labels, batch_size=32, epochs=5)


if __name__ == '__main__':
    # 1、Sequential
    # seq()
    # 2、Functional
    # func()
    # 3、SubClass
    sub()
