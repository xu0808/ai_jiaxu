from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
import numpy as np
import os

# 数据文件
data_dir = 'D:\\Users\\jiaxu\\data\\deepshare'
mnist = np.load(os.path.join(data_dir, "mnist.npz"))
x_train, y_train = mnist['x_train'], mnist['y_train']
print('x_train.shape = {}, y_train.shape = {}'.format(x_train.shape, y_train.shape))
x_test, y_test = mnist['x_test'], mnist['y_test']
print('x_test.shape = {}, y_test.shape = {}'.format(x_test.shape, y_test.shape))
# 图像归一化
x_train, x_test = x_train / 255.0, x_test / 255.0
# 添加通道维度
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]
print('x_train.shape = {}, x_test.shape = {}'.format(x_train.shape, x_test.shape))

ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(32)
ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)


class my_model(Model):
    def __init__(self):
        super(my_model, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)


# 1、定义模型
model = my_model()
# 2、损失函数
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
# 3、优化其
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predict = model(images)
        loss = loss_object(labels, predict)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predict)


@tf.function
def test_step(images, labels):
    predict = model(images)
    t_loss = loss_object(labels, predict)

    test_loss(t_loss)
    test_accuracy(labels, predict)


EPOCHS = 5

for epoch in range(EPOCHS):
    # 在下一个epoch开始时，重置评估指标
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    for images, labels in ds_train:
        train_step(images, labels)

    for test_images, test_labels in ds_test:
        test_step(test_images, test_labels)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch + 1, train_loss.result(), train_accuracy.result() * 100,
                          test_loss.result(), test_accuracy.result() * 100))
