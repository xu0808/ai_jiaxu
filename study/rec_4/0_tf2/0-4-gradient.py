# -*- 自动求导机制
import tensorflow as tf
import numpy as np


class MyModel(tf.keras.Model):
    def __init__(self, num_classes=10):
        super(MyModel, self).__init__(name='my_model')
        self.num_classes = num_classes
        # 定义自己需要的层
        self.dense_1 = tf.keras.layers.Dense(32, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(num_classes)

    def call(self, inputs):
        # 定义前向传播
        x = self.dense_1(inputs)
        return self.dense_2(x)


x_train = np.random.random((1000, 32))
y_train = np.random.random((1000, 10))
x_val = np.random.random((200, 32))
y_val = np.random.random((200, 10))
x_test = np.random.random((200, 32))
y_test = np.random.random((200, 10))

# 优化器
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
# 损失函数
loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

# 准备metrics函数
train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
val_acc_metric = tf.keras.metrics.CategoricalAccuracy()

# 准备训练数据集
batch_size = 64
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

# 准备测试数据集
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(64)

model = MyModel(num_classes=10)
epochs = 3
for epoch in range(epochs):
    print('Start of epoch %d' % (epoch,))

    # 遍历数据集的batch_size
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        # 一个batch
        with tf.GradientTape() as tape:
            logits = model(x_batch_train)
            loss_value = loss_fn(y_batch_train, logits)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # 更新训练集的metrics
        train_acc_metric(y_batch_train, logits)

        # 在每个epoch结束时显示metrics。
    train_acc = train_acc_metric.result()
    print('Training acc over epoch: %s' % (float(train_acc),))
    # 在每个epoch结束时重置训练指标
    train_acc_metric.reset_states()  # !!!!!!!!!!!!!!!

    # 在每个epoch结束时运行一个验证集。
    for x_batch_val, y_batch_val in val_dataset:
        val_logits = model(x_batch_val)
        # 更新验证集merics
        val_acc_metric(y_batch_val, val_logits)
    val_acc = val_acc_metric.result()
    print('Validation acc: %s' % (float(val_acc),))
    val_acc_metric.reset_states()
