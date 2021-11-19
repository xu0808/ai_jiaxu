# -*- keras方式的模型训练

import tensorflow as tf

print(tf.__version__)

# 1.1 构建模型
inputs = tf.keras.Input(shape=(32,))  # (batch_size=32,数据维度32)
x = tf.keras.layers.Dense(64, activation='relu')(inputs)  # （64个神经元，）
x = tf.keras.layers.Dense(64, activation='relu')(x)  # （63个神经元）
predictions = tf.keras.layers.Dense(10)(x)  # （输出是10类）

# - inputs(模型输入)
# - output(模型输出)
model = tf.keras.Model(inputs=inputs, outputs=predictions)
# 指定损失函数 (loss) tf.keras.optimizers.RMSprop
# 优化器 (optimizer) tf.keras.losses.SparseCategoricalCrossentropy
# 指标 (metrics) ['accuracy']

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),  # 优化器
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # 损失函数
              metrics=['accuracy'])  # 评估函数

# 构建数据集
import numpy as np

x_train = np.random.random((1000, 32))
y_train = np.random.randint(10, size=(1000,))

x_val = np.random.random((200, 32))
y_val = np.random.randint(10, size=(200,))

x_test = np.random.random((200, 32))
y_test = np.random.randint(10, size=(200,))

# 指定验证集
model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_val, y_val))
# 分割训练集中产生验证集
# model.fit(x_train, y_train, batch_size=64, validation_split=0.2, epochs=1)

# 1.3 模型验证
# Evaluate the model on the test gat using `evaluate`
print('\n# Evaluate on test gat')
results = model.evaluate(x_test, y_test, batch_size=128)
print('test loss, test acc:', results)

# Generate predictions (probabilities -- the output of the last layer)
# on new gat using `predict`
print('\n# Generate predictions for 3 samples')
predictions = model.predict(x_test[:3])
print('predictions shape:', predictions.shape)
