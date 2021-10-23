# -*- keras方式的样本加权和分类加权

import tensorflow as tf

print(tf.__version__)


# 2.1 构建模型
def get_uncompiled_model():
    inputs = tf.keras.Input(shape=(32,), name='digits')
    x = tf.keras.layers.Dense(64, activation='relu', name='dense_1')(inputs)
    x = tf.keras.layers.Dense(64, activation='relu', name='dense_2')(x)
    outputs = tf.keras.layers.Dense(10, name='predictions')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def get_compiled_model():
    model = get_uncompiled_model()
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-3),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['sparse_categorical_accuracy'])
    return model


# 2.2 模型训练
import numpy as np

x_train = np.random.random((1000, 32))
y_train = np.random.randint(10, size=(1000,))
# 分类加权
class_weight = {0: 1., 1: 1., 2: 1., 3: 1., 4: 1.,
                5: 2.,
                6: 1., 7: 1., 8: 1., 9: 1.}

print('Fit with class weight')
model = get_compiled_model()
model.fit(x_train, y_train,
          class_weight=class_weight,
          batch_size=64,
          epochs=4)

# 样本加权
sample_weight = np.ones(shape=(len(y_train),))
sample_weight[y_train == 5] = 2.
print('\nFit with sample weight')
print(y_train)

model = get_compiled_model()
model.fit(x_train, y_train,
          sample_weight=sample_weight,
          batch_size=64,
          epochs=4)
