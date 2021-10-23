# -*- 使用回调函数

import tensorflow as tf
print(tf.__version__)

# 构建模型
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


# 3.1 EarlyStopping(早停)
model = get_compiled_model()
#list
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        # 当‘val_loss’不再下降时候停止训练
        monitor='val_loss',
        # “不再下降”被定义为“减少不超过1e-2”
        min_delta=1e-2,
        # “不再改善”进一步定义为“至少2个epoch”
        patience=2,
        verbose=1)
]


model.fit(x_train, y_train,
          epochs=20,
          batch_size=64,
          callbacks=callbacks,
          validation_split=0.2)


# 3.2 checkpoint模型
model = get_compiled_model()

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        # 模型保存路径
        filepath='mymodel_{epoch}',
        # 下面的两个参数意味着当且仅当`val_loss`分数提高时，我们才会覆盖当前检查点。
        save_best_only=True,
        monitor='val_loss',
        #加入这个仅仅保存模型权重
        save_weights_only=True,
        verbose=1)
]
model.fit(x_train, y_train,
          epochs=3,
          batch_size=64,
          callbacks=callbacks,
          validation_split=0.2)

# 3.3、使用回调实现动态学习率调整
model = get_compiled_model()

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath='mymodel_{epoch}',
        # 模型保存路径
        # 下面的两个参数意味着当且仅当`val_loss`分数提高时，我们才会覆盖当前检查点。
        save_best_only=True,
        monitor='val_loss',
        # 加入这个仅仅保存模型权重
        save_weights_only=True,
        verbose=1),

    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_sparse_categorical_accuracy",
                                         verbose=1,
                                         mode='max',
                                         factor=0.5,
                                         patience=3)
]
model.fit(x_train, y_train,
          epochs=30,
          batch_size=64,
          callbacks=callbacks,
          validation_split=0.2
          )