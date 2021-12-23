#!/usr/bin/env python
# coding: utf-8


from model import AFM
from utils import create_criteo_dataset

import os
import tensorflow as tf
from tensorflow.keras import optimizers, losses, metrics
from sklearn.metrics import accuracy_score

data_dir = 'D:\\study\\ide\\ai_jiaxu\\study\\tf2\\rec_1222\\Data'
file_path = os.path.join(data_dir, 'train.txt')
model_dir = 'D:\\study\\data\\tf2_rec_1222'

if __name__ == '__main__':
    test_size = 0.2
    feature_columns, (X_train, y_train), (X_test, y_test) = \
        create_criteo_dataset(file_path, test_size=test_size)

    model = AFM(feature_columns, 'att')
    optimizer = optimizers.SGD(0.01)

    # dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    # dataset = dataset.batch(32).prefetch(tf.data.experimental.AUTOTUNE)
    #
    # model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    # model.fit(dataset, epochs=100)
    # pre = model.predict(X_test)

    summary = tf.summary.create_file_writer(model_dir)
    for i in range(100):
        with tf.GradientTape() as tape:
            pre = model(X_train)
            loss = tf.reduce_mean(losses.binary_crossentropy(y_train, pre))
            print(loss.numpy())
        # with summary.as_default():
        # tf.summary.scalar('loss', loss, i)
        grad = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grad, model.variables))
    pre = model(X_test)

    pre = [1 if x > 0.5 else 0 for x in pre]
    print("Accuracy: ", accuracy_score(y_test, pre))
