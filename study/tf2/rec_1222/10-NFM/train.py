#!/usr/bin/env python
# coding: utf-8


from utils import create_criteo_dataset
from model import NFM

import os
import tensorflow as tf
from tensorflow.keras import losses, optimizers
from sklearn.metrics import accuracy_score

data_dir = 'D:\\study\\ide\\ai_jiaxu\\study\\tf2\\rec_1222\\Data'
file_path = os.path.join(data_dir, 'train.txt')
model_dir = 'D:\\study\\data\\tf2_rec_1222'

if __name__ == '__main__':
    test_size = 0.4
    hidden_units = [256, 128, 64]
    output_dim = 1
    dropout = 0.3

    feature_columns, (X_train, y_train), (X_test, y_test) = \
        create_criteo_dataset(file_path, test_size=test_size)

    model = NFM(feature_columns, hidden_units, output_dim, 'relu', dropout)
    optimizer = optimizers.SGD(0.01)

    # dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    # dataset = dataset.batch(32).prefetch(tf.data.experimental.AUTOTUNE)
    #
    # model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    # model.fit(dataset, epochs=200)
    # pre = model.predict(X_test)

    summary_writer = tf.summary.create_file_writer(model_dir)
    for i in range(200):
        with tf.GradientTape() as tape:
            y_pre = model(X_train)
            loss = tf.reduce_mean(losses.binary_crossentropy(y_train, y_pre))
            print((loss.numpy()))
        grad = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grad, model.variables))

        pre = model(X_test)
        pre = [1 if x > 0.5 else 0 for x in pre]
        auc = accuracy_score(y_test, pre)
        with summary_writer.as_default():
            tf.summary.scalar('loss', loss, step=i)
            tf.summary.scalar('auc', auc, step=i)
    pre = model(X_test)

    pre = [1 if x > 0.5 else 0 for x in pre]
    print("Accuracy: ", accuracy_score(y_test, pre))
