#!/usr/bin/env python
# coding: utf-8

from model import xDeepFM
from utils import create_criteo_dataset

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import losses, optimizers
from sklearn.metrics import accuracy_score

data_dir = 'D:\\study\\ide\\ai_jiaxu\\study\\tf2\\rec_1222\\Data'
file_path = os.path.join(data_dir, 'train.txt')
model_dir = 'D:\\study\\data\\tf2_rec_1222'

if __name__ == '__main__':
    test_size = 0.2
    hidden_units = [256, 128, 64]
    dropout = 0.3
    cin_size = [128, 128]

    feature_columns, (X_train, y_train), (X_test, y_test) = create_criteo_dataset(file_path, test_size=test_size)

    model = xDeepFM(feature_columns, cin_size, hidden_units, dropout=dropout)
    optimizer = optimizers.SGD(0.01)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.batch(32).prefetch(tf.data.experimental.AUTOTUNE)

    summary_writer = tf.summary.create_file_writer(model_dir)
    for epoch in range(30):
        loss_summary = []
        for batch, data_batch in enumerate(train_dataset):
            X_train, y_train = data_batch[0], data_batch[1]
            with tf.GradientTape() as tape:
                y_pre = model(X_train)
                loss = tf.reduce_mean(losses.binary_crossentropy(y_true=y_train, y_pred=y_pre))
                grad = tape.gradient(loss, model.variables)
                optimizer.apply_gradients(grads_and_vars=zip(grad, model.variables))
            if batch % 10 == 0:
                print('epoch: {} batch: {} loss: {}'.format(epoch, batch, loss.numpy()))
            loss_summary.append(loss.numpy())
        with summary_writer.as_default():
            tf.summary.scalar("loss", np.mean(loss_summary), step=epoch)

    pre = model(X_test)
    pre = [1 if x > 0.5 else 0 for x in pre]
    print("Accuracy: ", accuracy_score(y_test, pre))
