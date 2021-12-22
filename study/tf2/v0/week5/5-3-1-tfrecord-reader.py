# 写入tfrecordfile(每个文件只取500个)
# 特别注意：
# 当前下载数据集本身已经图片格式了，课程中原始数据为涂鸦点集故需要画图生成图片
# 模型要求图片不小于32，resize（28 -> 128）
import os
import tensorflow as tf

data_dir = 'D:\\Users\\jiaxu\\data\\deepshare'
quick_draw_dir = os.path.join(data_dir, 'quick_draw')
img_size = 128

# 将数据集存储为TFRecord文件
train_tfrecord = os.path.join(quick_draw_dir, 'train.tfrecords')
test_tfrecord = os.path.join(quick_draw_dir, 'test.tfrecords')

# 定义Feature结构，告诉解码器每个Feature的类型是什么
feature_description = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.int64),
}


# 将 TFRecord 文件中的每一个序列化的 tf.train.Example 解码
def _parse_example(example_string):
    feature_dict = tf.io.parse_single_example(example_string, feature_description)
    image = tf.io.decode_raw(feature_dict['image'], tf.uint8)  # 解码JPEG图片
    image = tf.reshape(image, [img_size, img_size, 1])
    image = tf.dtypes.cast(image, tf.float32)
    image = image / 255.0
    label = tf.one_hot(feature_dict['label'], depth=340)
    return image, label


batch_size = 32
train_dataset = tf.data.TFRecordDataset(train_tfrecord)
train_dataset = train_dataset.map(_parse_example)
train_dataset = train_dataset.shuffle(buffer_size=200)
train_dataset = train_dataset.batch(batch_size)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)


class MobileNetModel(tf.keras.models.Model):
    def __init__(self, size, n_labels, **kwargs):
        super(MobileNetModel, self).__init__(**kwargs)
        self.base_model = tf.keras.applications.MobileNet(input_shape=(size, size, 1), include_top=False, weights=None,
                                                          classes=n_labels)
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(1024, activation='relu')
        self.outputs = tf.keras.layers.Dense(n_labels, activation='softmax')

    def call(self, inputs):
        x = self.base_model(inputs)
        x = self.flatten(x)
        x = self.dense(x)
        output_ = self.outputs(x)
        return output_


model = MobileNetModel(size=img_size, n_labels=340)
loss_object = tf.keras.losses.CategoricalCrossentropy()
learning_rate = 0.001
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
train_top3_accuracy = tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='train_top_3_categorical_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')
test_top3_accuracy = tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='test_top_3_categorical_accuracy')


# @tf.function
def train_one_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)
    train_top3_accuracy(labels, predictions)


def val_one_step(images, labels):
    predictions = model(images)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)
    test_top3_accuracy(labels, predictions)


EPOCHS = 3
for epoch in range(EPOCHS):
    # 在下一个epoch开始时，重置评估指标
    train_loss.reset_states()
    train_accuracy.reset_states()
    train_top3_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()
    test_top3_accuracy.reset_states()

    for step, (images, labels) in enumerate(train_dataset):
        train_one_step(images, labels)

        if step % 200 == 0:
            print("step:{0}; Samples:{1}; Train Loss:{2}; Train Accuracy:{3},Train Top3 Accuracy:{4}".format(step, (
                    step + 1) * 1024,
                                                                                                             train_loss.result(),
                                                                                                             train_accuracy.result() * 100,
                                                                                                             train_top3_accuracy.result() * 100))

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch + 1,
                          train_loss.result(),
                          train_accuracy.result() * 100,
                          train_top3_accuracy() * 100,
                          test_loss.result(),
                          test_accuracy.result() * 100,
                          test_top3_accuracy() * 100
                          ))
