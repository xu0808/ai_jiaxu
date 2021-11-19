import tensorflow as tf
import os

print(tf.__version__)
data_dir = 'D:\\Users\\jiaxu\\data\\deepshare'

########## 一、图片读取
#  1、所有图片
imags_dir = os.path.join(data_dir, 'cats_vs_dogs', 'train')
print('image total number is ', len(os.listdir(imags_dir)))
cat_imgs, dog_imgs = [], []
for img in os.listdir(imags_dir):
    if img.startswith('cat'):
        cat_imgs.append(img)
    if img.startswith('dog'):
        dog_imgs.append(img)

print('cat image total number is ', len(cat_imgs), ' ,image top 5: ', cat_imgs[:5])
print('dog image total number is ', len(dog_imgs), ' ,image top 5: ', dog_imgs[:5])

# 2、分割数据集6:3:1
# 训练、验证、测试集数量
total_num = 1000 # 12500 本地训练只取2000
train_num, vail_num = int(total_num * 0.6), int(total_num * 0.3)
test_num = total_num - train_num - vail_num
print('train_num: {}, vail_num: {}, test_num: {}'.format(train_num, vail_num, test_num))

# 训练、验证、测试图片名称集
train_names = cat_imgs[:train_num] + dog_imgs[:train_num]
vail_names = cat_imgs[train_num:train_num + vail_num] + dog_imgs[train_num:train_num + vail_num]
test_names = cat_imgs[train_num + vail_num:total_num] + dog_imgs[train_num + vail_num:total_num]

# 训练、验证、测试图片文件路径 tensor
train_files = tf.constant([imags_dir + "\\" + file for file in train_names])
vail_files = tf.constant([imags_dir + "\\" + file for file in vail_names])
test_files = tf.constant([imags_dir + "\\" + file for file in test_names])

# 训练、验证、测试图片标注 tensor
# cat 0, dag 1
train_labels = tf.concat([tf.zeros((train_num,), dtype=tf.int32),
                          tf.ones((train_num,), dtype=tf.int32)],
                         axis=-1)
vail_labels = tf.concat([tf.zeros((vail_num,), dtype=tf.int32),
                         tf.ones((vail_num,), dtype=tf.int32)],
                        axis=-1)
test_labels = tf.concat([tf.zeros((test_num,), dtype=tf.int32),
                         tf.ones((test_num,), dtype=tf.int32)],
                        axis=-1)


# 读取图片和标注
def _decode_and_resize(filename, label):
    image_string = tf.io.read_file(filename)  # 读取原始文件
    image_decoded = tf.image.decode_jpeg(image_string)  # 解码JPEG图片
    image_resized = tf.image.resize(image_decoded, [256, 256]) / 255.0
    return image_resized, label


# 显示首个图片
def show():
    img, label = _decode_and_resize(tf.constant(imags_dir + "\\" + 'cat.0.jpg'), tf.constant(0))
    import matplotlib.pyplot as plt

    plt.imshow(img.numpy())
    plt.show()


# 3、特征读取
# 读取训练集
batch_size = 32
train_dataset = tf.data.Dataset.from_tensor_slices((train_files, train_labels))
train_dataset = train_dataset.map(map_func=_decode_and_resize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
for img, label in train_dataset.take(1):
    print(img, label)

# 取出前buffer_size个数据放入buffer，并从其中随机采样，采样后的数据用后续数据替换
train_dataset = train_dataset.shuffle(buffer_size=200)
train_dataset = train_dataset.repeat(count=3)
train_dataset = train_dataset.batch(batch_size)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

# 读取测试集
test_dataset = tf.data.Dataset.from_tensor_slices((test_files, test_labels))
test_dataset = test_dataset.map(_decode_and_resize)
test_dataset = test_dataset.batch(batch_size)


########## 二、定义模型
# 1、模型类
class CNNModel(tf.keras.models.Model):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu')
        self.maxpool1 = tf.keras.layers.MaxPooling2D()
        self.conv2 = tf.keras.layers.Conv2D(32, 5, activation='relu')
        self.maxpool2 = tf.keras.layers.MaxPooling2D()
        self.flatten = tf.keras.layers.Flatten()
        self.d1 = tf.keras.layers.Dense(64, activation='relu')
        self.d2 = tf.keras.layers.Dense(2, activation='softmax')  # sigmoid 和softmax

    def call(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        return x


# 2、优化算法
learning_rate = 0.001
model = CNNModel()
# label 没有one-hot
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


@tf.function
def train_step(images, labels):
    # 自定义梯度下降
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)


def test_step(images, labels):
    predictions = model(images)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)


# 3、训练
EPOCHS = 2
for epoch in range(EPOCHS):
    # 在下一个epoch开始时，重置评估指标
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    for images, labels in train_dataset:
        train_step(images, labels)

    for test_images, test_labels in test_dataset:
        test_step(test_images, test_labels)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch + 1,
                          train_loss.result(),
                          train_accuracy.result() * 100,
                          test_loss.result(),
                          test_accuracy.result() * 100
                          ))
