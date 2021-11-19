# tfRecordFile文件写入
import tensorflow as tf
import os

print(tf.__version__)
data_dir = 'D:\\study\\deepshare\\rec_4\\data'

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

# 训练、验证、测试图片文件路径集合
train_files = [imags_dir + "\\" + file for file in train_names]
vail_files = [imags_dir + "\\" + file for file in vail_names]
test_files = [imags_dir + "\\" + file for file in test_names]

# 训练、验证、测试图片标注 tensor
# cat 0, dag 1
train_labels = [0] * len(train_names) + [1] * len(train_names)
vail_labels = [0] * len(vail_names) + [1] * len(vail_names)
test_labels = [0] * len(test_names) + [1] * len(test_names)

# 将数据集存储为TFRecord文件
tf_record_dir = os.path.join(data_dir, 'cats_vs_dogs', 'tf-record')
train_tfrecord = os.path.join(tf_record_dir, 'train.tfrecords')
test_tfrecord = os.path.join(tf_record_dir, 'test.tfrecords')

with tf.io.TFRecordWriter(train_tfrecord) as writer:
    for filename, label in zip(train_files, train_labels):
        image = open(filename, 'rb').read()  # 读取数据集图片到内存，image 为一个 Byte 类型的字符串

        feature = {  # 建立 tf.train.Feature 字典
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),  # 图片是一个 Bytes 对象
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))  # 标签是一个 Int 对象
        }

        example = tf.train.Example(features=tf.train.Features(feature=feature))  # 通过字典建立 Example
        writer.write(example.SerializeToString())  # 将Example序列化并写入 TFRecord 文件


with tf.io.TFRecordWriter(test_tfrecord) as writer:
    for filename, label in zip(test_files, test_labels):
        image = open(filename, 'rb').read()     # 读取数据集图片到内存，image 为一个 Byte 类型的字符串
        feature = {                             # 建立 tf.train.Feature 字典
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),  # 图片是一个 Bytes 对象
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))   # 标签是一个 Int 对象
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature)) # 通过字典建立 Example
        serialized = example.SerializeToString() #将Example序列化
        writer.write(serialized)   # 写入 TFRecord 文件
