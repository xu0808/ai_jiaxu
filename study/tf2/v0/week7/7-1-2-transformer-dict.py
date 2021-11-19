# 机器翻译-生成字典

# 导入包
import os
import tensorflow as tf
import tensorflow_datasets as tfds

# 数据文件
data_dir = 'D:\\Users\\jiaxu\\data\\deepshare\\transformer'
train_tfrecord = os.path.join(data_dir, 'train.tfrecord')
valid_tfrecord = os.path.join(data_dir, 'valid.tfrecord')

# 字典文件
en_vocab_file = os.path.join(data_dir, "en_vocab")
zh_vocab_file = os.path.join(data_dir, "zh_vocab")

# 1、数据读取
# 定义Feature结构，告诉解码器每个Feature的类型是什么
feature_description = {
    'en': tf.io.FixedLenFeature([], tf.string),
    'zh': tf.io.FixedLenFeature([], tf.string),
}


# 将 TFRecord 文件中的每一个序列化的 tf.train.Example 解码
def _parse_example(example_string):
    feature_dict = tf.io.parse_single_example(example_string, feature_description)
    return feature_dict['en'], feature_dict['zh']


train_examples = tf.data.TFRecordDataset(train_tfrecord).map(_parse_example)
vocab_files = [en_vocab_file, zh_vocab_file]
print("建立字典")  # 最好单独建立字典否则内存不够！！！
for i in [0, 1]:
    # 有需要可以调整字典大小
    corpus = [en_zh[i].numpy() for en_zh in train_examples]
    subword_encoder = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(corpus, target_vocab_size=10000)
    # 将字典档案存下以方便下次 warmstart
    subword_encoder.save_to_file(vocab_files[i])
    print(f"字典大小：{subword_encoder.vocab_size}")
    print(f"前 10 个 subwords：{subword_encoder.subwords[:10]}")
