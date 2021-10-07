import numpy as np
import tensorflow as tf
import time
from model import EGES
from utils import *

# 数据文件配置
data_dir = '../data'
# 特征数据
feature_file = '%s_1/feature.csv' % data_dir
# 采样组合
sample_pairs_file = '%s_1/sample_pairs_0.csv' % data_dir
# 负采样数
n_sampled = 100
# 输出向量维度
emb_dim = 128
batch_size = 100
epochs = 1


def mini_batch(all_pair, side_info, batch_size):
    # 随机生成批起始点
    start_idx = np.random.randint(0, len(all_pair) - batch_size)
    # 批索引
    batch_idx = np.array(range(start_idx, start_idx + batch_size))
    # 打乱顺序
    batch_idx = np.random.permutation(batch_idx)
    batch_pair = all_pair[batch_idx]
    feature = side_info[batch_pair[:, 0]]
    label = batch_pair[:, 1].reshape(batch_size, -1)
    print('feature + label = ')
    print(feature)
    print(label)
    return feature, label


if __name__ == '__main__':
    # read train_data
    print('read features start!')
    start_time = time.time()
    # 特征
    side_info = np.loadtxt(feature_file, dtype=np.int32, delimiter='\t')
    # 组合
    all_pair = np.loadtxt(sample_pairs_file, dtype=np.int64, delimiter=' ')
    # 特征数
    num_feature = side_info.shape[1]
    # 每个特征分类格式
    feature_lens = [len(set(side_info[:, i])) for i in range(num_feature)]
    print('read features start! %.2f s' % (time.time() - start_time))

    model = EGES(len(side_info), num_feature, feature_lens, n_sample=n_sampled, emb_dim=emb_dim)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, decay=5e-4)

    with tf.GradientTape() as tape:
        feature, label = mini_batch(all_pair, side_info, batch_size)
        _ = model(feature, label)
        loss = model.losses[0]
        # 梯度下降
        grads = tape.gradient(loss, model.trainable_variables)
        # 优化器
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    print('{:03d} Loss: {:.4f}'.format(loss))





