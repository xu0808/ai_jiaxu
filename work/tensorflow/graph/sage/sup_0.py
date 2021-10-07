#!/usr/bin/env python
# coding: utf-8
# GraphSAGE-Tensorflow 2.0-有监督分类模型
# https://zhuanlan.zhihu.com/p/149684079
# 图卷积神经网络入门实战-Tensorflow 2.0实现
# https://zhuanlan.zhihu.com/p/148107956
import tensorflow as tf
import numpy as np
from dataset import CoraData
import matplotlib.pyplot as plt


def sampling(src_nodes, sample_num, neighbor_table):
    """根据源节点采样指定数量的邻居节点，注意使用的是有放回的采样；
       某个节点的邻居节点数量少于采样数量时，采样结果出现重复的节点
    param src_nodes {list, ndarray} -- 源节点列表
    param sample_num {int} -- 需要采样的节点数
    param neighbor_table {dict} -- 节点到其邻居节点的映射表
    return np.ndarray -- 采样结果构成的列表
    """
    results = []
    for node_id in src_nodes:
        # 从节点的邻居中进行有放回地进行采样
        res = np.random.choice(neighbor_table[node_id], size=(sample_num,))
        results.append(res)
    return np.asarray(results).flatten()


def multihop_sampling(src_nodes, sample_nums, neighbor_table):
    """根据源节点进行多阶采样
    param src_nodes {list, np.ndarray} -- 源节点id
    param sample_nums {list of int} -- 每阶需要采样的个数
    param neighbor_table {dict} -- 节点到其邻居节点的映射
    return [list of ndarray] -- 每阶采样的总结果
    """
    # 源节点集合
    sampling_result = [src_nodes]  # 首先包含源节点
    # k阶采样
    # 先对源节点进行1阶采样 在与源节点距离为1的节点中采样hopk_num个节点；
    # 再对源节点进行2阶采样，即对源节点的所有1阶邻居进行1阶采样
    for k, hopk_num in enumerate(sample_nums):
        pre_nodes = sampling_result[k]
        # 针对上一阶节点采样hopk_num个
        hopk_result = sampling(pre_nodes, hopk_num, neighbor_table)
        sampling_result.append(hopk_result)

    # 每阶采样的总结果
    return sampling_result


class NeighborAgg(tf.keras.Model):
    def __init__(self, input_dim, output_dim,
                 use_bias=False, aggr_method='mean'):
        """聚合邻居节点
           param input_dim: 输入特征的维度
           param output_dim: 输出特征的维度
           param use_bias: 是否使用偏置 (default: {False})
           param aggr_method: 邻居聚合方式 (default: {mean})
        """
        super(NeighborAgg, self).__init__()

        self.shape = (input_dim, output_dim)
        self.use_bias = use_bias
        self.aggr_method = aggr_method
        self.weight = self.add_weight(shape=self.shape, initializer='glorot_uniform', name='kernel')
        if self.use_bias:
            self.bias = self.add_weight(shape=self.shape, initializer='zero', name='bias')

    def call(self, neighbor_feature):
        # 聚合邻节点
        if self.aggr_method == 'mean':
            aggr_neighbor = tf.math.reduce_mean(neighbor_feature, axis=1)
        elif self.aggr_method == 'sum':
            aggr_neighbor = tf.math.reduce_sum(neighbor_feature, axis=1)
        elif self.aggr_method == 'max':
            aggr_neighbor = tf.math.reduce_max(neighbor_feature, axis=1)
        else:
            raise ValueError('Unknown aggr type, expected sum, max, or mean, but got {}'
                             .format(self.aggr_method))

        # 全连接权重和偏置
        neighbor_hidden = tf.matmul(aggr_neighbor, self.weight)
        if self.use_bias:
            neighbor_hidden += self.bias

        return neighbor_hidden


class SageGCN(tf.keras.Model):
    def __init__(self, input_dim, hidden_dim,
                 activation=tf.keras.activations.relu,
                 aggr_neighbor_method='mean',
                 aggr_hidden_method='sum'):
        """SageGCN层定义(邻居聚合层和节点全连接成累加或者拼接)：
           param input_dim: 输入特征的维度
           param hidden_dim: 隐层特征的维度，
           param activation: 激活函数
           param aggr_neighbor_method: 邻居特征聚合方法['mean', 'sum', 'max']
           param aggr_hidden_method: 节点特征的更新方法['sum', 'concat']
        """
        super(SageGCN, self).__init__()

        assert aggr_neighbor_method in ['mean', 'sum', 'max']
        assert aggr_hidden_method in ['sum', 'concat']

        self.shape = (input_dim, hidden_dim)
        self.aggr_neighbor_method = aggr_neighbor_method
        self.aggr_hidden_method = aggr_hidden_method
        self.activation = activation
        self.aggregator = NeighborAgg(input_dim, hidden_dim, aggr_method=aggr_neighbor_method)
        self.weight = self.add_weight(shape=self.shape, initializer='glorot_uniform', name='kernel')

    def call(self, src_node_features, neighbor_node_features):
        # 邻居节点特征聚合层
        neighbor_hidden = self.aggregator(neighbor_node_features)
        # 本节点特征全连接层
        self_hidden = tf.matmul(src_node_features, self.weight)
        # 组合层(累计或拼接)
        if self.aggr_hidden_method == 'sum':
            hidden = self_hidden + neighbor_hidden
        elif self.aggr_hidden_method == 'concat':
            hidden = tf.concat(1, [self_hidden, neighbor_hidden])
        else:
            raise ValueError('Expected sum or concat, got {}'.format(self.aggr_hidden))
        # 激活函数
        if self.activation:
            return self.activation(hidden)
        else:
            return hidden


class GraphSage(tf.keras.Model):
    def __init__(self, input_dim, hidden_dims, num_neighbors):
        """SageGCN层定义(邻居聚合层和节点全连接成累加或者拼接)：
           param input_dim: 输入特征的维度
           param hidden_dims: 隐层特征的维度列表，
           param num_neighbors: 每阶采样邻居的节点数
        """
        super(GraphSage, self).__init__()

        # 每阶采样邻居的节点数
        self.num_neighbors = num_neighbors
        # 网络层数
        self.num_layers = len(hidden_dims)
        # 所有gcn层
        self.gcn = []
        dims = [input_dim] + hidden_dims
        acts = [tf.keras.activations.relu] * (self.num_layers - 1) + [tf.keras.activations.softmax]
        for i in range(0, self.num_layers):
            # 上一层的输出即本层输入,最后一层无需激活函数
            self.gcn.append(SageGCN(dims[i], dims[i + 1], activation=acts[i]))

    def call(self, features):
        hidden = features
        # 每次聚合后输入少一层
        for i in range(self.num_layers):
            next_hidden = []
            # 当前gcn网络
            gcn = self.gcn[i]
            # 每层网络采样阶数递减
            for hop in range(len(self.num_neighbors) - i):
                # 源节点集合
                src_node_features = hidden[hop]
                src_node_num = len(src_node_features)
                # 邻居节点集合(添加一个采样数维度)
                neighbor_node_features = tf.reshape(hidden[hop + 1], (src_node_num, self.num_neighbors[hop], -1))
                # 组合结果
                h = gcn(src_node_features, neighbor_node_features)
                next_hidden.append(h)
            hidden = next_hidden

        return hidden[0]


# 模型训练
def train():
    for e in range(EPOCHS):
        for batch in range(NUM_BATCH_PER_EPOCH):
            # 随机取一批节点
            batch_src_index = np.random.choice(train_index, size=(BTACH_SIZE,))
            # 本批节点的标签
            batch_src_label = train_label[batch_src_index].astype(float)
            # 本批节点的邻居节点采样
            batch_sampling_list = multihop_sampling(batch_src_index, NUM_NEIGHBORS_LIST, neighbors_dict)
            # 所有采样节点的特征
            batch_sampling_x = [features[idxs] for idxs in batch_sampling_list]

            loss = 0.0
            with tf.GradientTape() as tape:
                # 模型计算
                batch_train_logits = model(batch_sampling_x)
                # 计算损失函数
                loss = loss_object(batch_src_label, batch_train_logits)
                # 梯度下降
                grads = tape.gradient(loss, model.trainable_variables)
                # 优化器
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

            print('Epoch {:03d} Batch {:03d} Loss: {:.4f}'.format(e, batch, loss))

        # 由于测试时没有分批所以节点选择不能太大否则内存溢出
        acc_list = [loss] + [evaluate(indexs) for indexs in [train_index, val_index, test_index]]
        train_acc.append(acc_list)
        print('Epoch {:03d} train accuracy: {} val accuracy: {} test accuracy:{}'
              .format(e, acc_list[1], acc_list[2], acc_list[3]))

    # 训练过程可视化
    fig, axes = plt.subplots(4, sharex=True, figsize=(12, 8))
    fig.suptitle('Training Metrics')
    names = ['Loss', 'Accuracy', 'Val Acc', 'Test Acc']
    for i in range(len(names)):
        axes[i].set_ylabel(names[i], fontsize=14)
        axes[i].plot(train_acc[:][i])

    plt.show()


# 模型评估
def evaluate(index):
    test_sampling_result = multihop_sampling(index, NUM_NEIGHBORS_LIST, neighbors_dict)
    test_x = [features[idx.astype(np.int32)] for idx in test_sampling_result]
    test_logits = model(test_x)
    test_label = labels[index]
    ll = tf.math.equal(tf.math.argmax(test_label, -1), tf.math.argmax(test_logits, -1))
    accuarcy = tf.reduce_mean(tf.cast(ll, dtype=tf.float32))
    return accuarcy


if __name__ == '__main__':
    # 1、数据读取
    features, labels, neighbors_dict = CoraData().get_data()

    # 分割训练、验证、测试集
    [train_index, val_index, test_index] = [np.arange(i * 500, (i + 1) * 500) for i in range(3)]
    train_label = labels[train_index]

    # 2、模型参数
    INPUT_DIM = 1433  # 输入维度
    # Note: 采样的邻居阶数需要与GCN的层数保持一致
    HIDDEN_DIM = [128, 7]  # 隐藏单元节点数
    NUM_NEIGHBORS_LIST = [10, 10]  # 每阶采样邻居的节点数
    assert len(HIDDEN_DIM) == len(NUM_NEIGHBORS_LIST)
    BTACH_SIZE = 16  # 批处理大小
    EPOCHS = 40
    NUM_BATCH_PER_EPOCH = 40  # 每个epoch循环的批次数

    # 3、模型定义
    model = GraphSage(input_dim=INPUT_DIM, hidden_dims=HIDDEN_DIM,
                      num_neighbors=NUM_NEIGHBORS_LIST)

    # 4、损失函数
    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    # 5、优化器
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01, decay=5e-4)

    # 记录过程值，以便最后可视化
    train_acc = []
    train()
