#!/usr/bin/env python
# coding: utf-8
# 模型
import tensorflow as tf


@tf.function
def compute_uloss(emb_a, emb_b, emb_n, neg_weight):
    # positive affinity: pair-wise calculation
    # 正负样本相似度
    pos_affinity = tf.reduce_sum(tf.multiply(emb_a, emb_b), axis=1)
    # 每个正样本都和负样本求相似度
    neg_affinity = tf.matmul(emb_a, tf.transpose(emb_n))
    # 交叉熵
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits
    pos_xent = cross_entropy(tf.ones_like(pos_affinity), pos_affinity, 'positive_xent')
    neg_xent = cross_entropy(tf.zeros_like(neg_affinity), neg_affinity, 'negative_xent')
    weighted_neg = tf.multiply(neg_weight, tf.reduce_sum(neg_xent))
    batch_loss = tf.add(tf.reduce_sum(pos_xent), weighted_neg)

    return tf.divide(batch_loss, emb_a.shape[0])


class Neighbor_Agg(tf.keras.Model):
    def __init__(self, input_dim, output_dim, use_bias=False, agg_method='mean'):
        """聚合邻居节点
           param input_dim: 输入特征的维度
           param output_dim: 输出特征的维度
           param use_bias: 是否使用偏置 (default: {False})
           param aggr_method: 邻居聚合方式 (default: {mean})
        """
        super(Neighbor_Agg, self).__init__()

        self.shape = (input_dim, output_dim)
        self.use_bias = use_bias
        self.agg_method = agg_method
        self.weight = self.add_weight(shape=self.shape, initializer='glorot_uniform', name='kernel')
        if self.use_bias:
            self.bias = self.add_weight(shape=self.shape, initializer='zero', name='bias')

    def call(self, neighbor_feature):
        # 聚合邻节点
        if self.agg_method == 'mean':
            agg = tf.math.reduce_mean(neighbor_feature, axis=1)
        elif self.agg_method == 'sum':
            agg = tf.math.reduce_sum(neighbor_feature, axis=1)
        elif self.agg_method == 'max':
            agg = tf.math.reduce_max(neighbor_feature, axis=1)
        else:
            raise ValueError('Unknown aggr type, expected sum, max, or mean, but got {}'
                             .format(self.agg_method))

        # 全连接权重和偏置
        hidden = tf.matmul(agg, self.weight)
        return hidden + self.bias if self.use_bias else hidden


class SageGCN(tf.keras.Model):
    def __init__(self, input_dim, hidden_dim, act=tf.keras.activations.relu,
                 agg_method='mean', hidden_method='sum'):
        """SageGCN层定义(邻居聚合层和节点全连接成累加或者拼接)：
           param input_dim: 输入特征的维度
           param hidden_dim: 隐层特征的维度，
           param activation: 激活函数
           param agg_method: 邻居特征聚合方法['mean', 'sum', 'max']
           param hidden_method: 节点特征的更新方法['sum', 'concat']
        """
        super(SageGCN, self).__init__()

        assert agg_method in ['mean', 'sum', 'max']
        assert hidden_method in ['sum', 'concat']

        self.shape = (input_dim, hidden_dim)
        self.agg_method = agg_method
        self.agg_method = hidden_method
        self.act = act
        self.aggregator = Neighbor_Agg(input_dim, hidden_dim, agg_method=agg_method)
        self.weight = self.add_weight(shape=self.shape, initializer='glorot_uniform', name='kernel')

    def call(self, node_feature, neighbor_feature):
        # 邻居节点特征聚合层
        neighbor_hidden = self.aggregator(neighbor_feature)
        # 本节点特征全连接层
        self_hidden = tf.matmul(node_feature, self.weight)
        # 组合层(累计或拼接)
        if self.agg_method == 'sum':
            hidden = self_hidden + neighbor_hidden
        elif self.agg_method == 'concat':
            hidden = tf.concat(1, [self_hidden, neighbor_hidden])
        else:
            raise ValueError('Expected sum or concat, got {}'.format(self.aggr_hidden))
        # 激活函数
        return self.act(hidden) if self.act else hidden


class GraphSage(tf.keras.Model):
    def __init__(self, input_dim, hidden_dim, sample_sizes, num_class=None):
        """SageGCN层定义(邻居聚合层和节点全连接成累加或者拼接)：
           param input_dim: 输入特征的维度
           param hidden_dims: 隐层特征的维度列表，
           param sample_sizes: 每阶采样邻居的节点数
        """
        super(GraphSage, self).__init__()

        # 每阶采样邻居的节点数
        self.num_neighbors = sample_sizes
        # 网络层数
        self.num_layers = len(sample_sizes)
        # 分类器
        init_fn = tf.keras.initializers.GlorotUniform
        self.classifier = None
        if num_class:
            self.classifier = tf.keras.layers.Dense(num_class, activation=tf.nn.softmax, use_bias=False,
                                                    kernel_initializer=init_fn, name='classifier')

        # 所有gcn层
        self.gcn = []
        for i in range(0, self.num_layers):
            # 第一层输入维度为特征维度，其它为隐藏层维度
            input_size = input_dim if i == 0 else hidden_dim
            self.gcn.append(SageGCN(input_size, hidden_dim, act=tf.keras.activations.relu))

    def call(self, feature):
        hidden = feature
        # 每次聚合后输入少一层
        for i in range(self.num_layers):
            next_hidden = []
            # 当前gcn网络
            gcn = self.gcn[i]
            # 每层网络采样阶数递减
            for hop in range(len(self.num_neighbors) - i):
                # 源节点集合
                node_feature = hidden[hop]
                node_num = len(node_feature)
                # 邻居节点集合(添加一个采样数维度)
                neighbor_feature = tf.reshape(hidden[hop + 1], (node_num, self.num_neighbors[hop], -1))
                # 组合结果
                h = gcn(node_feature, neighbor_feature)
                next_hidden.append(h)
            hidden = next_hidden

        # 模型输出
        output = hidden[0]
        # 多分类输出
        if self.classifier:
            output = self.classifier(output)
        return output


class GraphSage_unsup(GraphSage):
    def __init__(self, input_dim, hidden_dims, sample_sizes, neg_weight):
        super().__init__(input_dim, hidden_dims, sample_sizes)
        self.neg_weight = neg_weight

    def call(self, feature, batch_a, batch_b, batch_n):
        out_0 = super().call(feature)
        # 输出正则化之后的emb
        emb_abn = tf.math.l2_normalize(out_0, 1)
        # 取出每个样本向量
        [emb_a, emb_b, emb_n] = [tf.gather(emb_abn, b) for b in [batch_a, batch_b, batch_n]]
        self.add_loss(compute_uloss(emb_a, emb_b, emb_n, self.neg_weight))
        return emb_abn