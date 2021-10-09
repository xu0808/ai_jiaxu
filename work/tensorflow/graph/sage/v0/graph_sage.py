#!/usr/bin/env python3
# 重要疑问：
# 1、邻阶矩阵只是为了实现均值聚合？

import tensorflow as tf
init_fn = tf.keras.initializers.GlorotUniform


class RawFeature(tf.keras.layers.Layer):
    def __init__(self, features, **kwargs):
        super().__init__(trainable=False, **kwargs)
        self.features = tf.constant(features)

    def call(self, nodes):
        """
        :param [int] nodes: node ids
        """
        return tf.gather(self.features, nodes)


class MeanAggregator(tf.keras.layers.Layer):
    def __init__(self, input_dim, out_dim, activ=True, **kwargs):
        super().__init__(**kwargs)
        self.activ_fn = tf.nn.relu if activ else tf.identity
        self.w = self.add_weight(name=kwargs["name"] + "_weight", shape=(input_dim * 2, out_dim)
                                 , dtype=tf.float32, initializer=init_fn, trainable=True)

    def call(self, feature, neighbors, nodes, adj_mat):
        neighbor_feature = tf.gather(feature, nodes)
        node_feature = tf.gather(feature, neighbors)
        agg_feature = tf.matmul(adj_mat, node_feature)
        concat_feature = tf.concat([agg_feature, neighbor_feature], 1)
        x = tf.matmul(concat_feature, self.w)
        return self.activ_fn(x)


class GraphSageBase(tf.keras.Model):
    def __init__(self, feature, input_dim, num_layers, last_has_activ):

        super().__init__()

        self.input_layer = RawFeature(feature, name="raw_feature_layer")

        self.seq_layers = []
        # 采样层数 num_layers = 2
        for i in range(1, num_layers + 1):
            layer_name = "agg_lv" + str(i)
            # i=1,internal_dim=128
            # else internal_dim=1433
            # 原始输入为特征维度，隐藏层均为128
            input_dim = input_dim if i > 1 else feature.shape[-1]
            has_activ = last_has_activ if i == num_layers else True
            aggregator_layer = MeanAggregator(input_dim, input_dim, name=layer_name, activ=has_activ)
            self.seq_layers.append(aggregator_layer)

    def call(self, batch):
        """
        :param [node] nodes: target nodes for embedding
        """
        # squeeze: 将原始input中所有维度为1的那些维都删掉的结果
        x = self.input_layer(tf.squeeze(batch.src_nodes))
        for aggregator_layer in self.seq_layers:
            x = aggregator_layer(x, batch.neighbors_index.pop(), batch.nodes_index.pop(), batch.adj_mats.pop())
        return x


class GraphSageUnsupervised(GraphSageBase):
    # # raw_features, 128, 2, 1.0
    def __init__(self, feature, input_dim, num_layers, neg_weight):
        super().__init__(feature, input_dim, num_layers, False)
        self.neg_weight = neg_weight

    def call(self, batch):
        emb_abn = tf.math.l2_normalize(super().call(batch), 1)
        self.add_loss(compute_uloss(tf.gather(emb_abn, batch.batch_a), tf.gather(emb_abn, batch.batch_b), tf.boolean_mask(emb_abn, batch.batch_n), self.neg_weight))
        return emb_abn


class GraphSageSupervised(GraphSageBase):
    def __init__(self, feature, input_dim, num_layers, num_classes):
        super().__init__(feature, input_dim, num_layers, True)
        # 分类器
        self.classifier = tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax, use_bias=False, kernel_initializer=init_fn, name="classifier")

    def call(self, batch):
        return self.classifier(super().call(batch))


@tf.function
def compute_uloss(emb_a, emb_b, emb_n, neg_weight):
    # positive affinity: pair-wise calculation
    # 边的两端节点对应相乘，求相似度
    pos_affinity = tf.reduce_sum(tf.multiply(emb_a, emb_b), axis=1)

    # negative affinity: enumeration of all combinations of (embeddingA, embeddingN)
    # 每个正样本都和负样本求相似度
    neg_affinity = tf.matmul(emb_a, tf.transpose(emb_n))

    pos_xent = tf.nn.sigmoid_cross_entropy_with_logits(tf.ones_like(pos_affinity)
                                                       , pos_affinity
                                                       , "positive_xent")
    # p1:[1,2,3],p2:[[0.1,0.5,0.4]]
    neg_xent = tf.nn.sigmoid_cross_entropy_with_logits(tf.zeros_like(neg_affinity)
                                                       , neg_affinity
                                                       , "negative_xent")

    weighted_neg = tf.multiply(neg_weight, tf.reduce_sum(neg_xent))
    batch_loss = tf.add(tf.reduce_sum(pos_xent), weighted_neg)

    # per batch loss: GraphSAGE:models.py line 378
    return tf.divide(batch_loss, emb_a.shape[0])
