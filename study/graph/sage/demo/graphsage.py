#!/usr/bin/env python3

import tensorflow as tf

init_fn = tf.keras.initializers.GlorotUniform


class GraphSageBase(tf.keras.Model):
    """
    GraphSage base model outputing embeddings of given nodes
    """

    def __init__(self, raw_features, internal_dim, num_layers, last_has_activ):

        assert num_layers > 0, 'illegal parameter "num_layers"'
        assert internal_dim > 0, 'illegal parameter "internal_dim"'

        super().__init__()
        # call主要作用：根据传入的 node 获得相应的特征
        self.input_layer = RawFeature(raw_features, name="raw_feature_layer")

        self.seq_layers = []
        # 采样层数 num_layers = 2
        for i in range(1, num_layers + 1):
            layer_name = "agg_lv" + str(i)
            # i=1,internal_dim=128
            # else internal_dim=1433
            input_dim = internal_dim if i > 1 else raw_features.shape[-1]
            has_activ = last_has_activ if i == num_layers else True
            # 
            aggregator_layer = MeanAggregator(input_dim
                                              , internal_dim
                                              , name=layer_name
                                              , activ=has_activ
                                              )
            self.seq_layers.append(aggregator_layer)

    def call(self, minibatch):
        """
        :param [node] nodes: target nodes for embedding
        """
        # squeeze: 将原始input中所有维度为1的那些维都删掉的结果
        x = self.input_layer(tf.squeeze(minibatch.src_nodes))
        for aggregator_layer in self.seq_layers:
            x = aggregator_layer(x
                                 , minibatch.dstsrc2srcs.pop()
                                 , minibatch.dstsrc2dsts.pop()
                                 , minibatch.dif_mats.pop()
                                 )
        return x


class GraphSageUnsupervised(GraphSageBase):
    # # raw_features, 128, 2, 1.0
    def __init__(self, raw_features, internal_dim, num_layers, neg_weight):
        super().__init__(raw_features, internal_dim, num_layers, False)
        self.neg_weight = neg_weight

    def call(self, minibatch):
        embABN = tf.math.l2_normalize(super().call(minibatch), 1)

        # 损失函数计算
        embA = tf.gather(embABN, minibatch.dst2batchA)
        embB = tf.gather(embABN, minibatch.dst2batchB)
        embN = tf.boolean_mask(embABN, minibatch.dst2batchN)


        # 1、正样本损失函数计算
        # 边的两端节点对应相乘，求相似度(点乘)
        multiply_ab = tf.multiply(embA, embB)
        # （行求和）
        pos_affinity = tf.reduce_sum(multiply_ab, axis=1)
        pos_label = tf.ones_like(pos_affinity)
        # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
        pos_losses = tf.nn.sigmoid_cross_entropy_with_logits(pos_label, pos_affinity, 'positive_xent')
        # 1维数组求和
        pos_loss = tf.reduce_sum(pos_losses)

        # 2、负样本损失函数计算
        # 每个正样本都和负样本求相似度
        # shape(512,14)
        neg_affinity = tf.matmul(embA, tf.transpose(embN))
        neg_label = tf.zeros_like(neg_affinity)
        # shape(512,14)
        neg_losses = tf.nn.sigmoid_cross_entropy_with_logits(neg_label, neg_affinity, 'negative_xent')
        neg_losse_sum = tf.reduce_sum(neg_losses)
        neg_loss = tf.multiply(self.neg_weight, neg_losse_sum)

        batch_loss = tf.add(pos_loss, neg_loss)
        loss = tf.divide(batch_loss, embA.shape[0])
        self.add_loss(loss)
        return embABN


class GraphSageSupervised(GraphSageBase):
    def __init__(self, raw_features, internal_dim, num_layers, num_classes):
        super().__init__(raw_features, internal_dim, num_layers, True)

        # 分类器
        self.classifier = tf.keras.layers.Dense(num_classes
                                                , activation=tf.nn.softmax
                                                , use_bias=False
                                                , kernel_initializer=init_fn
                                                , name="classifier"
                                                )

    def call(self, minibatch):
        """
        :param [node] nodes: target nodes for embedding
        """
        return self.classifier(super().call(minibatch))


################################################################
#                         Custom Layers                        #
################################################################

class RawFeature(tf.keras.layers.Layer):
    def __init__(self, features, **kwargs):
        """
        :param ndarray((#(node), #(feature))) features: a matrix, each row is feature for a node
        """
        super().__init__(trainable=False, **kwargs)
        self.features = tf.constant(features)

    def call(self, nodes):
        """
        :param [int] nodes: node ids
        """
        return tf.gather(self.features, nodes)


class MeanAggregator(tf.keras.layers.Layer):
    def __init__(self, src_dim, dst_dim, activ=True, **kwargs):
        """
        :param int src_dim: input dimension
        :param int dst_dim: output dimension
        """
        super().__init__(**kwargs)
        self.activ_fn = tf.nn.relu if activ else tf.identity
        self.w = self.add_weight(name=kwargs["name"] + "_weight"
                                 , shape=(src_dim * 2, dst_dim)
                                 , dtype=tf.float32
                                 , initializer=init_fn
                                 , trainable=True
                                 )

    def call(self, dstsrc_features, dstsrc2src, dstsrc2dst, dif_mat):
        """
        :param tensor dstsrc_features: the embedding from the previous layer
        :param tensor dstsrc2dst: 1d index mapping (prepraed by minibatch generator)
        :param tensor dstsrc2src: 1d index mapping (prepraed by minibatch generator)
        :param tensor dif_mat: 2d diffusion matrix (prepraed by minibatch generator)
        """
        dst_features = tf.gather(dstsrc_features, dstsrc2dst)
        src_features = tf.gather(dstsrc_features, dstsrc2src)
        aggregated_features = tf.matmul(dif_mat, src_features)
        concatenated_features = tf.concat([aggregated_features, dst_features], 1)
        x = tf.matmul(concatenated_features, self.w)
        return self.activ_fn(x)
