

import tensorflow as tf

learning_rate = 0.001


class EGES(tf.keras.Model):
    def __init__(self, num_nodes, num_featrue, feature_lens, n_sample=100, emb_dim=128):
        
        super(EGES, self).__init__()

        self.shape = (num_nodes, emb_dim)
        self.num_nodes = num_nodes
        self.num_featrue = num_featrue
        self.feature_lens = feature_lens
        self.n_sample = n_sample
        self.emb_dim = emb_dim

        self.weight = self.add_weight(shape=self.shape, initializer='glorot_uniform', name='kernel')
        self.bias = self.add_weight(shape=self.shape, initializer='zero', name='bias')
        
        # 所有emb层
        self.feature_emb_layers = []
        for i in range(num_featrue):
            layers = tf.keras.layers.Embedding(input_dim=self.feature_lens[i]+1, output_dim=emb_dim)
            self.feature_emb_layers.append(layers)

        self.node_emb_layers = tf.keras.layers.Embedding(input_dim=self.feature_lens[0]+1, output_dim=emb_dim)

    def call(self, feature, label):
        cat_emb_list = []
        for i in range(self.num_featrue):
            cat_emb_list.append(self.feature_emb_layers[i](feature[:, i]))
        cat_embed = tf.stack(cat_emb_list, axis=-1)
        # attention merge
        alpha_embed = self.node_emb_layers(feature[:, 0])
        alpha_embed_expand = tf.expand_dims(alpha_embed, 2)
        alpha_i_sum = tf.reduce_sum(tf.exp(alpha_embed_expand), axis=-1)
        merge_emb = tf.reduce_sum(cat_embed * tf.exp(alpha_embed_expand), axis=-1) / alpha_i_sum

        # 损失函数
        self.add_loss(self.skipgram_loss(merge_emb, label))

        return merge_emb

    def skipgram_loss(self, merge_emb, label):
        sampled_values = tf.random.uniform_candidate_sampler(
            true_classes=tf.cast(label, tf.int64),
            num_true=1, num_sampled=self.n_sample,
            unique=True, range_max=self.num_nodes
        )

        softmax_loss = tf.nn.sampled_softmax_loss(
            weights=self.weights,
            biases=self.bias,
            labels=label,
            inputs=merge_emb,
            num_sampled=self.n_sample,
            num_classes=self.num_nodes,
            num_true=1,
            sampled_values=sampled_values
        )
        loss = tf.reduce_mean(softmax_loss)
        return loss
