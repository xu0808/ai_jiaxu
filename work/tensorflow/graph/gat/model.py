#!/usr/bin/env python
# coding: utf-8
# 模型定义

import tensorflow as tf

l2_coef = 0.0005


class sp_attn_head(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, nb_nodes, in_drop=0.0, coef_drop=0.0, activation=tf.nn.elu, residual=False):
        super(sp_attn_head, self).__init__()
        self.hidden_dim = hidden_dim
        self.nb_nodes = nb_nodes
        self.activation = activation
        self.residual = residual

        self.in_dropout = tf.keras.layers.Dropout(in_drop)
        self.coef_dropout = tf.keras.layers.Dropout(coef_drop)

        self.conv_no_bias = tf.keras.layers.Conv1D(hidden_dim, 1, use_bias=False)
        self.conv_f1 = tf.keras.layers.Conv1D(1, 1)
        self.conv_f2 = tf.keras.layers.Conv1D(1, 1)

        self.conv_residual = tf.keras.layers.Conv1D(hidden_dim, 1)
        self.bias_zero = tf.Variable(tf.zeros(hidden_dim))

    def __call__(self, seq, bias_mat, training):

        adj_mat = bias_mat

        seq = self.in_dropout(seq, training=training)
        seq_fts = self.conv_no_bias(seq)
        f_1 = self.conv_f1(seq_fts)
        f_2 = self.conv_f2(seq_fts)

        f_1 = tf.reshape(f_1, (self.nb_nodes, 1))
        f_2 = tf.reshape(f_2, (self.nb_nodes, 1))
        f_1 = adj_mat * f_1
        f_2 = adj_mat * tf.transpose(f_2, [1, 0])

        logits = tf.compat.v1.sparse_add(f_1, f_2)
        lrelu = tf.SparseTensor(indices=logits.indices,
                                values=tf.nn.leaky_relu(logits.values),
                                dense_shape=logits.dense_shape)
        coefs = tf.compat.v2.sparse.softmax(lrelu)

        if training != False:
            coefs = tf.SparseTensor(indices=coefs.indices,
                                    values=self.coef_dropout(coefs.values, training=training),
                                    dense_shape=coefs.dense_shape)
            seq_fts = self.in_dropout(seq_fts, training=training)

        coefs = tf.compat.v2.sparse.reshape(coefs, [nb_nodes, nb_nodes])

        seq_fts = tf.squeeze(seq_fts)
        vals = tf.sparse.sparse_dense_matmul(coefs, seq_fts)
        vals = tf.expand_dims(vals, axis=0)
        vals.set_shape([1, self.nb_nodes, self.hidden_dim])

        ret = vals + self.bias_zero

        if self.residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + self.conv_residual(seq)
            else:
                ret = ret + seq
        return self.activation(ret)


class attn_head(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, nb_nodes=None, in_drop=0.0, coef_drop=0.0, activation=tf.nn.elu, residual=False):
        super(attn_head, self).__init__()
        self.activation = activation
        self.residual = residual

        self.in_dropout = tf.keras.layers.Dropout(in_drop)
        self.coef_dropout = tf.keras.layers.Dropout(coef_drop)

        self.conv_no_bias = tf.keras.layers.Conv1D(hidden_dim, 1, use_bias=False)
        self.conv_f1 = tf.keras.layers.Conv1D(1, 1)
        self.conv_f2 = tf.keras.layers.Conv1D(1, 1)

        self.conv_residual = tf.keras.layers.Conv1D(hidden_dim, 1)
        self.bias_zero = tf.Variable(tf.zeros(hidden_dim))

    def __call__(self, seq, bias_mat, training):

        seq = self.in_dropout(seq, training=training)
        seq_fts = self.conv_no_bias(seq)
        f_1 = self.conv_f1(seq_fts)
        f_2 = self.conv_f2(seq_fts)

        logits = f_1 + tf.transpose(f_2, [0, 2, 1])
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)

        coefs = self.coef_dropout(coefs, training=training)
        seq_fts = self.in_dropout(seq_fts, training=training)

        vals = tf.matmul(coefs, seq_fts)
        vals = tf.cast(vals, dtype=tf.float32)
        ret = vals + self.bias_zero

        if self.residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + self.conv_residual(seq)
            else:
                ret = ret + seq
        return self.activation(ret)


def choose_attn_head(Sparse):
    if Sparse:
        chosen_attention = sp_attn_head
    else:
        chosen_attention = attn_head

    return chosen_attention


class inference(tf.keras.layers.Layer):
    def __init__(self, n_heads, hid_units, nb_classes, nb_nodes, Sparse, ffd_drop=0.0, attn_drop=0.0,
                 activation=tf.nn.elu, residual=False):
        super(inference, self).__init__()

        attned_head = choose_attn_head(Sparse)

        self.attns = []
        self.sec_attns = []
        self.final_attns = []
        self.final_sum = n_heads[-1]

        for i in range(n_heads[0]):
            self.attns.append(attned_head(hidden_dim=hid_units[0], nb_nodes=nb_nodes,
                                          in_drop=ffd_drop, coef_drop=attn_drop,
                                          activation=activation,
                                          residual=residual))

        for i in range(1, len(hid_units)):
            # h_old = h_1
            sec_attns = []
            for j in range(n_heads[i]):
                sec_attns.append(attned_head(hidden_dim=hid_units[i], nb_nodes=nb_nodes,
                                             in_drop=ffd_drop, coef_drop=attn_drop,
                                             activation=activation,
                                             residual=residual))
                self.sec_attns.append(sec_attns)

        for i in range(n_heads[-1]):
            self.final_attns.append(attned_head(hidden_dim=nb_classes, nb_nodes=nb_nodes,
                                                in_drop=ffd_drop, coef_drop=attn_drop,
                                                activation=lambda x: x,
                                                residual=residual))

    def __call__(self, inputs, bias_mat, training):
        first_attn = []
        out = []

        for indiv_attn in self.attns:
            first_attn.append(indiv_attn(seq=inputs, bias_mat=bias_mat, training=training))
        h_1 = tf.concat(first_attn, axis=-1)

        for sec_attns in self.sec_attns:
            next_attns = []
            for indiv_attns in sec_attns:
                next_attns.append(indiv_attn(seq=h_1, bias_mat=bias_mat, training=training))
            h_1 = tf.concat(next_attns, axis=-1)

        for indiv_attn in self.final_attns:
            out.append(indiv_attn(seq=h_1, bias_mat=bias_mat, training=training))
        logits = tf.add_n(out) / self.final_sum

        return logits


class GAT(tf.keras.Model):
    def __init__(self, hid_units, n_heads, nb_classes, nb_nodes, Sparse, ffd_drop=0.0, attn_drop=0.0,
                 activation=tf.nn.elu, residual=False):
        super(GAT, self).__init__()
        '''
        hid_units: This is the number of hidden units per each attention head in each layer (8). Array of hidden layer dimensions
        n_heads: This is the additional entry of the output layer [8,1]. More specifically the output that calculates attn    
        nb_classes: This refers to the number of classes (7)
        nb_nodes: This refers to the number of nodes (2708)    
        activation: This is the activation function tf.nn.elu
        residual: This determines whether we add seq to ret (False)
        '''
        self.hid_units = hid_units  # [8]
        self.n_heads = n_heads  # [8,1]
        self.nb_classes = nb_classes
        self.nb_nodes = nb_nodes
        self.activation = activation
        self.residual = residual

        self.inferencing = inference(n_heads, hid_units, nb_classes, nb_nodes, Sparse=Sparse, ffd_drop=ffd_drop,
                                     attn_drop=attn_drop, activation=activation, residual=residual)

    ##########################
    # Adapted from tkipf/gcn #
    ##########################

    def masked_softmax_cross_entropy(self, logits, labels, mask):
        """Softmax cross-entropy loss with masking."""
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        loss *= mask
        return tf.reduce_mean(loss)

    def masked_sigmoid_cross_entropy(self, logits, labels, mask):
        """Softmax cross-entropy loss with masking."""
        labels = tf.cast(labels, dtype=tf.float32)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
        loss = tf.reduce_mean(loss, axis=1)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        loss *= mask
        return tf.reduce_mean(loss)

    def masked_accuracy(self, logits, labels, mask):
        """Accuracy with masking."""
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        accuracy_all = tf.cast(correct_prediction, tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        accuracy_all *= mask
        return tf.reduce_mean(accuracy_all)

    def micro_f1(self, logits, labels, mask):
        """Accuracy with masking."""
        predicted = tf.round(tf.nn.sigmoid(logits))

        # Use integers to avoid any nasty FP behaviour
        predicted = tf.cast(predicted, dtype=tf.int32)
        labels = tf.cast(labels, dtype=tf.int32)
        mask = tf.cast(mask, dtype=tf.int32)

        # expand the mask so that broadcasting works ([nb_nodes, 1])
        mask = tf.expand_dims(mask, -1)

        # Count true positives, true negatives, false positives and false negatives.
        tp = tf.count_nonzero(predicted * labels * mask)
        tn = tf.count_nonzero((predicted - 1) * (labels - 1) * mask)
        fp = tf.count_nonzero(predicted * (labels - 1) * mask)
        fn = tf.count_nonzero((predicted - 1) * labels * mask)

        # Calculate accuracy, precision, recall and F1 score.
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        fmeasure = (2 * precision * recall) / (precision + recall)
        fmeasure = tf.cast(fmeasure, tf.float32)
        return fmeasure

    def __call__(self, inputs, training, bias_mat, lbl_in, msk_in):
        logits = self.inferencing(inputs=inputs, bias_mat=bias_mat, training=training)

        log_resh = tf.reshape(logits, [-1, self.nb_classes])
        lab_resh = tf.reshape(lbl_in, [-1, self.nb_classes])
        msk_resh = tf.reshape(msk_in, [-1])

        loss = self.masked_softmax_cross_entropy(log_resh, lab_resh, msk_resh)

        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in self.trainable_variables if v.name not
                           in ['bias', 'gamma', 'b', 'g', 'beta']]) * l2_coef

        loss = loss + lossL2
        accuracy = self.masked_accuracy(log_resh, lab_resh, msk_resh)

        return logits, accuracy, loss