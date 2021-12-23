#!/usr/bin/env python
# coding: utf-8


from layer import AFM_layer
from tensorflow.keras.models import Model


class AFM(Model):
    def __init__(self, feature_columns, mode):
        super().__init__()
        self.afm_layer = AFM_layer(feature_columns, mode)

    def call(self, inputs, training=None, mask=None):
        output = self.afm_layer(inputs)
        return output
