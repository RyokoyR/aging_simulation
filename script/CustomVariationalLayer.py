#!/lustre7/home/lustre4/ryoyokosaka/python/.pyenv/shims
import sys
sys.path.append('/lustre7/home/lustre4/ryoyokosaka/.pyenv/versions/3.6.0/lib/python3.6/site-packages')

import os
import tensorflow as tf
from keras.layers import Input, Dense, Lambda, Layer, Activation, merge
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential
from keras import backend as K
from keras import metrics, optimizers
from keras.callbacks import Callback
import keras

import pandas as pd

import optuna_objective_for_Tybalt

rnaseq_file = os.path.join('/home/ryoyokosaka/python/1_54rnaseq_drop.txt')
rnaseq_df = pd.read_table(rnaseq_file, index_col=0)

original_dim = rnaseq_df.shape[1]

class CustomVariationalLayer(Layer):
    def __init__(self, var_layer, mean_layer, **kwargs):
        # https://keras.io/layers/writing-your-own-keras-layers/
        self.is_placeholder = True
        self.var_layer = var_layer
        self.mean_layer = mean_layer
        super(CustomVariationalLayer, self).__init__(**kwargs) #superはクラスの多重継承(3つ以上のクラスの継承)

    def vae_loss(self, x_input, x_decoded):

        reconstruction_loss = original_dim * metrics.binary_crossentropy(x_input, x_decoded)
        kl_loss = - 0.5 * K.sum(1 + self.var_layer - K.square(self.mean_layer) -
                                K.exp(self.var_layer), axis=-1)
        return K.mean(reconstruction_loss + (K.get_value(optuna_objective_for_Tybalt.beta) * kl_loss))

    def call(self, inputs):
        x = inputs[0]
        x_decoded = inputs[1]
        loss = self.vae_loss(x, x_decoded)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return x
