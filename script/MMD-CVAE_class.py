#!/lustre7/home/lustre4/ryoyokosaka/python/.pyenv/shims
import sys
sys.path.append('/lustre7/home/lustre4/ryoyokosaka/python/')

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras.layers import Input, Dense, Lambda, Layer, Activation, merge
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model, Sequential
from keras import backend as K
from keras import metrics, optimizers
from keras.callbacks import Callback
import keras
import optuna
import pydot
import graphviz
from keras.utils import plot_model
#from keras_tqdm import TQDMNotebookCallback
#from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from sklearn.cluster import KMeans
from sklearn import manifold
from sklearn import datasets
from matplotlib.cm import get_cmap
from sklearn.metrics import explained_variance_score
import plotly.offline as offline
import h5py
import sampling,CustomVariationalLayer,rnaseqdata_1_54_import,WarmUpCallback

np.random.seed(123)

class MMDCVAE():
        def __init__(self, original_dim,first_hidden_dim,second_hidden_dim,third_hidden_dim,latent_dim,label_dim,batch_size,epochs,learning_rate,kappa,beta,dr_rate):
                self.original_dim = original_dim
                self.first_hidden_dim = first_hidden_dim
                self.second_hidden_dim= second_hidden_dim
                self.third_hidden_dim = third_hidden_dim
                self.latent_dim = latent_dim
                self.label_dim = label_dim
                self.batch_size = batch_size
                self.epochs = epochs
                self.learning_rate = learning_rate
                self.kappa = kappa
                self.beta = beta
                self.dr_rate = dr_rate
                self.kernel_method = kwargs.get("kernel", "multi-scale-rbf")
                self.mmd_computation_way = kwargs.get("mmd_computation_way", "general")
                self.clip_value = kwargs.get('clip_value', 3.0)
                #self.layer_depth = layer_depth
                self.input = Input(shape=(self.original_dim,), name="data")
                self.z = Input(shape=(self.latent_dim,), name="latent_data")
                self.encoder_labels = Input(shape=(self.label_dim,), name="encoder_labels")
                self.decoder_labels = Input(shape=(self.label_dim,), name="decoder_labels")
        def build_encoder_layer(self):
                #global z
                #self.rnaseq_input = Input(shape=(self.original_dim, ))
                #self.y_label_input = Input(shape=(1,))
                #self.merged_encode = keras.layers.concatenate([self.rnaseq_input,self.y_label_input],axis=-1)
                encoder_input = concatenate([self.input,self.encoder_labels], axis = 1)
                h = Dense(self.first_hidden_dim, kernel_initializer='glorot_uniform', use_bias=False)(encoder_input)
                h = BatchNormalization(axis=1, scale=True)(h)
                h = LeakyReLU(name="mmd")(h)
                h = Dropout(self.dr_rate)(h)
                h = Dense(self.second_hidden_dim, kernel_initializer='glorot_uniform', use_bias=False)(h)
                h = BatchNormalization(axis=1, scale=True)(h)
                h = LeakyReLU()(h)
                h = Dropout(self.dr_rate)(h)
                h = Dense(self.third_hidden_dim, kernel_initializer='glorot_uniform', use_bias=False)(h)
                h = BatchNormalization(axis=1, scale=True)(h)
                h = LeakyReLU()(h)
                h = Dropout(self.dr_rate)(h)
                mean = Dense(self.latent_dim, kernel_initializer='glorot_uniform')(h)
                log_var = Dense(self.latent_dim, kernel_initializer='glorot_uniform')(h)
                z = Lambda(sampling.sampling, output_shape=(self.latent_dim,))([mean, log_var])
                encoder_model = Model(inputs=[self.input, self.encoder_labels], outputs=z)

                return mean,log_var,encoder_model

        def build_decoder_layer(self):

                #self.merged_decode = keras.layers.concatenate([z,self.y_label_input], axis=-1) #merged_decodeはデータフレームじゃなさそうなのでエラーでるかも
                #変更点
                latent_variable_label_input = concatenate([self.z,self.decoder_label],axis = -1)
                #self.latent_variable_label_input = Input(shape=(self.latent_dim + 1, ))
                h = Dense(self.third_hidden_dim, kernel_initializer='glorot_uniform', use_bias=False)(latent_variable_label_input)
                h = BatchNormalization(axis=1, scale=True)(h)
                h = LeakyReLU(name="mmd")(h)
                h = Dropout(self.dr_rate)(h)
                h = Dense(self.second_hidden_dim, kernel_initializer='glorot_uniform', use_bias=False)(h)
                h = BatchNormalization(axis=1, scale=True)(h)
                h = LeakyReLU()(h)
                h = Dropout(self.dr_rate)(h)
                h = Dense(self.first_hidden_dim, kernel_initializer='glorot_uniform', use_bias=False)(h)
                h = BatchNormalization(axis=1, scale=True)(h)
                h = LeakyReLU()(h)
                h = Dropout(self.dr_rate)(h)
                h = Dense(self.original_dim, kernel_initializer='glorot_uniform', use_bias=True)(h)
                h = Activation('sigmoid')(h)
                #今までの実装では活性化関数は"sigmoid"だった(遺伝子発現データにはシグモイド関数による活性化が適している?)が、trVAEでは"relu"になってた。optunaに任せて最適化することも考えておく。
                self.decoder_model = Model([self.z,self.decoder_label],h)
                self.decoder_mmd_model = Model([self.z,self.decoder_label],self.z)
                return self.decoder_model, self.decoder_mmd_model

        def create_network(self):
                inputs = [self.input, self.encoder_labels, self.decoder_labels]
                self.mu, self.log_var, self.encoder_model = self.build_encoder_layer()
                self.decoder_model, self.decoder_mmd_model = self.build_decoder_layer()
                decoder_output = self.decoder_model([self.encoder_model(inputs[:2])[2], self.decoder_labels])
                mmd_output = self.decoder_mmd_model([self.encoder_model(inputs[:2])[2], self.decoder_labels])

                reconstruction_output = Lambda(lambda x: x, name="kl_mse")(decoder_output)
                mmd_output = Lambda(lambda x: x, name="mmd")(mmd_output)

                self.cvae_model = Model(inputs=inputs,outputs=[reconstruction_output, mmd_output])

        def kl_loss(mu, log_var, alpha=0.1):
                def kl_recon_loss(y_true, y_pred):
                        kl_loss = 0.5 * K.mean(K.exp(log_var) + K.square(mu) - 1. - log_var, 1)
                        kl_loss_x = alpha * kl_loss
                        return tf.where(tf.is_nan(kl_loss_x), tf.zeros_like(kl_loss_x) + np.inf, kl_loss_x)

                return kl_recon_loss

        def compute_mmd(x, y, kernel, **kwargs):
                x_kernel = compute_kernel(x, x, kernel=kernel, **kwargs)
                y_kernel = compute_kernel(y, y, kernel=kernel, **kwargs)
                xy_kernel = compute_kernel(x, y, kernel=kernel, **kwargs)

                return K.mean(x_kernel) + K.mean(y_kernel) - 2 * K.mean(xy_kernel)

        def mmd(label_dim, beta, kernel_method='multi-scale-rbf', computation_way="general"):
                def mmd_loss(real_labels, y_pred):
                        with tf.variable_scope("mmd_loss", reuse=tf.AUTO_REUSE):
                                real_labels = K.reshape(K.cast(real_labels, 'int32'), (-1,))
                                conditions_mmd = tf.dynamic_partition(y_pred, real_labels, num_partitions=label_dim)
                                loss = 0.0
                                if computation_way.isdigit():
                                        boundary = int(computation_way)
                                        for i in range(boundary):
                                                for j in range(boundary, n_conditions):
                                                        loss += self.compute_mmd(conditions_mmd[i], conditions_mmd[j], kernel_method)
                                else:
                                        for i in range(len(conditions_mmd)):
                                                for j in range(i):
                                                        loss += self.compute_mmd(conditions_mmd[j], conditions_mmd[j + 1], kernel_method)
                        mmd_x = beta * loss
                        return tf.where(tf.is_nan(mmd_x), tf.zeros_like(mmd_x), mmd_x)

                return mmd_loss

        def _calculate_loss(self):
                loss = self.kl_loss(self.mu, self.log_var, self.alpha, self.eta)
                mmd_loss = mmd_loss(self.label_dim, self.beta, self.kernel_method, self.mmd_computation_way)

                return loss, mmd_loss

        def _loss_function(self):
                loss, mmd_loss = self._calculate_loss()
                self.cvae_optimizer = keras.optimizers.Adam(lr=self.learning_rate, clipvalue=self.clip_value)
                self.cvae_model.compile(optimizer=self.cvae_optimizer,loss=[loss, mmd_loss],metrics={self.cvae_model.outputs[0].name: loss,self.cvae_model.outputs[1].name: mmd_loss})


        def compile_mmdcvae(self):

                adam = optimizers.Adam(lr=self.learning_rate)
                mmdcvae_layer = CustomVariationalLayer.CustomVariationalLayer(self.z_log_var_encoded,self.z_mean_encoded)([self.rnaseq_input, self.rnaseq_reconstruct])
                self.mmdcvae = Model([self.rnaseq_input,self.y_label_input], mmdcvae_layer)
                self.mmdcvae.compile(optimizer=adam, loss=None, loss_weights=[self.beta])

                return self.mmdcvae

        def get_summary(self):
                self.mmdcvae.summary()

        def visualize_architecture(self, output_file):

                plot_model(self.mmdcvae, to_file=output_file)
                #SVG(model_to_dot(self.cvae).create(prog='dot', format='svg'))

        def train_mmdcvae(self):
                self.hist = self.mmdcvae.fit([np.array(rnaseqdata_1_54_import.rnaseq_df_train),np.array(rnaseqdata_1_54_import.age_label_train)],
                                  shuffle=True,
                                  epochs=self.epochs,
                                  batch_size=self.batch_size,
                                  validation_data=([np.array(rnaseqdata_1_54_import.rnaseq_df_test),np.array(rnaseqdata_1_54_import.age_label_test)],None),
                                  callbacks=[WarmUpCallback.WarmUpCallback(self.beta, self.kappa)])
                return self.hist

        def get_mmdcvae_loss(self):

                loss=float(self.hist.history['val_loss'][-1])

                return loss

        def visualize_training(self, output_file):

                history_df = pd.DataFrame(self.hist.history)
                ax = history_df.plot()
                ax.set_xlabel('Epochs')
                ax.set_ylabel('VAE Loss')
                fig = ax.get_figure()
                fig.savefig(output_file)

                return fig

        def compress(self, df,age_label):
                # エンコーディングするオブジェクト    #dfにはラベルと合わせたrna-seqデータを入力してください。
                self.encoder = Model([self.rnaseq_input,self.y_label_input], z)
                encoded_df = self.encoder.predict_on_batch([df,label_df])
                encoded_df = pd.DataFrame(encoded_df, columns=range(1, self.latent_dim + 1),
                                  index=rnaseq_df.index)
                return encoded_df

        def get_decoder_weights(self):

                decoder_input = Input(shape=(self.latent_dim, ))  # can generate from any sampled z vector
                _x_decoded_mean = self.decoder_model(decoder_input)
                self.decoder = Model(decoder_input, _x_decoded_mean)
                weights = []
                for layer in self.decoder.layers:
                        weights.append(layer.get_weights())
                return(weights)

        def reconstruct(self,df):
                self.decoder = Model(self.latent_variable_label_input,self.decoder_to_reconstruct)
                rna_reconstruction = self.decoder.predict(df)
                rna_reconstruction_df = pd.DataFrame(rna_reconstruction,columns = rnaseq_df.columns,index = rnaseq_df.index)

                return rna_reconstruction_df

        def predict(self, df):
                return self.rnaseq_reconstruct.predict(np.array(df))

        def save_models(self, encoder_file_path, decoder_file_path):
                self.encoder.save(encoder_file_path)
                self.decoder.save(decoder_file_path)
