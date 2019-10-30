import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras.layers import Input, Dense, Lambda, Layer, Activation, merge
from keras.layers.normalization import BatchNormalization
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

class Tybalt():
        def __init__(self, original_dim,first_hidden_dim,second_hidden_dim,latent_dim,batch_size,epochs,learning_rate,kappa,beta,layer_depth):
                self.original_dim = original_dim
                self.first_hidden_dim = first_hidden_dim
                self.second_hidden_dim= second_hidden_dim
                self.latent_dim = latent_dim
                self.batch_size = batch_size
                self.epochs = epochs
                self.learning_rate = learning_rate
                self.kappa = kappa
                self.beta = beta
                self.layer_depth = layer_depth

        def build_encoder_layer(self):
                global z
                self.rnaseq_input = Input(shape=(self.original_dim, ))
                self.y_label_input = Input(shape=(1,))
                self.merged_encode = keras.layers.concatenate([self.rnaseq_input,self.y_label_input],axis=-1)




                if self.layer_depth == 1:
                        z_mean_dense_linear = Dense(self.latent_dim, kernel_initializer='glorot_uniform')(self.merged_encode)
                        z_mean_dense_batchnorm = BatchNormalization()(z_mean_dense_linear)
                        self.z_mean_encoded = Activation('relu')(z_mean_dense_batchnorm)

                        z_log_var_dense_linear = Dense(self.latent_dim, kernel_initializer='glorot_uniform')(self.merged_encode)
                        z_log_var_dense_batchnorm = BatchNormalization()(z_log_var_dense_linear)
                        self.z_log_var_encoded = Activation('relu')(z_log_var_dense_batchnorm)


                        z = Lambda(sampling.sampling, output_shape=(self.latent_dim, ))([self.z_mean_encoded, self.z_log_var_encoded])

                elif self.layer_depth == 2:
                        first_hidden_dense_linear = Dense(self.first_hidden_dim, kernel_initializer='glorot_uniform')(self.merged_encode)
                        first_hidden_dense_batchnorm = BatchNormalization()(first_hidden_dense_linear)
                        first_hidden_encoded = Activation('relu')(first_hidden_dense_batchnorm)

                        z_mean_dense_linear = Dense(self.latent_dim, kernel_initializer='glorot_uniform')(first_hidden_encoded)
                        z_mean_dense_batchnorm = BatchNormalization()(z_mean_dense_linear)
                        self.z_mean_encoded = Activation('relu')(z_mean_dense_batchnorm)

                        z_log_var_dense_linear = Dense(self.latent_dim, kernel_initializer='glorot_uniform')(first_hidden_encoded)
                        z_log_var_dense_batchnorm = BatchNormalization()(z_log_var_dense_linear)
                        self.z_log_var_encoded = Activation('relu')(z_log_var_dense_batchnorm)

                        z = Lambda(sampling.sampling, output_shape=(self.latent_dim, ))([self.z_mean_encoded, self.z_log_var_encoded])

                elif self.layer_depth == 3:
                        first_hidden_dense_linear = Dense(self.first_hidden_dim, kernel_initializer='glorot_uniform')(self.merged_encode)
                        first_hidden_dense_batchnorm = BatchNormalization()(first_hidden_dense_linear)
                        first_hidden_encoded = Activation('relu')(first_hidden_dense_batchnorm)

                        second_hidden_dense_linear = Dense(self.second_hidden_dim, kernel_initializer='glorot_uniform')(first_hidden_encoded )
                        second_hidden_dense_batchnorm = BatchNormalization()(second_hidden_dense_linear)
                        second_hidden_encoded = Activation('relu')(second_hidden_dense_batchnorm)

                        z_mean_dense_linear = Dense(self.latent_dim, kernel_initializer='glorot_uniform')(second_hidden_encoded)
                        z_mean_dense_batchnorm = BatchNormalization()(z_mean_dense_linear)
                        self.z_mean_encoded = Activation('relu')(z_mean_dense_batchnorm)

                        z_log_var_dense_linear = Dense(self.latent_dim, kernel_initializer='glorot_uniform')(second_hidden_encoded)
                        z_log_var_dense_batchnorm = BatchNormalization()(z_log_var_dense_linear)
                        self.z_log_var_encoded = Activation('relu')(z_log_var_dense_batchnorm)

                        z = Lambda(sampling.sampling, output_shape=(self.latent_dim, ))([self.z_mean_encoded, self.z_log_var_encoded])
        def build_decoder_layer(self):

                self.merged_decode = keras.layers.concatenate([z,self.y_label_input], axis=-1) #merged_decodeはデータフレームじゃなさそうなのでエラーでるかも
                #変更点
                self.latent_variable_label_input = Input(shape=(self.latent_dim + 1, ))

                if self.layer_depth == 1:
                        self.decoder_to_reconstruct = Dense(self.original_dim,activation='sigmoid', input_dim=self.latent_dim+1)(self.latent_variable_label_input)
                        #self.rnaseq_reconstruct = self.decoder_to_reconstruct(self.merged_decode)
                        self.decoder_model = Model(self.latent_variable_label_input,self.decoder_to_reconstruct)

                        self.rnaseq_reconstruct = self.decoder_model(self.merged_decode)
                elif self.layer_depth == 2:
                        self.decoder_hidden1 = Dense(self.first_hidden_dim, activation='relu', input_dim=self.latent_dim+1)(self.latent_variable_label_input) #デコーダーにカーネルイニシ $
                        self.decoder_to_reconstruct = Dense(self.original_dim, activation='sigmoid')(self.decoder_hidden1)

                        #self.hidden1 = self.decoder_hidden1(self.merged_decode)
                        #self.rnaseq_reconstruct = self.decoder_activate(self.hidden1)
                        self.decoder_model = Model(self.latent_variable_label_input,self.decoder_to_reconstruct)
                        self.rnaseq_reconstruct = self.decoder_model(self.merged_decode)
                elif self.layer_depth == 3:
                        self.decoder_hidden1 = Dense(self.first_hidden_dim, activation='relu', input_dim=self.latent_dim+1)(self.latent_variable_label_input)
                        self.decoder_hidden2 = Dense(self.second_hidden_dim, activation='relu', input_dim=self.first_hidden_dim)(self.decoder_hidden1)
                        self.decoder_to_reconstruct = Dense(self.original_dim, activation='sigmoid')(self.decoder_hidden2)

                        #self.hidden1 = self.decoder_hidden1(self.merged_decode)
                        #self.hidden2 = self.decoder_hidden2(self.hidden1)
                        #self.rnaseq_reconstruct = self.decoder_activate(self.hidden2)
                        self.decoder_model = Model(self.latent_variable_label_input,self.decoder_to_reconstruct)
                        self.rnaseq_reconstruct = self.decoder_model(self.merged_decode)

        def compile_cvae(self):

                adam = optimizers.Adam(lr=self.learning_rate)
                cvae_layer = CustomVariationalLayer.CustomVariationalLayer(self.z_log_var_encoded,self.z_mean_encoded)([self.rnaseq_input, self.rnaseq_reconstruct])
                self.cvae = Model([self.rnaseq_input,self.y_label_input], cvae_layer)
                self.cvae.compile(optimizer=adam, loss=None, loss_weights=[self.beta])

                return self.cvae

        def get_summary(self):
                self.cvae.summary()

        def visualize_architecture(self, output_file):

                plot_model(self.cvae, to_file=output_file)
                #SVG(model_to_dot(self.cvae).create(prog='dot', format='svg'))

        def train_cvae(self):
                self.hist = self.cvae.fit([np.array(rnaseqdata_1_54_import.rnaseq_df_train),np.array(rnaseqdata_1_54_import.age_label_train)],
                                  shuffle=True,
                                  epochs=self.epochs,
                                  batch_size=self.batch_size,
                                  validation_data=([np.array(rnaseqdata_1_54_import.rnaseq_df_test),np.array(rnaseqdata_1_54_import.age_label_test)],None),
                                  callbacks=[WarmUpCallback.WarmUpCallback(self.beta, self.kappa)])
                return self.hist

        def get_cvae_loss(self):

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

        def compress(self, df):
                # エンコーディングするオブジェクト    #dfにはラベルと合わせたrna-seqデータを入力してください。
                self.encoder = Model([self.rnaseq_input,self.y_label_input], z)
                encoded_df = self.encoder.predict_on_batch(df)
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

        def save_models(self, encoder_file, decoder_file):
                self.encoder.save(encoder_file)
                self.decoder.save(decoder_file)
