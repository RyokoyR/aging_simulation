import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from keras.layers import Input, Dense, Lambda, Layer, Activation, merge
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import backend as K
from keras import metrics, optimizers
from keras.callbacks import Callback
import keras
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

import h5py

class inter_aging_simulate():
        #このクラスをインスタンス化して使うたびに変更したい引数をここにかく
        def __init__(self,sample_age,latent_variable_df_path,decoder_model_path):
                self.sample_age = sample_age
                self.latent_variable_df_path = latent_variable_df_path
                self.latent_variable_df = pd.read_csv(self.latent_variable_df_path,index_col=0)
                self.decoder_model_path = decoder_model_path

        def choice_sample_age(self):
                if sample_age == "21":
                        self.simulation_input_df = self.latent_variable_df.iloc[0:505]
                if sample_age == "38":
                        self.simulation_input_df = self.latent_variable_df.iloc[505:979]
                if sample_age == "01":
                        self.simulation_input_df = self.latent_variable_df.iloc[979:1200]
                if sample_age == "05":
                        self.simulation_input_df = self.latent_variable_df.iloc[1200:1531]
                if sample_age == "44":
                        self.simulation_input_df = self.latent_variable_df.iloc[1531:1808]
                if sample_age == "06":
                        self.simulation_input_df = self.latent_variable_df.iloc[1808:1986]
                if sample_age == "54":
                        self.simulation_input_df = self.latent_variable_df.iloc[1986:2258]
                if sample_age == "22":
                        self.simulation_input_df = self.latent_variable_df.iloc[2258:len(self.latent_variable_df)]

                return self.simulation_input_df

        #取ってきた任意の年齢のサンプル全てに対応する潜在変数からデコーダーを使って遺伝子発現をシミュレーションする関数
        def get_simulation_df(self):
                #学習済みのデコーダーモデルを読み込む
                decoder_model = keras.models.load_model(decoder_model_path)
                simulation_list = []
                #0才から100才まで変化する年齢ラベルをつくる
                #for をサンプル名で回すように編集する
                for age in range(0,100):
                        label_df = pd.DataFrame([age]*len(self.latent_variable_df))
                        label_df.index = self.latent_varible_df.index
                        label_df.columns = ["age"]
                        self.decoder_input_df = pd.concat([self.simulation_input_df,label_df],axis=1)
                        rnaseq_simulation = decoder_model.predict(np.array(self.decoder_input_df))
                        rnaseq_simulation_df = pd.DataFrame(rnaseq_simulation,index=self.simulation_input_df.index,columns=rnaseq_df.columns) #rnaseq_dfを読み込んでおく必要あり
