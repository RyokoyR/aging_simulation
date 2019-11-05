#!/lustre7/home/lustre4/ryoyokosaka/python/.pyenv/shims
import sys
sys.path.append('/lustre7/home/lustre4/ryoyokosaka/python/')

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

rnaseq_file = os.path.join('/home/ryoyokosaka/python/1_54rnaseq_drop.txt')
rnaseq_df = pd.read_table(rnaseq_file, index_col=0)

class inter_aging_simulate():
        #このクラスをインスタンス化して使うたびに変更したい引数をここにかく
        def __init__(self,sample_age,latent_variable_df_path,decoder_model_path):
                self.sample_age = sample_age
                self.latent_variable_df_path = latent_variable_df_path
                self.latent_variable_df = pd.read_csv(self.latent_variable_df_path,index_col=0)
                self.decoder_model_path = decoder_model_path

        def choice_sample_age(self):
                if self.sample_age == "21":
                        self.simulation_input_df = self.latent_variable_df.iloc[0:505]
                if self.sample_age == "38":
                        self.simulation_input_df = self.latent_variable_df.iloc[505:979]
                if self.sample_age == "01":
                        self.simulation_input_df = self.latent_variable_df.iloc[979:1200]
                if self.sample_age == "05":
                        self.simulation_input_df = self.latent_variable_df.iloc[1200:1531]
                if self.sample_age == "44":
                        self.simulation_input_df = self.latent_variable_df.iloc[1531:1808]
                if self.sample_age == "06":
                        self.simulation_input_df = self.latent_variable_df.iloc[1808:1986]
                if self.sample_age == "54":
                        self.simulation_input_df = self.latent_variable_df.iloc[1986:2258]
                if self.sample_age == "22":
                        self.simulation_input_df = self.latent_variable_df.iloc[2258:len(self.latent_variable_df)]

                return self.simulation_input_df

        #取ってきた任意の年齢のサンプル全てに対応する潜在変数からデコーダーを使って遺伝子発現をシミュレーションする関数
        def get_simulation_df(self,save_simulation_path):
                #学習済みのデコーダーモデルを読み込む
                decoder_model = keras.models.load_model(self.decoder_model_path)
                simulation_list = []
                #0才から100才まで変化する年齢ラベルをつくる
                for sample_name in self.simulation_input_df.index:
                        df_template = pd.DataFrame()
                        for age in range(0,101):
                                sample_latent = self.simulation_input_df.loc[sample_name]
                                #sample_latent_ = pd.DataFrame(sample_latent,columns = sample_name)
                                #sample_latent_df = sample_latent_.T
                                #label_df = pd.DataFrame([age])
                                #label_df.index = [sample_name]
                                #label_df.columns = ["age"]
                                label = pd.Series([float(age)])
                                self.decoder_input = pd.concat([sample_latent,label])
                                rnaseq_simulation = decoder_model.predict(np.array(self.decoder_input))
                                rnaseq_simulation_df = pd.DataFrame(rnaseq_simulation,index=[str(age)],columns=rnaseq_df.columns) #rnaseq_dfを読み込んでおく必要あり
                                df_template = pd.concat([df_template,rnaseq_simulation_df],axis = 0)
                        simulation_list.append(df_template)
                self.simulation_df_nested = pd.DataFrame(simulation_list,columns = self.simulation_input_df.index)
                self.simulation_df_nested.to_csv(save_simulation_path,index=True,header=True)

        #発現変化をシミュレートしたい遺伝子の名前をdrow_gene_name_listにリスト型で与えて使う
        def visualize_gene_expression_simulation(self,drow_gene_name_list,save_fig_directory):
                #save_fig_directoryには画像を保存するディレクトリパスを指定する。ファイル名は勝手に決まる。
                for gene_name in drow_gene_name_list:
                        plt.style.use('default')
                        sns.set()
                        fig = plt.figure(figsize=(10, 10))
                        plt.title(gene_name + 'inter_aging_expression')
                        ax = fig.add_subplot(1,1,1)
                        ax.set_xlabel("age(0~100)")
                        ax.set_ylabel("Expression_level")
                        for sample_name in self.simulation_df_nested.columns:
                                x = self.simulation_df_nested.loc[sample_name].index
                                y = self.simulation_df_nested.iloc[:,[gene_name]]
                                ax.plot(x,y,c = "dodgerblue",alpha = 0.6)
                        filename = str(self.sample_age)+"yo_samle"+gene_name+"simulation.pdf"
                        path = os.path.join(save_fig_directory,filename)
                        fig.savefig(path)
