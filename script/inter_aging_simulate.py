#!/lustre7/home/lustre4/ryoyokosaka/python/.pyenv/shims
import sys
sys.path.append('/lustre7/home/lustre4/ryoyokosaka/.pyenv/versions/3.6.0/lib/python3.6/site-packages')

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
import sampling,CustomVariationalLayer,WarmUpCallback,simulate_by_CVAE

#保存したモデルを読み込む
encoder_model = keras.models.load_model('/lustre7/home/lustre4/ryoyokosaka/python/CVAE_result/encoder_model.hdf5')
rnaseq_df = pd.read_table('/lustre7/home/lustre4/ryoyokosaka/python/CVAE_result/1_54rnaseq_drop.txt',index_col = 0)
latent_variable_df = pd.read_csv('/lustre7/home/lustre4/ryoyokosaka/python/CVAE_result/latent_variable_by_cvae_df.csv',index_col = 0)
latent_path = '/lustre7/home/lustre4/ryoyokosaka/python/CVAE_result/latent_variable_by_cvae_df.csv'
decoder_path = '/lustre7/home/lustre4/ryoyokosaka/python/CVAE_result/decoder_model.hdf5'

sampleage = ["21","38","01","05","44","06","54","22"]
save_fig_directory = '/lustre7/home/lustre4/ryoyokosaka/python/CVAE_result'
simulation_df_nested_path = '/lustre7/home/lustre4/ryoyokosaka/python/CVAE_result/simulation_df_nested.csv'
drow_gene_name_list = ["SLC41A3","CASP1","FCGR1A","MIR106A","MIR20A","ARL6IP6"]
for id in sampleage:
        simulater = inter_aging_simulate(id,latent_path,decoder_path)
        simulater.choice_sample_age()
        simulater.get_simulation_df(simulation_df_nested_path)
        simulater.visualize_gene_expression_simulation(drow_gene_name_list,save_fig_directory)
