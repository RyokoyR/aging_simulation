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
from keras_tqdm import TQDMNotebookCallback
#from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from sklearn.cluster import KMeans
from sklearn import manifold
from sklearn import datasets
from matplotlib.cm import get_cmap
from sklearn.metrics import explained_variance_score
import plotly.offline as offline
import h5py

class WarmUpCallback(Callback):
    def __init__(self, beta, kappa):
        self.beta = beta
        self.kappa = kappa
    # Behavior on each epoch
    def on_epoch_end(self, epoch, logs={}):
        #self.beta = trial.suggest_uniform("beta",0,1)

        if K.get_value(self.beta) <= 1:
            K.set_value(self.beta, K.get_value(self.beta) + self.kappa)
