#!/lustre7/home/lustre4/ryoyokosaka/python/.pyenv/shims
import sys
sys.path.append('lustre7/home/lustre4/ryoyokosaka/python/')

from keras import backend as K

epsilon_std = 1.0


def sampling(args):

    global z
    import tensorflow as tf
    # Function with args required for Keras Lambda function
    z_mean, z_log_var = args

    # Draw epsilon of the same shape from a standard normal distribution
    epsilon = K.random_normal(shape=tf.shape(z_mean), mean=0.,
                              stddev=epsilon_std)

    # The latent vector is non-deterministic and differentiable
    # in respect to z_mean and z_log_var
    z = z_mean + K.exp(z_log_var / 2) * epsilon
    return z
