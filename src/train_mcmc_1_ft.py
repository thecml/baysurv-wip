import pandas as pd
import numpy as np
import tensorflow as tf
from data_loader import load_veterans_ds, prepare_veterans_ds
from sklearn.preprocessing import StandardScaler
from utility import InputFunction, CindexMetric, CoxPHLoss, _make_riskset, _TFColor
import matplotlib.pyplot as plt

TFColor = _TFColor()

import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

if __name__ == "__main__":

    # Load data
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_veterans_ds()
    t_train, t_valid, t_test, e_train, e_valid, e_test  = prepare_veterans_ds(y_train, y_valid, y_test)

    X_train = np.array(X_train)

    y_obs = tf.convert_to_tensor(t_train[e_train])
    y_cens = tf.convert_to_tensor(t_train[~e_train])

    x_obs = tf.convert_to_tensor(X_train[e_train][:,0])
    x_cens = tf.convert_to_tensor(X_train[~e_train][:,0])

    obs_model = tfd.JointDistributionSequential(
    [
        tfd.Normal(3, 3), #alpha
        tfd.Normal(0, 3), #beta
        lambda beta, alpha:
        tfd.Independent(tfd.Sample(
            tfd.Exponential(rate =
                1/tf.math.exp(tf.cast(x_obs[tf.newaxis,...], beta.dtype)*beta[...,tf.newaxis]+\
                            alpha[...,tf.newaxis])
            )), reinterpreted_batch_ndims = 1)
    ]
    )

    def log_prob(x_obs, x_cens, y_obs, y_cens, alpha, beta):
        lp = obs_model.log_prob([alpha, beta, tf.cast(y_obs, alpha.dtype)[tf.newaxis,...]])
        potential = exponential_lccdf(x_cens, y_cens, alpha, beta)
        return lp + potential

    def exponential_lccdf(x_cens, y_cens, alpha, beta):
        return tf.reduce_sum(
            -tf.cast(y_cens[tf.newaxis,...],alpha.dtype) / tf.exp(tf.cast(x_cens[tf.newaxis,...],
                                                                        beta.dtype) * beta[...,tf.newaxis] + alpha[...,tf.newaxis]),
            axis=-1
        )

    from utility import sample_hmc_1_ft

    n_chains = 1
    number_of_steps = 1000
    number_burnin_steps = 100

    initial_coeffs = obs_model.sample(n_chains)
    unnormalized_post_log_prob = lambda *args: log_prob(x_obs, x_cens, y_obs, y_cens, *args)
    alphas, betas = sample_hmc_1_ft(unnormalized_post_log_prob, [tf.zeros_like(initial_coeffs[0]),
                                                                 tf.zeros_like(initial_coeffs[1])],
                            n_steps=number_of_steps, n_burnin_steps=number_burnin_steps)
    
    print(alphas)
    print(betas)