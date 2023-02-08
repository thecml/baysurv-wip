import pandas as pd
import numpy as np
import tensorflow as tf
from data_loader import load_veterans_ds, prepare_veterans_ds
from sklearn.preprocessing import StandardScaler
from utility import InputFunction, CindexMetric, CoxPHLoss, _make_riskset, sample_hmc
import matplotlib.pyplot as plt
import os
from pathlib import Path

import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

number_of_steps = 1000
burnin = 1000

dtype = tf.float32

alpha, sigma = 1, 1
beta = [1, 2.5]

if __name__ == "__main__":
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_veterans_ds()
    t_train, t_valid, t_test, e_train, e_valid, e_test = prepare_veterans_ds(y_train, y_valid, y_test)

    X_train = np.array(X_train)

    y_obs = tf.convert_to_tensor(t_train[e_train], dtype=dtype)
    y_cens = tf.convert_to_tensor(t_train[~e_train], dtype=dtype)

    x_obs = tf.convert_to_tensor(X_train[e_train][:,:2], dtype=dtype)
    x_cens = tf.convert_to_tensor(X_train[~e_train][:,:2], dtype=dtype)
    
    def obs_model_fn(x_obs):
        return tfd.JointDistributionNamedAutoBatched(dict(
            alpha=tfd.Normal(loc=[tf.cast(0.0, dtype)], scale=[tf.cast(10.0, dtype)]),
            beta=tfd.Normal(loc=[[tf.cast(0.0, dtype)], [tf.cast(0.0, dtype)]], 
                            scale=[[tf.cast(10.0, dtype)], [tf.cast(10.0, dtype)]]
            ),
            y=lambda beta, alpha: tfd.Exponential(
                rate=1/tf.math.exp(tf.linalg.matmul(x_obs, beta) + alpha)
            )
        ))
    
    def exponential_lccdf(x_cens, y_cens, beta, alpha):
        return tf.reduce_sum(-y_cens / tf.exp(tf.linalg.matmul(x_cens, beta) + alpha), axis=[0,1])
    
    def target_log_prob_fn(x_obs, x_cens, y_obs, y_cens, beta, alpha):
        lp = obs_model_fn(x_obs).log_prob(beta=beta, alpha=alpha, y=y_obs)
        potential = exponential_lccdf(x_cens, y_cens, beta, alpha)
        return lp + potential
    
    n_chains = 4
    number_of_steps = 100
    number_burnin_steps = 10

    initial_coeffs = obs_model_fn(x_obs).sample(n_chains)
    unnormalized_post_log_prob = lambda *args: target_log_prob_fn(x_obs, x_cens, y_obs, y_cens, *args)
    initial_alpha = tf.convert_to_tensor([tf.reduce_mean(initial_coeffs['alpha'])], dtype=dtype)
    initial_beta = tf.convert_to_tensor(tf.reduce_mean(initial_coeffs['beta'], axis=0), dtype=dtype)
    initial_states = [initial_beta, initial_alpha]
    
    alphas, betas = sample_hmc(unnormalized_post_log_prob, initial_states,
                               n_steps=number_of_steps, n_burnin_steps=number_burnin_steps)

    print(alphas)
    print(betas)