import pandas as pd
import numpy as np
import tensorflow as tf
from data_loader import load_veterans_ds, prepare_veterans_ds
from sklearn.preprocessing import StandardScaler
from utility import InputFunction, CindexMetric, CoxPHLoss, _make_riskset
import matplotlib.pyplot as plt

import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

def log_prob(is_train_obs, obs_times, is_train_cens, cens_times, alpha, beta):
    exp_rate = 1/tf.math.exp(tf.cast(is_train_obs, beta.dtype) * beta + alpha)
    
    obs_model = tfd.JointDistributionSequential([tfd.Independent(tfd.Sample(tfd.Exponential(rate=exp_rate)), reinterpreted_batch_ndims=1)])
    exp_lccdf = exponential_lccdf(alpha, beta, is_train_cens, cens_times)
    
    return obs_model.log_prob([alpha, beta, tf.cast(obs_times, alpha.dtype)]) + tf.reduce_mean(exp_lccdf) # TODO: figure out if this is OK

def exponential_lccdf(alpha, beta, is_train_cens, cens_times):
    return tf.reduce_sum(
        -tf.cast(cens_times, alpha.dtype) / tf.exp(tf.cast(tf.transpose(is_train_cens), beta.dtype) * beta + alpha),
        axis=-1
    )

def sampleHMC(log_prob, inits, bijectors_list = None):
    inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=log_prob,
        step_size=step_size,
        num_leapfrog_steps=8
    )
    if bijectors_list is not None:
        inner_kernel = tfp.mcmc.TransformedTransitionKernel(inner_kernel, bijectors_list)
        
    adaptive_kernel = tfp.mcmc.SimpleStepSizeAdaptation(
        inner_kernel=inner_kernel,
        num_adaptation_steps=800
    )
    return tfp.mcmc.sample_chain(
        num_results=number_of_steps,
        current_state=inits,
        kernel=adaptive_kernel,
        num_burnin_steps=burnin,
        trace_fn=None
    )

if __name__ == "__main__":
    # Load data
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_veterans_ds()
    t_train, t_valid, t_test, e_train, e_valid, e_test  = prepare_veterans_ds(y_train, y_valid, y_test)
    
    # Scale data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)

    # Create data holders
    is_train_obs = tf.convert_to_tensor(X_train[e_train])
    is_train_cens = tf.convert_to_tensor(X_train[~e_train])
    obs_times = tf.convert_to_tensor(t_train[e_train])
    cens_times = tf.convert_to_tensor(t_train[~e_train])
    
    number_of_steps = 1000
    burnin = 1000
    step_size = 0.1
    
    initial_chain_state = [
        tf.cast(1., dtype=tf.float32) * tf.ones([], name='init_alpha', dtype=tf.float32),
        tf.cast(0.01, dtype=tf.float32) * tf.ones([], name='init_beta', dtype=tf.float32),
    ]
    
    posterior_log_prob = lambda *args: log_prob(is_train_obs, obs_times, is_train_cens, cens_times, *args)

    alphas, betas = sampleHMC(posterior_log_prob, initial_chain_state)