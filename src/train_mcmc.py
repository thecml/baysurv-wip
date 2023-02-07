import pandas as pd
import numpy as np
import tensorflow as tf
from data_loader import load_veterans_ds, prepare_veterans_ds
from sklearn.preprocessing import StandardScaler
from utility import InputFunction, CindexMetric, CoxPHLoss, _make_riskset
import matplotlib.pyplot as plt
import os
from pathlib import Path
import seaborn as sns

import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

obs_model = tfd.JointDistributionSequential(
  [
    tfd.Normal(3, 3), #alpha
    tfd.Normal(0, 3), #beta
    lambda beta, alpha:
      tfd.Independent(tfd.Sample(
        tfd.Exponential(rate =
            1/tf.math.exp(tf.cast(is_black_obs[tf.newaxis,...], beta.dtype)*beta[...,tf.newaxis]+\
                        alpha[...,tf.newaxis])
        )), reinterpreted_batch_ndims = 1)
  ]
)

def log_prob(alpha, beta):
    lp = obs_model.log_prob([alpha, beta, tf.cast(y_obs, alpha.dtype)[tf.newaxis,...]])
    potential = exponential_lccdf(alpha, beta)
    return lp + potential

def exponential_lccdf(alpha, beta):
    return tf.reduce_sum(
        -tf.cast(y_cens[tf.newaxis,...],alpha.dtype) / tf.exp(tf.cast(is_black_cens[tf.newaxis,...], beta.dtype) * beta[...,tf.newaxis] + alpha[...,tf.newaxis]),
        axis=-1
    )

def sampleHMC(log_prob, inits, bijectors_list = None):
    inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=log_prob,
        step_size=0.1,
        num_leapfrog_steps=8
    )
    if bijectors_list is not None:
        inner_kernel = tfp.mcmc.TransformedTransitionKernel(inner_kernel, bijectors_list)

    adaptive_kernel = tfp.mcmc.SimpleStepSizeAdaptation(
        inner_kernel=inner_kernel,
        num_adaptation_steps=800
    )
    return tfp.mcmc.sample_chain(
        num_results=1000,
        current_state=inits,
        kernel=adaptive_kernel,
        num_burnin_steps=1000,
        trace_fn=None
    )

if __name__ == "__main__":
    curr_dir = os.getcwd()
    root_dir = Path(curr_dir).absolute().parent.parent
    cats_file = Path.joinpath(root_dir, "downloads/AustinCats.csv")
    df = pd.read_csv(cats_file)

    df['black'] = df.color.apply(lambda x: x=='Black')
    df['adopted'] = df.out_event.apply(lambda x: x=='Adoption')

    is_black_cens = tf.convert_to_tensor(df.query('adopted==0').black.values.astype(float))
    is_black_obs = tf.convert_to_tensor(df.query('adopted==1').black.values.astype(float))

    X_train, X_valid, X_test, y_train, y_valid, y_test = load_veterans_ds()
    t_train, t_valid, t_test, e_train, e_valid, e_test  = prepare_veterans_ds(y_train, y_valid, y_test)
    X_train = np.array(X_train)

    y_obs = tf.convert_to_tensor(t_train[e_train])
    y_cens = tf.convert_to_tensor(t_train[~e_train])

    x_obs = tf.convert_to_tensor(X_train[e_train][:,0]) # 1 feature
    x_cens = tf.convert_to_tensor(X_train[~e_train][:,0]) # 1 feature

    n_chains = 4
    initial_coeffs = obs_model.sample(n_chains)

    [alphas, betas], kernel_results = sampleHMC(log_prob, [tf.zeros_like(initial_coeffs[0]), tf.zeros_like(initial_coeffs[1])])

    alphas = alphas.numpy().flatten()
    betas = betas.numpy().flatten()

    lambda_black = np.exp(alphas + betas)
    lambda_non_black = np.exp(alphas)

    sns.distplot(lambda_black,color='k',label='black')
    sns.distplot(lambda_non_black,color='orange',label='non black')
    plt.legend(fontsize=15)
    plt.xlabel("Days to adoption",size=15)
    plt.ylabel("Density",size=15)
    plt.show()