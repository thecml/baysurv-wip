import pandas as pd
import numpy as np
import tensorflow as tf
from data_loader import load_veterans_ds, prepare_veterans_ds
from sklearn.preprocessing import StandardScaler
from utility import InputFunction, CindexMetric, CoxPHLoss, _make_riskset, _TFColor
import matplotlib.pyplot as plt
from utility import sample_hmc
import os
from pathlib import Path
import seaborn as sns

TFColor = _TFColor()

dtype = tf.float32

import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

if __name__ == "__main__":
    # Load data
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_veterans_ds()
    t_train, t_valid, t_test, e_train, e_valid, e_test  = prepare_veterans_ds(y_train, y_valid, y_test)

    # Convert to arrays
    X_train = np.array(X_train)
    X_valid = np.array(X_valid)
    X_test =  np.array(X_test)

    curr_dir = os.getcwd()
    root_dir = Path(curr_dir).absolute().parent.parent
    df = pd.read_csv(Path().joinpath(root_dir, "downloads/AustinCats.csv"))

    df['black'] = df.color.apply(lambda x: x=='Black')
    df['adopted'] = df.out_event.apply(lambda x: x=='Adoption')

    x_cens = tf.convert_to_tensor(df.query('adopted==0').black.values, dtype=dtype)
    x_obs = tf.convert_to_tensor(df.query('adopted==1').black.values, dtype=dtype)

    y_cens = tf.convert_to_tensor(df.query('adopted==0').days_to_event, dtype=dtype)
    y_obs = tf.convert_to_tensor(df.query('adopted==1').days_to_event, dtype=dtype)

    """
    y_obs = tf.convert_to_tensor(t_train[e_train], dtype=dtype)
    y_cens = tf.convert_to_tensor(t_train[~e_train], dtype=dtype)

    x_obs = tf.convert_to_tensor(X_train[e_train][:,1], dtype=dtype)
    x_cens = tf.convert_to_tensor(X_train[~e_train][:,1], dtype=dtype)
    """

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

    n_chains = 1
    number_of_steps = 1000
    number_burnin_steps = 1000

    initial_coeffs = obs_model.sample(n_chains)
    unnormalized_post_log_prob = lambda *args: log_prob(x_obs, x_cens, y_obs, y_cens, *args)
    alphas, betas = sample_hmc(unnormalized_post_log_prob, [tf.zeros_like(initial_coeffs[0]),
                                                            tf.zeros_like(initial_coeffs[1])],
                            n_steps=number_of_steps, n_burnin_steps=number_burnin_steps)

    alphas = alphas.numpy().flatten()
    betas = betas.numpy().flatten()

    print(alphas.mean())
    print(betas.mean())

    lambda_feature = np.exp(alphas + betas)
    lambda_no_feature = np.exp(alphas)

    sns.histplot(lambda_feature,color='k',label='beta0')
    sns.histplot(lambda_no_feature,color='orange',label='no beta0')
    plt.legend(fontsize=15)
    plt.xlabel("Time",size=15)
    plt.ylabel("Density",size=15)

    # plotting the Posterior Samples
    plt.figure(figsize=(5,3))
    plt.plot(np.arange(number_of_steps), alphas, color=TFColor[0])
    plt.title('HMC α convergence progression', fontsize=14)

    plt.figure(figsize=(5,3))
    plt.plot(np.arange(number_of_steps), betas, color=TFColor[1])
    plt.title(f'HMC β convergence progression', fontsize=14)
    plt.show()