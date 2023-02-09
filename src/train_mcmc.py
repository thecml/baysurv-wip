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
from utility import _TFColor
import seaborn as sns

TFColor = _TFColor()

tfd = tfp.distributions
tfb = tfp.bijectors

DTYPE = tf.float32
N_DIMS = 2

def load_cats_ds():
    curr_dir = os.getcwd()
    root_dir = Path(curr_dir).absolute().parent.parent
    df = pd.read_csv(Path().joinpath(root_dir, "downloads/AustinCats.csv"))
    df['black'] = df.color.apply(lambda x: x=='Black')
    df['adopted'] = df.out_event.apply(lambda x: x=='Adoption')
    x_cens = tf.convert_to_tensor(df.query('adopted==0').black.values, dtype=DTYPE)
    x_obs = tf.convert_to_tensor(df.query('adopted==1').black.values, dtype=DTYPE)
    y_cens = tf.convert_to_tensor(df.query('adopted==0').days_to_event, dtype=DTYPE)
    y_obs = tf.convert_to_tensor(df.query('adopted==1').days_to_event, dtype=DTYPE)
    return x_cens, x_obs, y_cens, y_obs

def load_veteran_ds():
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_veterans_ds()
    t_train, t_valid, t_test, e_train, e_valid, e_test  = prepare_veterans_ds(y_train, y_valid, y_test)
    X_train = np.array(X_train)
    X_valid = np.array(X_valid)
    X_test =  np.array(X_test)
    y_obs = tf.convert_to_tensor(t_train[e_train], dtype=DTYPE)
    y_cens = tf.convert_to_tensor(t_train[~e_train], dtype=DTYPE)
    x_obs = tf.convert_to_tensor(X_train[e_train][:,:N_DIMS], dtype=DTYPE)
    x_cens = tf.convert_to_tensor(X_train[~e_train][:,:N_DIMS], dtype=DTYPE)
    return x_cens, x_obs, y_cens, y_obs

if __name__ == "__main__":
    x_cens, x_obs, y_cens, y_obs = load_veteran_ds()

    # Sources:
    # https://adamhaber.github.io/post/survival-analysis/
    # https://juanitorduz.github.io/tfp_lm/
    # https://www.tensorflow.org/probability/examples/JointDistributionAutoBatched_A_Gentle_Tutorial
    obs_model = tfd.JointDistributionSequentialAutoBatched([
            tfd.Normal(loc=0, scale=1), # alpha
            tfd.Normal(loc=[[tf.cast(0.0, DTYPE)], [tf.cast(0.0, DTYPE)]],
                       scale=[[tf.cast(1.0, DTYPE)], [tf.cast(1.0, DTYPE)]]), # beta
            lambda beta, alpha: tfd.Exponential(rate=1/tf.math.exp(tf.transpose(x_obs)*beta + alpha))]
        )

    def log_prob(x_obs, x_cens, y_obs, y_cens, alpha, beta):
        lp = obs_model.log_prob([alpha, beta, y_obs])
        potential = exponential_lccdf(x_cens, y_cens, alpha, beta)
        return lp + potential

    def exponential_lccdf(x_cens, y_cens, alpha, beta):
        return tf.reduce_sum(-y_cens / tf.exp(tf.transpose(x_cens)*beta + alpha))

    number_of_steps = 1000
    number_burnin_steps = 1000

    # Sample from the prior
    initial_coeffs = obs_model.sample(1)

    # Run sampling.
    num_chains = 4
    unnormalized_post_log_prob = lambda *args: log_prob(x_obs, x_cens, y_obs, y_cens, *args)
    chains = [sample_hmc(unnormalized_post_log_prob, [tf.zeros_like(initial_coeffs[0]),
                                                      tf.zeros_like(initial_coeffs[1])],
                         n_steps=number_of_steps, n_burnin_steps=number_burnin_steps) for i in range(num_chains)]
    samples = chains[0] # samples from chain 0

    # Save alpha and betas
    alphas = samples[0].numpy().flatten()
    betas = np.zeros((N_DIMS, number_of_steps))

    for i in range(N_DIMS):
        betas[i] = samples[1][:,:,i].numpy().flatten()

    print(alphas.mean())
    print(betas[0].mean())
    print(betas[1].mean())

    lambda_all = np.exp(alphas + betas[0] + betas[1])
    lambda_b0 = np.exp(alphas + betas[0])
    lambda_no_beta = np.exp(alphas)

    sns.histplot(lambda_all,color='r',label='all')
    sns.histplot(lambda_b0,color='k',label='beta0')
    sns.histplot(lambda_no_beta,color='orange',label='no beta0')
    plt.legend(fontsize=15)
    plt.xlabel("Time",size=15)
    plt.ylabel("Density",size=15)

    # plotting the Posterior Samples
    plt.figure(figsize=(5,3))
    plt.plot(np.arange(number_of_steps), alphas, color=TFColor[0])
    plt.title('HMC α convergence progression', fontsize=14)

    for i in range(betas.shape[0]):
        plt.figure(figsize=(5,3))
        plt.plot(np.arange(number_of_steps), betas[i], color=TFColor[1])
        plt.title(f'HMC β_{i} convergence progression', fontsize=14)

    plt.show()