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
import pickle

TFColor = _TFColor()

tfd = tfp.distributions
tfb = tfp.bijectors

DTYPE = tf.float32
N_CHAINS = 1

def load_cats_ds():
    curr_dir = os.getcwd()
    root_dir = Path(curr_dir).absolute().parent.parent
    df = pd.read_csv(Path().joinpath(root_dir, "downloads/AustinCats.csv"), delimiter=";")
    df['black'] = df.color.apply(lambda x: x=='Black')
    df['adopted'] = df.out_event.apply(lambda x: x=='Adoption')
    x_cens = tf.convert_to_tensor(df.query('adopted==0').black.values, dtype=DTYPE)
    x_obs = tf.convert_to_tensor(df.query('adopted==1').black.values, dtype=DTYPE)
    y_cens = tf.convert_to_tensor(df.query('adopted==0').days_to_event, dtype=DTYPE)
    y_obs = tf.convert_to_tensor(df.query('adopted==1').days_to_event, dtype=DTYPE)
    return x_cens, x_obs, y_cens, y_obs

def load_veteran_ds():
    X_train, _, _, y_train, y_valid, y_test = load_veterans_ds()
    t_train, _, _, e_train, _, _  = prepare_veterans_ds(y_train, y_valid, y_test)
    X_train = np.array(X_train)
    y_obs = tf.convert_to_tensor(t_train[e_train], dtype=DTYPE)
    y_cens = tf.convert_to_tensor(t_train[~e_train], dtype=DTYPE)
    x_obs = tf.convert_to_tensor(X_train[e_train], dtype=DTYPE)
    x_cens = tf.convert_to_tensor(X_train[~e_train], dtype=DTYPE)
    return x_cens, x_obs, y_cens, y_obs

if __name__ == "__main__":
    x_cens, x_obs, y_cens, y_obs = load_cats_ds()
    n_dims = 1

    obs_model = tfd.JointDistributionSequentialAutoBatched([
            tfd.Normal(loc=tf.zeros([1]), scale=tf.ones([1])), # alpha
            tfd.Normal(loc=tf.zeros([n_dims,1]), scale=tf.ones([n_dims,1])), # beta
            lambda beta, alpha: tfd.Exponential(rate=1/tf.math.exp(tf.transpose(x_obs)*beta + alpha))]
        )

    def log_prob(x_obs, x_cens, y_obs, y_cens, alpha, beta):
        lp = obs_model.log_prob([alpha, beta, y_obs])
        potential = exponential_lccdf(x_cens, y_cens, alpha, beta)
        return lp + potential

    def exponential_lccdf(x_cens, y_cens, alpha, beta):
        return tf.reduce_sum(-y_cens / tf.exp(tf.transpose(x_cens)*beta + alpha))

    number_of_steps = 10000
    number_burnin_steps = 1000

    # Sample from the prior
    initial_coeffs = obs_model.sample(1)

    # Run sampling for number of chains
    unnormalized_post_log_prob = lambda *args: log_prob(x_obs, x_cens, y_obs, y_cens, *args)
    chains = [sample_hmc(unnormalized_post_log_prob, [tf.zeros_like(initial_coeffs[0]),
                                                      tf.zeros_like(initial_coeffs[1])],
                         n_steps=number_of_steps, n_burnin_steps=number_burnin_steps) for i in range(N_CHAINS)]
    
    # Calculate accepted mean
    accepted_samples = chains[0][1][number_burnin_steps:]
    print('Acceptance rate: %0.1f%%' % (100*np.mean(accepted_samples)))
    
    # Save chains
    curr_dir = os.getcwd()
    root_dir = Path(curr_dir).absolute()
    with open(f'{root_dir}/models/mcmc_chains.pkl', 'wb') as fp:
        pickle.dump(chains, fp)