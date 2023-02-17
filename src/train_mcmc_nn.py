from __future__ import print_function

import math
from datetime import datetime
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import sklearn.model_selection
import sklearn.preprocessing
import tensorflow as tf
import tensorflow_probability as tfp
from utility import sample_hmc
from sklearn.preprocessing import StandardScaler
from data_loader import load_veterans_ds, prepare_veterans_ds
from utility import _make_riskset
from utility import CoxPHLoss

import os
from pathlib import Path

tfd = tfp.distributions

DTYPE = tf.float32

loss_fn = CoxPHLoss()

def dense(X, W, b, activation):
    return activation(tf.matmul(X, W) + b)

def build_network(weights_list, biases_list, activation=tf.nn.relu):
    def model(X):
        net = X
        for (weights, biases) in zip(weights_list[:-1], biases_list[:-1]):
            net = dense(net, weights, biases, activation)
        # final linear layer
        net = tf.matmul(net, weights_list[-1]) + biases_list[-1]
        preds = net[:, 0]
        #std_devs = tf.exp(-net[:, 1])
        # preds and std_devs each have size N = X.shape(0) (the number of data samples)
        # and are the model's predictions and (log-sqrt of) learned loss attenuations, resp.
        return tfd.Exponential(rate=1/preds)
        #return tfd.Normal(loc=preds, scale=std_devs)

    return model

def get_initial_state(weight_prior, bias_prior, num_features, layers=None):
    """generate starting point for creating Markov chain
        of weights and biases for fully connected NN
    Keyword Arguments:
        layers {tuple} -- number of nodes in each layer of the network
    Returns:
        list -- architecture of FCNN with weigths and bias tensors for each layer
    """
    # make sure the last layer has two nodes, so that output can be split into
    # predictive mean and learned loss attenuation (see https://arxiv.org/abs/1703.04977)
    # which the network learns individually
    if layers is not None:
        assert layers[-1] == 1 #
    if layers is None:
        layers = (
            num_features,
            num_features // 2,
            num_features // 5,
            num_features // 10,
            1, # 1 node for exp rate
        )
    else:
        layers.insert(0, num_features)

    architecture = []
    for idx in range(len(layers) - 1):
        weigths = weight_prior.sample((layers[idx], layers[idx + 1]))
        biases = bias_prior.sample((layers[idx + 1]))
        # weigths = tf.zeros((layers[idx], layers[idx + 1]))
        # biases = tf.zeros((layers[idx + 1]))
        architecture.extend((weigths, biases))
    return architecture

def bnn_joint_log_prob_fn(weight_prior, bias_prior, x_cens, x_obs, y_cens, y_obs, *args):
    weights_list = args[::2]
    biases_list = args[1::2]

    # prior log-prob
    lp = sum([tf.reduce_sum(weight_prior.log_prob(weights)) for weights in weights_list])
    lp += sum([tf.reduce_sum(bias_prior.log_prob(bias)) for bias in biases_list])
    
    # likelihood of predicted observed labels
    network = build_network(weights_list, biases_list)
    exp_dist = network(x_obs)
    lp = tf.reduce_sum(exp_dist.log_prob(y_obs))
    
    # likelihood of predicted unobserved labels
    #net = x_cens
    #for (weights, biases) in zip(weights_list[:-1], biases_list[:-1]):
    #    net = dense(net, weights, biases, tf.nn.relu)
    #potential = tf.reduce_sum(-y_cens / tf.exp(tf.matmul(net, weights_list[-1]) + biases_list[-1]))
    
    return lp #+ potential
    
    #lp += tf.reduce_sum(labels_dist.log_prob(y_obs)) # log prob of Normal dist TODO: Add cens log prob
    #lp += tf.reduce_mean(train_loss)
    #return lp
    """
    
def get_map(target_log_prob_fn, state, num_iters=1000, save_every=100):
  state_vars = [tf.Variable(s) for s in state]
  opt = tf.optimizers.Adam()
  def map_loss():
      return -target_log_prob_fn(*state_vars)
  
  @tf.function
  def minimize():
      opt.minimize(map_loss, state_vars)
  
  traces = [[] for _ in range(len(state))]
  for i in range(num_iters):
    if i % save_every == 0:
      for t, s in zip(traces, state_vars):
        t.append(s.numpy())
    minimize()
  return [np.array(t) for t in traces]

"""
def plot_curves(chain):
    weights_list = chain[::2]
    biases_list = chain[1::2]

    train_trace = []
    test_trace = []
    for i in range(len(weights_list[0])):
        network = build_network([w[i] for w in weights_list], [b[i] for b in biases_list])(X_train.astype(np.float32))
        train_trace.append(-tf.reduce_mean(network.log_prob(y_train[:, 0])).numpy())
        network = build_network([w[i] for w in weights_list], [b[i] for b in biases_list])(X_test.astype(np.float32))
        test_trace.append(-tf.reduce_mean(network.log_prob(y_test[:, 0])).numpy())
    
    plt.plot(train_trace, label='train')
    plt.plot(test_trace, label='test')
    plt.legend(loc='best')

def load_veteran():
    X_train, _, _, y_train, y_valid, y_test = load_veterans_ds()
    t_train, _, _, e_train, _, _  = prepare_veterans_ds(y_train, y_valid, y_test)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    y_obs = tf.convert_to_tensor(t_train[e_train], dtype=DTYPE)
    y_cens = tf.convert_to_tensor(t_train[~e_train], dtype=DTYPE)
    x_obs = tf.convert_to_tensor(X_train[e_train], dtype=DTYPE)
    x_cens = tf.convert_to_tensor(X_train[~e_train], dtype=DTYPE)
    return x_cens, x_obs, y_cens, y_obs

if __name__ == "__main__":
    weight_prior = tfd.Normal(0.0, 0.1)
    bias_prior = tfd.Normal(0.0, 1.0)  # near-uniform
    
    x_cens, x_obs, y_cens, y_obs = load_veteran()
    
    bnn_joint_log_prob = partial(bnn_joint_log_prob_fn, weight_prior, bias_prior,
                                 x_cens, x_obs, y_cens, y_obs)
    num_features = x_cens.shape[1]
    initial_state = get_initial_state(weight_prior, bias_prior, num_features)

    z = 0
    for s in initial_state:
        print("State shape", s.shape)
    z += s.shape.num_elements()
    print("Total params", z)
    
    number_of_steps = 20000
    number_burnin_steps = 5000
    
    # Results is tuple with shape [chains, trace]
    results = sample_hmc(bnn_joint_log_prob, initial_state,
                         n_steps=number_of_steps, n_burnin_steps=number_burnin_steps)
    
    log_accept_ratio = results[1][1][number_burnin_steps:]
    target_accept_prob = tf.math.exp(tfp.math.reduce_logmeanexp(tf.minimum(log_accept_ratio, 0.))).numpy()
    print(f'Target acceptance probability: {round(100*target_accept_prob)}%')
    
    plt.figure(figsize=(10,6))
    accepted_samples = results[1][0][number_burnin_steps:]
    print(f'Acceptance rate: {round(100*np.mean(accepted_samples), 2)}%')
    n_accepted_samples = len(accepted_samples)
    n_bins = int(n_accepted_samples/100)
    sample_indicies = np.linspace(0, n_accepted_samples, n_bins)
    means = [np.mean(accepted_samples[:int(idx)]) for idx in sample_indicies[5:]]
    plt.plot(np.arange(len(means)), means)
    plt.show()

    """
    for chain in chain:
        print("ESS/step", tf.reduce_min(tfp.mcmc.effective_sample_size(chain[-1000:]) / 1000).numpy())

    plt.figure()
    plt.title("Chains")
    for i in range(14):
        plt.plot(chain[6][:, i, 0])

    plt.figure()
    #plt.title("Step size")
    #plt.plot(trace.inner_results.accepted_results.step_size)

    plot_curves([c[::50] for c in chain])
    plt.ylim(-1, 2)
    plt.yticks(np.linspace(-1, 2, 16))
    plt.show()
    """