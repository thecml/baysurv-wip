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
from sklearn.preprocessing import StandardScaler
from tools import data_loader
from utility.loss import CoxPHLoss
from utility.mcmc import sample_hmc
from utility.risk import _make_riskset
from utility.risk import InputFunction
from sksurv.metrics import concordance_index_censored
from collections import defaultdict

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
        std_devs = tf.exp(-net[:, 1])
        # preds and std_devs each have size N = X.shape(0) (the number of data samples)
        # and are the model's predictions and (log-sqrt of) learned loss attenuations, resp.
        return tfd.Normal(loc=preds, scale=std_devs)

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
        assert layers[-1] == 2
    if layers is None:
        layers = (
            num_features,
            num_features // 2,
            num_features // 5,
            num_features // 10,
            2,
        )
    else:
        layers.insert(0, num_features)

    architecture = []
    for idx in range(len(layers) - 1):
        weigths = weight_prior.sample((layers[idx], layers[idx + 1]))
        biases = bias_prior.sample((layers[idx + 1]))
        architecture.extend((weigths, biases))
    return architecture

def bnn_joint_log_prob_fn(weight_prior, bias_prior, X_train, e_train, t_train, *args):
    weights_list = args[::2]
    biases_list = args[1::2]

    # prior log-prob
    lp = sum([tf.reduce_sum(weight_prior.log_prob(weights)) for weights in weights_list])
    lp += sum([tf.reduce_sum(bias_prior.log_prob(bias)) for bias in biases_list])

    # get logits
    network = build_network(weights_list, biases_list)
    logits = network(X_train).sample()

    # compute loss
    train_event_set = tf.expand_dims(e_train.astype(np.int32), axis=1)
    train_risk_set = tf.convert_to_tensor(_make_riskset(t_train), dtype=np.bool_)
    logits = tf.reshape(logits, (logits.shape[0], 1))

    # reduce loss
    loss = loss_fn(y_true=[train_event_set, train_risk_set], y_pred=logits)
    lp += -tf.reduce_sum(loss)
    return lp

def plot_curves(chains):
    # Plot loss curves
    for chain_id, chain in enumerate(chains):
        weights_list = chain[::2]
        biases_list = chain[1::2]

        train_trace, valid_trace = [], []
        for i in range(number_burnin_steps, len(weights_list[0]), 100):
            network = build_network([w[i] for w in weights_list], [b[i] for b in biases_list])
            logits = tf.reshape(network(X_valid).sample(), (X_valid.shape[0], 1))
            event_set = tf.expand_dims(e_valid.astype(np.int32), axis=1)
            risk_set = tf.convert_to_tensor(_make_riskset(t_valid), dtype=np.bool_)
            valid_loss = loss_fn(y_true=[event_set, risk_set], y_pred=logits)
            valid_trace.append(valid_loss.numpy())

        plt.plot(valid_trace, label=f'C{chain_id} valid loss')

    plt.legend(loc='best')
    plt.show()

if __name__ == "__main__":
    weight_prior = tfd.Normal(0.0, 0.1)
    bias_prior = tfd.Normal(0.0, 1.0)  # near-uniform

    # Load data
    dl = data_loader.CancerDataLoader().load_data()
    X_train, X_valid, X_test, y_train, y_valid, y_test = dl.prepare_data()
    t_train, t_valid, t_test, e_train, e_valid, e_test = dl.make_time_event_split(y_train, y_valid, y_test)

    bnn_joint_log_prob = partial(bnn_joint_log_prob_fn, weight_prior, bias_prior, X_train, e_train, t_train)
    num_features = X_train.shape[1]
    n_layers = 4
    layers = [20, 50, 20, 2]
    initial_state = get_initial_state(weight_prior, bias_prior, num_features, layers)

    z = 0
    for s in initial_state:
        print("State shape", s.shape)
    z += s.shape.num_elements()
    print("Total params", z)

    # Results is tuple with shape [chains, trace]
    n_chains = 3
    number_of_steps = 10000
    number_burnin_steps = 1000
    chains = [sample_hmc(bnn_joint_log_prob, initial_state,
                         n_steps=number_of_steps, n_burnin_steps=number_burnin_steps)
              for _ in range(n_chains)]

    # Calculate target accept prob
    for chain_id in range(n_chains):
        log_accept_ratio = chains[chain_id][1][1][number_burnin_steps:]
        target_accept_prob = tf.math.exp(tfp.math.reduce_logmeanexp(tf.minimum(log_accept_ratio, 0.))).numpy()
        print(f'Target acceptance probability for {chain_id}: {round(100*target_accept_prob)}%')

    # Calculate accepted rate
    plt.figure(figsize=(10,6))
    for chain_id in range(n_chains):
        accepted_samples = chains[chain_id][1][0][number_burnin_steps:]
        print(f'Acceptance rate chain for {chain_id}: {round(100*np.mean(accepted_samples), 2)}%')
        n_accepted_samples = len(accepted_samples)
        n_bins = int(n_accepted_samples/100)
        sample_indicies = np.linspace(0, n_accepted_samples, n_bins)
        means = [np.mean(accepted_samples[:int(idx)]) for idx in sample_indicies[5:]]
        plt.plot(np.arange(len(means)), means)
    plt.show()

    chains = [chain[0] for chain in chains] # leave out accept traces
    plot_curves(chains) # plot loss curves

    # Take sample mean of combined chains
    chain_weights, chain_biases = defaultdict(list), defaultdict(list)
    for chain in chains:
        for layer_i, chain_idx in enumerate(range(0, len(chain)-1, 2)):
            weight_samples = chain[chain_idx][number_burnin_steps:,:,:]
            bias_samples = chain[chain_idx+1][number_burnin_steps:,:]
            chain_weights[f'layer_{layer_i}'].append(tf.math.reduce_mean(weight_samples, axis=0))
            chain_biases[f'layer_{layer_i}'].append(tf.math.reduce_mean(bias_samples, axis=0))

    weights = [tf.reduce_mean(chain_weights[f'layer_{layer}'], axis=0) for layer in range(n_layers)]
    biases = [tf.reduce_mean(chain_biases[f'layer_{layer}'], axis=0) for layer in range(n_layers)]
    network = build_network(weights, biases)

    # Compute test loss
    logits = network(X_test).sample() # output is a CPD
    event_set = tf.expand_dims(e_test.astype(np.int32), axis=1)
    risk_set = tf.convert_to_tensor(_make_riskset(t_test), dtype=np.bool_)
    logits = tf.reshape(logits, (logits.shape[0], 1))
    loss = loss_fn(y_true=[event_set, risk_set], y_pred=logits).numpy()

    # Compute C-index
    predictions = tf.reshape(logits, -1)
    ci = concordance_index_censored(e_test, t_test, predictions)[0]
    print("Training completed, test loss/C-index: %.4f/%.4f" % (round(loss, 4), round(ci, 4)))