from __future__ import print_function

import math
from datetime import datetime
from functools import partial
from sklearn.model_selection import train_test_split
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
from utility.training import get_data_loader, scale_data, make_time_event_split

import os
from pathlib import Path

tfd = tfp.distributions

DTYPE = tf.float32
BATCH_SIZE = 32

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

    # get random batch
    #idx = np.random.randint(10, size=32)
    #X_sample = X_train[idx,:]
    #e_train_sample = e_train[idx]
    #t_train_sample = t_train[idx]

    # get logits
    network = build_network(weights_list, biases_list)
    logits = network(X_train).sample()

    # compute loss
    train_event_set = tf.expand_dims(e_train.astype(np.int32), axis=1)
    train_risk_set = tf.convert_to_tensor(_make_riskset(t_train), dtype=np.bool_)
    logits = tf.reshape(logits, (logits.shape[0], 1))

    # reduce loss
    loss = loss_fn(y_true=[train_event_set, train_risk_set], y_pred=logits)
    lp + -tf.reduce_sum(loss)
    return lp

def plot_curves(chains):
    # Plot loss curves
    for chain_id, chain in enumerate(chains):
        weights_list = chain[::2]
        biases_list = chain[1::2]

        train_trace, test_trace = [], []
        for i in range(number_burnin_steps, len(weights_list[0]), 100):
            network = build_network([w[i] for w in weights_list], [b[i] for b in biases_list])
            logits = tf.reshape(network(X_test).sample(), (X_test.shape[0], 1))
            event_set = tf.expand_dims(e_test.astype(np.int32), axis=1)
            risk_set = tf.convert_to_tensor(_make_riskset(t_test), dtype=np.bool_)
            test_loss = loss_fn(y_true=[event_set, risk_set], y_pred=logits)
            test_trace.append(test_loss.numpy())
    
        plt.plot(test_trace, label=f'C{chain_id} test loss')

    plt.legend(loc='best')
    plt.show()

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

if __name__ == "__main__":
    weight_prior = tfd.Normal(0.0, 0.1)
    bias_prior = tfd.Normal(0.0, 1) # near-uniform

    # Load data
    dl = get_data_loader("WHAS").load_data()
    X, y = dl.get_data()
    num_features, cat_features = dl.get_features()

    # Split data in train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=0)

    # Scale data
    X_train, X_test = scale_data(X_train, X_test, cat_features, num_features)

    # Make time/event split
    t_train, e_train = make_time_event_split(y_train)
    t_test, e_test = make_time_event_split(y_test)

    bnn_joint_log_prob = partial(bnn_joint_log_prob_fn, weight_prior, bias_prior, X_train, e_train, t_train)
    num_features = X_train.shape[1]
    n_layers = 3
    layers = [32, 32, 2]
    initial_state = get_initial_state(weight_prior, bias_prior, num_features, layers)

    z = 0
    for s in initial_state:
        print("State shape", s.shape)
    z += s.shape.num_elements()
    print("Total params", z)

    # Results is tuple with shape [chains, trace]
    n_chains = 10
    number_of_steps = 100000
    number_burnin_steps = 10000
    
    # Get MAP estimate
    map_trace = get_map(bnn_joint_log_prob, initial_state, num_iters=25000, save_every=50)
    map_initial_state = [tf.constant(t[-1]) for t in map_trace]
    
    # Use MAP estiamte as initial state
    chains = [sample_hmc(bnn_joint_log_prob, map_initial_state,
                         n_steps=number_of_steps, n_burnin_steps=number_burnin_steps)
              for _ in range(n_chains)]

    # Calculate target accept prob
    for chain_id in range(n_chains):
        log_accept_ratio = chains[chain_id][1][1][number_burnin_steps:]
        target_accept_prob = tf.math.exp(tfp.math.reduce_logmeanexp(tf.minimum(log_accept_ratio, 0.))).numpy()
        print(f'Target acceptance probability for {chain_id}: {round(100*target_accept_prob)}%')

    # Calculate accepted rate
    plt.figure(figsize=(10,6))
    accepted_rates = list()
    for chain_id in range(n_chains):
        accepted_samples = chains[chain_id][1][0][number_burnin_steps:]
        accepted_rate = 100*np.mean(accepted_samples)
        print(f'Acceptance rate chain for {chain_id}: {round(accepted_rate, 2)}%')
        accepted_rates.append(accepted_rate)
        n_accepted_samples = len(accepted_samples)
        n_bins = int(n_accepted_samples/100)
        sample_indicies = np.linspace(0, n_accepted_samples, n_bins)
        means = [np.mean(accepted_samples[:int(idx)]) for idx in sample_indicies[5:]]
        plt.plot(np.arange(len(means)), means)
    plt.show()

    # Find most accepted chain
    max_chain_id = np.argmax(accepted_rates)

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

    # Compute logits across 100 runs
    runs = 100
    logits_cpd = np.zeros((runs, len(X_test)), dtype=np.float32)
    for i in range(0, runs):
        logits_cpd[i,:] = np.reshape(network(X_test).sample(), len(X_test))
    logits = tf.transpose(tf.reduce_mean(logits_cpd, axis=0, keepdims=True))

    # Compute loss
    event_set = tf.expand_dims(e_test.astype(np.int32), axis=1)
    risk_set = tf.convert_to_tensor(_make_riskset(t_test), dtype=np.bool_)
    logits = tf.reshape(logits, (logits.shape[0], 1))
    loss = loss_fn(y_true=[event_set, risk_set], y_pred=logits).numpy()

    # Compute C-index
    predictions = tf.reshape(logits, -1)
    ci = concordance_index_censored(e_test, t_test, predictions)[0]
    print("Training completed, test loss/C-index: %.4f/%.4f" % (round(loss, 4), round(ci, 4)))