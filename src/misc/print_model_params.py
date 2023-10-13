import pandas as pd
import paths as pt
from pathlib import Path
import glob
import os
import numpy as np
import tensorflow as tf
from pathlib import Path
matplotlib_style = 'fivethirtyeight'
import matplotlib.pyplot as plt; plt.style.use(matplotlib_style)             
from sklearn.model_selection import train_test_split
from utility.training import get_data_loader, scale_data, make_time_event_split
from utility.config import load_config
from utility.loss import CoxPHLoss
import paths as pt
from utility.survival import compute_survival_times
from utility.model import load_mlp_model, load_mlp_alea_model, load_vi_model, load_vi_epi_model, load_mcd_model
import math
from utility.survival import coverage, make_event_times, compute_survival_function
import torch
from utility.model import load_sota_model
from sksurv.linear_model.coxph import BreslowEstimator

DATASET_NAME = "WHAS500"
MODEL_NAMES = ["MLP", "MLP-ALEA", "VI", "VI-EPI", "MCD"]

if __name__ == "__main__":
    # Load config
    config = load_config(pt.MLP_CONFIGS_DIR, f"{DATASET_NAME}.yaml")
    optimizer = tf.keras.optimizers.deserialize(config['optimizer'])
    loss_fn = CoxPHLoss()
    activation_fn = config['activiation_fn']
    layers = config['network_layers']
    dropout_rate = config['dropout_rate']
    l2_reg = config['l2_reg']
    batch_size = config['batch_size']

    # Load data
    dl = get_data_loader(DATASET_NAME).load_data()
    X, y = dl.get_data()
    num_features, cat_features = dl.get_features()

    # Split data in train, valid and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    X_train, X_valid, y_train, y_valid  = train_test_split(X_train, y_train, test_size=0.25, random_state=0)

    # Scale data
    X_train, X_valid, X_test = scale_data(X_train, X_valid, X_test, cat_features, num_features)
    X_train, X_valid, X_test = np.array(X_train), np.array(X_valid), np.array(X_test)

    # Load models
    n_input_dims = X_train.shape[1:]
    n_train_samples = X_train.shape[0]
    mlp_model = load_mlp_model(DATASET_NAME, n_input_dims)
    mlp_alea_model = load_mlp_alea_model(DATASET_NAME, n_input_dims)
    vi_model = load_vi_model(DATASET_NAME, n_train_samples, n_input_dims)
    vi_epi_model = load_vi_epi_model(DATASET_NAME, n_train_samples, n_input_dims)
    mcd_model = load_mcd_model(DATASET_NAME, n_input_dims)

    # Print total paramters
    models = [mlp_model, mlp_alea_model, vi_model, vi_epi_model, mcd_model]
    for (model_name, model) in zip(MODEL_NAMES, models):
        total_parameters = 0
        for variable in model.trainable_variables:
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim
            total_parameters += variable_parameters
        print(f'{model_name} has {total_parameters} total parameters')
