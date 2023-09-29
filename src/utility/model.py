matplotlib_style = 'fivethirtyeight'
import matplotlib.pyplot as plt; plt.style.use(matplotlib_style)
import numpy as np
import tensorflow as tf
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model.coxph import BreslowEstimator
matplotlib_style = 'fivethirtyeight'
import matplotlib.pyplot as plt; plt.style.use(matplotlib_style)
from sklearn.model_selection import train_test_split
from utility.training import get_data_loader, scale_data, make_time_event_split
from tools.model_builder import make_mcd_model, make_mlp_model, make_vi_model
from utility.config import load_config
from utility.loss import CoxPHLoss
import paths as pt
from utility.survival import get_breslow_survival_times, compute_survival_times
import joblib

curr_dir = os.getcwd()
root_dir = Path(curr_dir).absolute().parent

def load_sota_model(dataset_name, model_name):
    return joblib.load(Path.joinpath(pt.MODELS_DIR,
                                     f"{dataset_name.lower()}_{model_name.lower()}.joblib"))

def load_mlp_model(dataset_name, n_input_dims):
    config = load_config(pt.MLP_CONFIGS_DIR, f"{dataset_name}.yaml")
    optimizer = tf.keras.optimizers.deserialize(config['optimizer'])
    loss_fn = CoxPHLoss()
    activation_fn = config['activiation_fn']
    layers = config['network_layers']
    dropout_rate = config['dropout_rate']
    l2_reg = config['l2_reg']
    mlp_model = make_mlp_model(input_shape=n_input_dims, output_dim=1,
                               layers=layers, activation_fn=activation_fn,
                               dropout_rate=dropout_rate, regularization_pen=l2_reg)
    mlp_model.load_weights(f'{root_dir}/models/{dataset_name.lower()}_mlp_coxphloss')
    mlp_model.compile(loss=loss_fn, optimizer=optimizer)
    return mlp_model

def load_vi_model(dataset_name, n_train_samples, n_input_dims):
    config = load_config(pt.MLP_CONFIGS_DIR, f"{dataset_name}.yaml")
    optimizer = tf.keras.optimizers.deserialize(config['optimizer'])
    loss_fn = CoxPHLoss()
    activation_fn = config['activiation_fn']
    layers = config['network_layers']
    dropout_rate = config['dropout_rate']
    l2_reg = config['l2_reg']
    vi_model = make_vi_model(n_train_samples=n_train_samples, input_shape=n_input_dims,
                            output_dim=2, layers=layers, activation_fn=activation_fn,
                            dropout_rate=dropout_rate, regularization_pen=l2_reg)
    vi_model.load_weights(f'{root_dir}/models/{dataset_name.lower()}_vi_coxphloss')
    vi_model.compile(loss=loss_fn, optimizer=optimizer)
    return vi_model

def load_mcd_model(dataset_name, n_input_dims):
    config = load_config(pt.MLP_CONFIGS_DIR, f"{dataset_name}.yaml")
    optimizer = tf.keras.optimizers.deserialize(config['optimizer'])
    loss_fn = CoxPHLoss()
    activation_fn = config['activiation_fn']
    layers = config['network_layers']
    dropout_rate = config['dropout_rate']
    l2_reg = config['l2_reg']
    mcd_model = make_mcd_model(input_shape=n_input_dims, output_dim=2,
                            layers=layers, activation_fn=activation_fn,
                            dropout_rate=dropout_rate, regularization_pen=l2_reg)
    mcd_model.load_weights(f'{root_dir}/models/{dataset_name.lower()}_mcd_coxphloss')
    mcd_model.compile(loss=loss_fn, optimizer=optimizer)
    return mcd_model
    