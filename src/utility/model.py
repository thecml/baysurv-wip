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
from utility.training import get_data_loader, scale_data, split_time_event
from tools.baysurv_builder import make_mcd_model, make_mlp_model, make_vi_model, make_sngp_model
from utility.config import load_config
from utility.loss import CoxPHLoss, CoxPHLossGaussian
import paths as pt
from utility.survival import compute_nondeterministic_survival_curve, compute_survival_times
import joblib
import numpy as np

curr_dir = os.getcwd()
root_dir = Path(curr_dir).absolute().parent # TODO: Fix this to properly set path

def map_model_name(model_name):
    if model_name == "MLP":
        model_name = "Baseline (MLP)"
    elif model_name == "SNGP":
        model_name = "+ SNGP"
    elif model_name == "VI":
        model_name = "+ VI"
    elif model_name == "VI-VA":
        model_name = "+ VI-VA"
    elif model_name == "MCD":
        model_name = "+ MCD"
    elif model_name == "MCD-VA":
        model_name = "+ MCD-VA"
    elif model_name == "cox":
        model_name = "CoxPH"
    elif model_name == "coxnet":
        model_name = "CoxNet"
    elif model_name == "coxboost":
        model_name = "CoxBoost"
    elif model_name == "rsf":
        model_name = "RSF"
    elif model_name == "dsm":
        model_name = "DSM"
    elif model_name == "dcm":
        model_name = "DCM"
    elif model_name == "baycox":
        model_name = "BayCox"
    elif model_name == "baymtlr":
        model_name = "BayMTLR"
    return model_name

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
    mlp_model.load_weights(f'{root_dir}/models/{dataset_name.lower()}_mlp')
    mlp_model.compile(loss=loss_fn, optimizer=optimizer)
    return mlp_model

def load_sngp_model(dataset_name, n_input_dims):
    config = load_config(pt.MLP_CONFIGS_DIR, f"{dataset_name}.yaml")
    optimizer = tf.keras.optimizers.deserialize(config['optimizer'])
    loss_fn = CoxPHLoss()
    activation_fn = config['activiation_fn']
    layers = config['network_layers']
    dropout_rate = config['dropout_rate']
    l2_reg = config['l2_reg']
    mlp_model = make_sngp_model(input_shape=n_input_dims, output_dim=1,
                                layers=layers, activation_fn=activation_fn,
                                dropout_rate=dropout_rate, regularization_pen=l2_reg)
    mlp_model.load_weights(f'{root_dir}/models/{dataset_name.lower()}_sngp')
    mlp_model.compile(loss=loss_fn, optimizer=optimizer)
    return mlp_model

def load_mlp_alea_model(dataset_name, n_input_dims):
    config = load_config(pt.MLP_CONFIGS_DIR, f"{dataset_name}.yaml")
    optimizer = tf.keras.optimizers.deserialize(config['optimizer'])
    loss_fn = CoxPHLoss()
    activation_fn = config['activiation_fn']
    layers = config['network_layers']
    dropout_rate = config['dropout_rate']
    l2_reg = config['l2_reg']
    mlp_model = make_mlp_model(input_shape=n_input_dims, output_dim=2,
                               layers=layers, activation_fn=activation_fn,
                               dropout_rate=dropout_rate, regularization_pen=l2_reg)
    mlp_model.load_weights(f'{root_dir}/models/{dataset_name.lower()}_mlp-alea')
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
    vi_model.load_weights(f'{root_dir}/models/{dataset_name.lower()}_vi')
    vi_model.compile(loss=loss_fn, optimizer=optimizer)
    return vi_model

def load_vi_epi_model(dataset_name, n_train_samples, n_input_dims):
    config = load_config(pt.MLP_CONFIGS_DIR, f"{dataset_name}.yaml")
    optimizer = tf.keras.optimizers.deserialize(config['optimizer'])
    loss_fn = CoxPHLoss()
    activation_fn = config['activiation_fn']
    layers = config['network_layers']
    dropout_rate = config['dropout_rate']
    l2_reg = config['l2_reg']
    vi_model = make_vi_model(n_train_samples=n_train_samples, input_shape=n_input_dims,
                             output_dim=1, layers=layers, activation_fn=activation_fn,
                             dropout_rate=dropout_rate, regularization_pen=l2_reg)
    vi_model.load_weights(f'{root_dir}/models/{dataset_name.lower()}_vi-epi')
    vi_model.compile(loss=loss_fn, optimizer=optimizer)
    return vi_model

def load_mcd_model(dataset_name, n_input_dims):
    config = load_config(pt.MLP_CONFIGS_DIR, f"{dataset_name}.yaml")
    optimizer = tf.keras.optimizers.deserialize(config['optimizer'])
    loss_fn = CoxPHLossGaussian()
    activation_fn = config['activiation_fn']
    layers = config['network_layers']
    dropout_rate = config['dropout_rate']
    l2_reg = config['l2_reg']
    mcd_model = make_mcd_model(input_shape=n_input_dims, output_dim=2,
                               layers=layers, activation_fn=activation_fn,
                               dropout_rate=dropout_rate, regularization_pen=l2_reg)
    mcd_model.load_weights(f'{root_dir}/models/{dataset_name.lower()}_mcd')
    mcd_model.compile(loss=loss_fn, optimizer=optimizer)
    return mcd_model
    