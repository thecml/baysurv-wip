import tensorflow as tf
import numpy as np
import pandas as pd
import random

from sklearn.model_selection import train_test_split
matplotlib_style = 'fivethirtyeight'
import matplotlib.pyplot as plt; plt.style.use(matplotlib_style)

from tools.baysurv_trainer import Trainer
from utility.config import load_config
from utility.training import get_data_loader, scale_data, split_time_event
from utility.plot import plot_training_curves
from tools.baysurv_builder import make_mlp_model, make_vi_model, make_mcd_model
from utility.risk import InputFunction
from utility.loss import CoxPHLoss, CoxPHLossLLA
from pathlib import Path
import paths as pt
from utility.survival import calculate_event_times, calculate_percentiles

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

np.seterr(divide ='ignore')
np.seterr(invalid='ignore')

np.random.seed(0)
tf.random.set_seed(0)
random.seed(0)

DATASET = "SEER"
N_EPOCHS = 10

if __name__ == "__main__":
    dataset_name = DATASET
    print(f"Now training {dataset_name}")        
    
    # Load training parameters
    config = load_config(pt.MLP_CONFIGS_DIR, f"{dataset_name.lower()}.yaml")
    optimizer = tf.keras.optimizers.deserialize(config['optimizer'])
    activation_fn = config['activiation_fn']
    layers = config['network_layers']
    dropout_rate = config['dropout_rate']
    l2_reg = config['l2_reg']
    batch_size = config['batch_size']
    early_stop = config['early_stop']
    patience = config['patience']

    # Load data
    dl = get_data_loader(dataset_name).load_data()
    X, y = dl.get_data()
    num_features, cat_features = dl.get_features()
    
    # Split data in train, valid and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    X_train, X_valid, y_train, y_valid  = train_test_split(X_train, y_train, test_size=0.25, random_state=0)
    
    # Scale data
    X_train, X_valid, X_test = scale_data(X_train, X_valid, X_test, cat_features, num_features)
    X_train, X_valid, X_test = np.array(X_train), np.array(X_valid), np.array(X_test)

    # Make time/event split
    t_train, e_train = split_time_event(y_train)
    t_valid, e_valid = split_time_event(y_valid)
    t_test, e_test = split_time_event(y_test)

    # Make event times
    event_times = calculate_event_times(t_train, e_train)
    
    # Calculate quantiles
    event_times_pct = calculate_percentiles(event_times)

    # Make data loaders
    train_ds = InputFunction(X_train, t_train, e_train, batch_size=batch_size, drop_last=True, shuffle=True)()
    valid_ds = InputFunction(X_valid, t_valid, e_valid, batch_size=batch_size)()

    # Make models
    vi_model = make_vi_model(n_train_samples=X_train.shape[0], input_shape=X_train.shape[1:],
                             output_dim=2, layers=layers, activation_fn=activation_fn,
                             dropout_rate=dropout_rate, regularization_pen=l2_reg)

    # Make trainers
    trainer = Trainer(model=vi_model, model_name="VI",
                      train_dataset=train_ds, valid_dataset=valid_ds,
                      test_dataset=None, optimizer=optimizer,
                      loss_function=CoxPHLossLLA(), num_epochs=N_EPOCHS,
                      event_times=event_times, early_stop=early_stop,
                      patience=patience, event_times_pct=event_times_pct)

    # Train models
    print(f"Started training models for {dataset_name}")
    trainers, model_names = [], []
    trainer.train_and_evaluate()
    trainers.append(trainer)
    model_names.append("VI")
    
    # Train
    train_loss = trainer.train_loss
    train_ctd = trainer.train_ci_scores
    train_ibs = trainer.train_ibs_scores
    train_inbll = trainer.train_inbll_scores
    train_ece = trainer.train_ici_scores
    train_e50 = trainer.train_e50_scores
    train_times = trainer.train_times
    best_ep = trainer.best_ep

    # Valid
    valid_loss = trainer.valid_loss
    res_df = pd.DataFrame(np.column_stack([train_loss, train_ctd, train_ibs, train_inbll, train_ece, train_e50, valid_loss]), # valid
                          columns=["TrainLoss", "TrainCTD", "TrainIBS", "TrainINBLL", "TrainECE", "TrainE50", "ValidLoss"])
    print(res_df)