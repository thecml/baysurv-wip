"""
tune_mlp_model.py
====================================
Tuning script for mlp model
--dataset: Dataset name, one of "SUPPORT", "NHANES", "GBSG2", "WHAS500", "FLCHAIN", "METABRIC"
"""

import numpy as np
import os
import tensorflow as tf
from tools.baysurv_builder import make_mlp_model
from utility.risk import InputFunction
from utility.loss import CoxPHLoss
from tools import baysurv_trainer, data_loader
import os
from sklearn.model_selection import train_test_split
from utility.tuning import get_mlp_sweep_config
import argparse
import pandas as pd
from pycox.evaluation import EvalSurv
import numpy as np
import os
import argparse
from tools import data_loader
from sklearn.model_selection import train_test_split
import pandas as pd
from pycox.evaluation import EvalSurv
from utility.training import scale_data, make_time_event_split
from utility.survival import make_event_times
import config as cfg
from sksurv.linear_model.coxph import BreslowEstimator

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

os.environ["WANDB_SILENT"] = "true"
import wandb

N_RUNS = 1
PROJECT_NAME = "baysurv_bo_mlp"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        required=True,
                        default=None)
    args = parser.parse_args()
    global dataset
    if args.dataset:
        dataset = args.dataset
    
    sweep_config = get_mlp_sweep_config()
    sweep_id = wandb.sweep(sweep_config, project=PROJECT_NAME)
    wandb.agent(sweep_id, train_model, count=N_RUNS)

def train_model():
    config_defaults = cfg.MLP_DEFAULT_PARAMS

    # Initialize a new wandb run
    wandb.init(config=config_defaults, group=dataset)
    config = wandb.config
    num_epochs = config['num_epochs']
    batch_size = config['batch_size']
    early_stop = config['early_stop']
    patience = config['patience']

    # Load data
    if dataset == "SUPPORT":
        dl = data_loader.SupportDataLoader().load_data()
    elif dataset == "GBSG2":
        dl = data_loader.GbsgDataLoader().load_data()
    elif dataset == "WHAS500":
        dl = data_loader.WhasDataLoader().load_data()
    elif dataset == "FLCHAIN":
        dl = data_loader.FlchainDataLoader().load_data()
    elif dataset == "METABRIC":
        dl = data_loader.MetabricDataLoader().load_data()
    elif dataset == "SEER":
        dl = data_loader.SeerDataLoader().load_data()
    else:
        raise ValueError("Dataset not found")

    num_features, cat_features = dl.get_features()
    X, y = dl.get_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    X_train, X_valid, y_train, y_valid  = train_test_split(X_train, y_train, test_size=0.25, random_state=0)
    
    # Scale data
    X_train, X_valid, X_test = scale_data(X_train, X_valid, X_test, cat_features, num_features)
    X_train, X_valid, X_test = np.array(X_train), np.array(X_valid), np.array(X_test)
    
    # Make time/event split
    t_train, e_train = make_time_event_split(y_train)
    t_valid, e_valid = make_time_event_split(y_valid)

    # Make event times
    event_times = make_event_times(t_train, e_train)

    # Make datasets
    train_ds = InputFunction(X_train, t_train, e_train, batch_size=batch_size,
                             drop_last=True, shuffle=True)()
    valid_ds = InputFunction(X_valid, t_valid, e_valid, batch_size=batch_size)()
    
    model = make_mlp_model(input_shape=X_train.shape[1:],
                           output_dim=1,
                           layers=config['network_layers'],
                           activation_fn=config['activation_fn'],
                           dropout_rate=config['dropout'],
                           regularization_pen=config['l2_reg'])
    
    # Define optimizer
    if config['optimizer'] == "Adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=wandb.config.learning_rate,
                                             weight_decay=wandb.config.weight_decay)
    elif config['optimizer'] == "SGD":
        optimizer = tf.keras.optimizers.SGD(learning_rate=wandb.config.learning_rate,
                                            weight_decay=wandb.config.weight_decay,
                                            momentum=wandb.config.momentum)
    elif config['optimizer'] == "RMSprop":
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=wandb.config.learning_rate,
                                                weight_decay=wandb.config.weight_decay,
                                                momentum=wandb.config.momentum)
    else:
        raise ValueError("Optimizer not found")
    
    # Train model
    trainer = baysurv_trainer.Trainer(model=model,
                                      model_name="MLP",
                                      train_dataset=train_ds,
                                      valid_dataset=valid_ds,
                                      test_dataset=None,
                                      optimizer=optimizer,
                                      loss_function=CoxPHLoss(),
                                      num_epochs=num_epochs,
                                      event_times=event_times,
                                      early_stop=early_stop,
                                      patience=patience)
    trainer.train_and_evaluate()

    # Compute survival function
    train_predictions = model.predict(X_train).reshape(-1)
    test_predictions = model.predict(X_valid).reshape(-1)
    breslow = BreslowEstimator().fit(train_predictions, e_train, t_train)
    test_surv_fn = breslow.get_survival_function(test_predictions)
    surv_preds = np.row_stack([fn(event_times) for fn in test_surv_fn])
    
    # Compute CTD
    surv_test = pd.DataFrame(surv_preds, columns=event_times)
    ev = EvalSurv(surv_test.T, y_valid["time"], y_valid["event"], censor_surv="km")
    ctd = ev.concordance_td()
    
    # Log to wandb
    wandb.log({"val_ctd": ctd})

if __name__ == "__main__":
    main()