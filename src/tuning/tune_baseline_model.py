"""
tune_alarm_rsf_wb.py
====================================
Random Survival Forest tune script for Alarm case on WanDB.
"""

import numpy as np
import os
from pathlib import Path
import tensorflow as tf
from tools.model_builder import make_baseline_model
from utility.risk import InputFunction
from utility.loss import CoxPHLoss
from tools import data_loader, model_trainer
from utility.config import load_config
import os
from pathlib import Path
import paths as pt
import random
from sklearn.model_selection import train_test_split, KFold
from tools.preprocessor import Preprocessor

os.environ["WANDB_SILENT"] = "true"
import wandb

np.random.seed(0)
tf.random.set_seed(0)
random.seed(0)

N_RUNS = 10
N_EPOCHS = 10
PROJECT_NAME = "baysurv"
GROUP_NAME = "baseline"

sweep_config = {
    "method": "random", # try grid or random
    "metric": {
      "name": "val_loss",
      "goal": "minimize"
    },
    "parameters": {
        "batch_size": {
            "values": [8, 16, 32, 64, 128]
        },
        "network_layers": {
            "values": [[32], [32, 64], [32, 32], [16, 32],
                       [128], [128, 64], [64, 64, 32]]
        },
        "learning_rate": {
            "values": [0.0001, 0.001, 0.01, 0.1],
        },
        "optimizer": {
            "values": ["Adam", "RMSprop", "SGD", "Nadam"],
        },
        "activation_fn": {
            "values": ["relu", "selu"]
        },
        "dropout": {
            "values": [None, 0.1, 0.25, 0.3, 0.5]
        },
        "l2_kernel_regularization": {
            "values": [None, 0.1, 0.01, 0.001]
        }
    }
}

def main():
    sweep_id = wandb.sweep(sweep_config, project=PROJECT_NAME)
    wandb.agent(sweep_id, train_model, count=N_RUNS)

def train_model():
    config_defaults = {
        'batch_size': [32],
        'network_layers': [32, 32, 32],
        'learning_rate': [0.001],
        'optimizer': ["Adam"],
        'activation_fn': ["relu"],
        'dropout': [None],
        'l2_kernel_regularization': [None]
    }
    
    # Initialize a new wandb run
    wandb.init(config=config_defaults, group=GROUP_NAME)
    wandb.config.epochs = N_EPOCHS
    wandb.config.dataset_name = "CANCER"

    # Load data
    dl = data_loader.CancerDataLoader().load_data()
    num_features, cat_features = dl.get_features()
    X, y = dl.get_data()
    
    # Split data in T1 and HOS
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=0)
    T1, HOS = (X_train, y_train), (X_test, y_test)
    
    # Perform K-fold cross-validation
    split_train_loss_scores, split_train_ci_scores = list(), list()
    split_valid_loss_scores, split_valid_ci_scores = list(), list()
    kf = KFold(n_splits=3, shuffle=True, random_state=0)
    for n_fold, (train, test) in enumerate(kf.split(T1[0], T1[1])):
        ti_X = T1[0].iloc[train]
        ti_y = T1[1][train]
        cvi_X = T1[0].iloc[test]
        cvi_y = T1[1][test]
        
        # Scale data split
        preprocessor = Preprocessor(cat_feat_strat='ignore', num_feat_strat='mean')
        transformer = preprocessor.fit(ti_X, cat_feats=cat_features, num_feats=num_features,
                                       one_hot=True, fill_value=-1)
        ti_X = np.array(transformer.transform(ti_X))
        cvi_X = np.array(transformer.transform(cvi_X))
    
        # Make time event split
        t_train = np.array(ti_y['time'])
        t_valid = np.array(cvi_y['time'])
        e_train = np.array(ti_y['cens'])
        e_valid = np.array(cvi_y['cens'])
        
        train_ds = InputFunction(ti_X, t_train, e_train, batch_size=wandb.config['batch_size'],
                                 drop_last=True, shuffle=True)()
        valid_ds = InputFunction(cvi_X, t_valid, e_valid, batch_size=wandb.config['batch_size'])()
        
        # Make model
        model = make_baseline_model(input_shape=ti_X.shape[1:], output_dim=1, # scalar risk
                                    layers=wandb.config['network_layers'],
                                    activation_fn=wandb.config['activation_fn'],
                                    dropout_rate=wandb.config['dropout'],
                                    regularization_pen=wandb.config['l2_kernel_regularization'])

        # Define optimizer
        if wandb.config['optimizer'] == "Adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=wandb.config.learning_rate)
        elif wandb.config['optimizer'] == "RMSprop":
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=wandb.config.learning_rate)
        elif wandb.config['optimizer'] == "SGD":
            optimizer = tf.keras.optimizers.SGD(learning_rate=wandb.config.learning_rate)
        elif wandb.config['optimizer'] == "Nadam":
            optimizer = tf.keras.optimizers.Nadam(learning_rate=wandb.config.learning_rate)

        # Train model
        loss_fn = CoxPHLoss()
        trainer = model_trainer.Trainer(model=model,
                                        train_dataset=train_ds,
                                        valid_dataset=valid_ds,
                                        test_dataset=None,
                                        optimizer=optimizer,
                                        loss_function=loss_fn,
                                        num_epochs=N_EPOCHS)
        trainer.train_and_evaluate()
        
        split_train_loss_scores.append(trainer.train_loss_scores)
        split_train_ci_scores.append(trainer.train_ci_scores)
        split_valid_loss_scores.append(trainer.valid_loss_scores)
        split_valid_ci_scores.append(trainer.valid_ci_scores)
        
    train_loss_per_epoch = np.mean(split_train_loss_scores, axis=0)
    train_ci_per_epoch = np.mean(split_train_ci_scores, axis=0)
    valid_loss_per_epoch = np.mean(split_valid_loss_scores, axis=0)
    valid_ci_per_epoch = np.mean(split_valid_ci_scores, axis=0)
    
    for i in range(N_EPOCHS): # log mean for every epoch
        wandb.log({'loss': train_loss_per_epoch[i],
                   'ci': train_ci_per_epoch[i],
                   'valid loss': valid_loss_per_epoch[i],
                   'valid ci': valid_ci_per_epoch[i]})
        
if __name__ == "__main__":
    main()