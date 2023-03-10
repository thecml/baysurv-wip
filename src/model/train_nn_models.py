import tensorflow as tf
from tools.model_builder import make_baseline_model
from utility.risk import InputFunction
from utility.loss import CoxPHLoss
from tools import data_loader, model_trainer
from utility.config import load_config
from utility.training import get_data_loader, scale_data, make_time_event_split
import os
from pathlib import Path
import paths as pt
import numpy as np
import random
import tensorflow as tf
from tools.model_builder import make_baseline_model, make_vi_model, make_mc_model
from utility.risk import InputFunction
from utility.loss import CoxPHLoss
from utility.config import load_config
import os
import numpy as np
from pathlib import Path
import paths as pt
matplotlib_style = 'fivethirtyeight'
import matplotlib.pyplot as plt; plt.style.use(matplotlib_style)
from sklearn.model_selection import train_test_split
from tools.preprocessor import Preprocessor
import pandas as pd

np.random.seed(0)
tf.random.set_seed(0)
random.seed(0)

DATASETS = ["WHAS", "GBSG", "FLCHAIN", "SUPPORT", "METABRIC"]
MODEL_NAMES = ["Baseline", "MC"]
N_EPOCHS = 2
BATCH_SIZE = 32
results = pd.DataFrame()

if __name__ == "__main__":
    # For each dataset, train three models (Baseline, VI, MC) and plot scores
    for dataset in DATASETS:
        # Load training parameters
        config = load_config(pt.CONFIGS_DIR, f"{dataset.lower()}_arch.yaml")            
        optimizer = tf.keras.optimizers.deserialize(config['optimizer'])
        custom_objects = {"CoxPHLoss": CoxPHLoss()}
        with tf.keras.utils.custom_object_scope(custom_objects):
            loss_fn = tf.keras.losses.deserialize(config['loss_fn'])
        activation_fn = config['activiation_fn']
        layers = config['network_layers']
        dropout_rate = config['dropout_rate']
        l2_reg = config['l2_reg']
        
        # Load data
        dl = get_data_loader(dataset).load_data()
        X, y = dl.get_data()
        num_features, cat_features = dl.get_features()
        
        # Split data in train and test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=0)
                
        # Scale data
        X_train, X_test = scale_data(X_train, X_test, cat_features, num_features)
        
        # Make time/event split
        t_train, e_train = make_time_event_split(y_train)
        t_test, e_test = make_time_event_split(y_test)
        
        # Make data loaders
        train_ds = InputFunction(X_train, t_train, e_train, batch_size=BATCH_SIZE, drop_last=True, shuffle=True)()
        test_ds = InputFunction(X_test, t_test, e_test, batch_size=BATCH_SIZE)()
        
        # Make models
        baseline_model = make_baseline_model(input_shape=X_train.shape[1:], output_dim=1,
                                             layers=layers, activation_fn=activation_fn,
                                             dropout_rate=dropout_rate, regularization_pen=l2_reg)
        vi_model = make_vi_model(n_train_samples=X_train.shape[0], input_shape=X_train.shape[1:],
                                 output_dim=2, layers=layers, activation_fn=activation_fn,
                                 dropout_rate=dropout_rate, regularization_pen=l2_reg)
        mc_model = make_mc_model(input_shape=X_train.shape[1:], output_dim=2,
                                 layers=layers, activation_fn=activation_fn,
                                 dropout_rate=dropout_rate, regularization_pen=l2_reg)
        
        # Make trainers
        baseline_trainer = model_trainer.Trainer(model=baseline_model, model_type="BASELINE",
                                                 train_dataset=train_ds, valid_dataset=None,
                                                 test_dataset=test_ds, optimizer=optimizer,
                                                 loss_function=loss_fn, num_epochs=N_EPOCHS)
        mc_trainer = model_trainer.Trainer(model=mc_model, model_type="MC",
                                           train_dataset=train_ds, valid_dataset=None,
                                           test_dataset=test_ds, optimizer=optimizer,
                                           loss_function=loss_fn, num_epochs=N_EPOCHS)
        
        # Train models
        baseline_trainer.train_and_evaluate()
        mc_trainer.train_and_evaluate()
        
        # Save results
        trainers = [baseline_trainer, mc_trainer]
        for model_name, trainer in zip(MODEL_NAMES, trainers):
            loss = trainer.test_loss_scores
            ci = trainer.test_ci_scores
            ctd = trainer.test_ctd_scores
            ibs = trainer.test_ibs_scores
            res_df = pd.DataFrame(np.column_stack([loss, ci, ctd, ibs]),
                                  columns=["Loss", "CI", "CTD", "IBS"])
            res_df['Dataset'] = dataset
            res_df['Model'] = model_name
            results = pd.concat([results, res_df], axis=0)
        
    # TODO: Add call to plot performance
            
    print(0)
        
        
