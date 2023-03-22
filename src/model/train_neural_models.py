import tensorflow as tf
import numpy as np
import pandas as pd
import random

from sklearn.model_selection import train_test_split
matplotlib_style = 'fivethirtyeight'
import matplotlib.pyplot as plt; plt.style.use(matplotlib_style)

from tools.model_trainer import Trainer
from utility.config import load_config
from utility.training import get_data_loader, scale_data, make_time_event_split
from utility.plotter import plot_training_curves
from tools.model_builder import make_mlp_model, make_vi_model, make_mcd_model, make_mlp_model
from utility.risk import InputFunction
from utility.loss import CoxPHLoss
from pathlib import Path
import paths as pt
import os

np.random.seed(0)
tf.random.set_seed(0)
random.seed(0)

DATASETS = ["WHAS", "SEER", "GBSG2", "FLCHAIN", "SUPPORT", "METABRIC"]
MODEL_NAMES = ["MLP", "MLP-ALEA", "VI", "VI-EPI", "MCD"]
N_EPOCHS = 25
BATCH_SIZE = 32
results = pd.DataFrame()

if __name__ == "__main__":
    # For each dataset, train models and plot scores
    for dataset_name in DATASETS:
        # Load training parameters
        config = load_config(pt.MLP_CONFIGS_DIR, f"{dataset_name.lower()}.yaml")
        optimizer = tf.keras.optimizers.deserialize(config['optimizer'])
        custom_objects = {"CoxPHLoss": CoxPHLoss()}
        with tf.keras.utils.custom_object_scope(custom_objects):
            loss_fn = tf.keras.losses.deserialize(config['loss_fn'])
        activation_fn = config['activiation_fn']
        layers = config['network_layers']
        dropout_rate = config['dropout_rate']
        l2_reg = config['l2_reg']

        # Load data
        dl = get_data_loader(dataset_name).load_data()
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
        mlp_model = make_mlp_model(input_shape=X_train.shape[1:], output_dim=1,
                                   layers=layers, activation_fn=activation_fn,
                                   dropout_rate=dropout_rate, regularization_pen=l2_reg)
        mlp_alea_model = make_mlp_model(input_shape=X_train.shape[1:], output_dim=2,
                                        layers=layers, activation_fn=activation_fn,
                                        dropout_rate=dropout_rate, regularization_pen=l2_reg)
        vi_model = make_vi_model(n_train_samples=X_train.shape[0], input_shape=X_train.shape[1:],
                                 output_dim=2, layers=layers, activation_fn=activation_fn,
                                 dropout_rate=dropout_rate, regularization_pen=l2_reg)
        vi_epi_model = make_vi_model(n_train_samples=X_train.shape[0], input_shape=X_train.shape[1:],
                                     output_dim=1, layers=layers, activation_fn=activation_fn,
                                     dropout_rate=dropout_rate, regularization_pen=l2_reg)
        mc_model = make_mcd_model(input_shape=X_train.shape[1:], output_dim=2,
                                 layers=layers, activation_fn=activation_fn,
                                 dropout_rate=dropout_rate, regularization_pen=l2_reg)

        # Make trainers
        mlp_trainer = Trainer(model=mlp_model, model_type="MLP",
                              train_dataset=train_ds, valid_dataset=None,
                              test_dataset=test_ds, optimizer=optimizer,
                              loss_function=loss_fn, num_epochs=N_EPOCHS)
        mlp_alea_trainer = Trainer(model=mlp_model, model_type="MLP",
                                   train_dataset=train_ds, valid_dataset=None,
                                   test_dataset=test_ds, optimizer=optimizer,
                                   loss_function=loss_fn, num_epochs=N_EPOCHS)
        vi_trainer = Trainer(model=vi_model, model_type="VI",
                             train_dataset=train_ds, valid_dataset=None,
                             test_dataset=test_ds, optimizer=optimizer,
                             loss_function=loss_fn, num_epochs=N_EPOCHS)
        vi_epi_trainer = Trainer(model=vi_model, model_type="VI",
                                 train_dataset=train_ds, valid_dataset=None,
                                 test_dataset=test_ds, optimizer=optimizer,
                                 loss_function=loss_fn, num_epochs=N_EPOCHS)
        mc_trainer = Trainer(model=mc_model, model_type="MCD",
                             train_dataset=train_ds, valid_dataset=None,
                             test_dataset=test_ds, optimizer=optimizer,
                             loss_function=loss_fn, num_epochs=N_EPOCHS)

        # Train models
        mlp_trainer.train_and_evaluate()
        mlp_alea_trainer.train_and_evaluate()
        vi_trainer.train_and_evaluate()
        vi_epi_trainer.train_and_evaluate()
        mc_trainer.train_and_evaluate()

        # Save results per dataset
        trainers = [mlp_trainer, mlp_alea_trainer, vi_trainer, vi_epi_trainer, mc_trainer]
        for model_name, trainer in zip(MODEL_NAMES, trainers):
            # Training
            train_loss = trainer.train_loss_scores
            train_ci = trainer.train_ci_scores
            train_ctd = trainer.train_ctd_scores
            train_ibs = trainer.train_ibs_scores
            train_times = trainer.train_times

            # Test
            test_loss = trainer.test_loss_scores
            tests_ci = trainer.test_ci_scores
            test_ctd = trainer.test_ctd_scores
            test_ibs = trainer.test_ibs_scores
            test_times = trainer.test_times

            # Save to df
            res_df = pd.DataFrame(np.column_stack([train_loss, train_ci, train_ctd, train_ibs, # train
                                                   test_loss, tests_ci, test_ctd, test_ibs, # test
                                                   train_times, test_times]), # times
                                  columns=["TrainLoss", "TrainCI", "TrainCTD", "TrainIBS",
                                           "TestLoss", "TestCI", "TestCTD", "TestIBS",
                                           "TrainTime", "TestTime"])
            res_df['ModelName'] = model_name
            res_df['DatasetName'] = dataset_name
            results = pd.concat([results, res_df], axis=0)

            # Save model weights
            model = trainer.model
            path = Path.joinpath(pt.MODELS_DIR, f"{model_name.lower()}/")
            model.save_weights(path)

    # Save results
    results.to_csv(Path.joinpath(pt.RESULTS_DIR, f"neural_training_results.csv"), index=False)

    # Plot training curves
    plot_training_curves(results)