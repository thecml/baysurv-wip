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
from utility.plot import plot_training_curves
from tools.model_builder import make_mlp_model, make_vi_model, make_mcd_model
from utility.risk import InputFunction
from utility.loss import CoxPHLoss, CoxPHLossLLA
from pathlib import Path
import paths as pt

np.random.seed(0)
tf.random.set_seed(0)
random.seed(0)

DATASETS = ["WHAS500"]
LOSS_FNS = [CoxPHLoss(), CoxPHLossLLA()]
SPLITS = [1]
N_EPOCHS = 1

if __name__ == "__main__":
    # For each dataset, train models and plot scores
    for loss_fn in LOSS_FNS:
        for dataset_name in DATASETS:
            for split in SPLITS:
                results = pd.DataFrame()
                print(f"Now training {dataset_name} with split {split}")
                
                # Load training parameters
                config = load_config(pt.MLP_CONFIGS_DIR, f"{dataset_name.lower()}.yaml")
                optimizer = tf.keras.optimizers.deserialize(config['optimizer'])
                activation_fn = config['activiation_fn']
                layers = config['network_layers']
                dropout_rate = config['dropout_rate']
                l2_reg = config['l2_reg']
                batch_size = config['batch_size']

                # Load data
                dl = get_data_loader(dataset_name).load_data()
                X, y = dl.get_data()

                num_features, cat_features = dl.get_features()

                # Split data in train and test set
                X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=0)
                
                # Apply split of training data
                X_train = np.array_split(X_train, split)[0]
                y_train = np.array_split(y_train, split)[0]
                    
                # Scale data
                X_train, X_test = scale_data(X_train, X_test, cat_features, num_features)
                X_train = np.array(X_train)
                X_test = np.array(X_test)

                # Make time/event split
                t_train, e_train = make_time_event_split(y_train)
                t_test, e_test = make_time_event_split(y_test)

                # Make event times
                lower, upper = np.percentile(t_test[t_test.dtype.names], [10, 90])
                event_times = np.arange(lower, upper+1)

                # Make data loaders
                train_ds = InputFunction(X_train, t_train, e_train, batch_size=batch_size, drop_last=True, shuffle=True)()
                test_ds = InputFunction(X_test, t_test, e_test, batch_size=batch_size)()

                # Make models
                if not isinstance(loss_fn, CoxPHLossLLA): # dont run MLP with LLA loss
                    mlp_model = make_mlp_model(input_shape=X_train.shape[1:], output_dim=1,
                                            layers=layers, activation_fn=activation_fn,
                                            dropout_rate=dropout_rate, regularization_pen=l2_reg)
                mlp_alea_model = make_mlp_model(input_shape=X_train.shape[1:], output_dim=2,
                                                layers=layers, activation_fn=activation_fn,
                                                dropout_rate=dropout_rate, regularization_pen=l2_reg)
                vi_model = make_vi_model(n_train_samples=X_train.shape[0], input_shape=X_train.shape[1:],
                                         output_dim=2, layers=layers, activation_fn=activation_fn,
                                         dropout_rate=dropout_rate, regularization_pen=l2_reg)
                if not isinstance(loss_fn, CoxPHLossLLA): # dont run VI-EPI with LLA loss
                    vi_epi_model = make_vi_model(n_train_samples=X_train.shape[0], input_shape=X_train.shape[1:],
                                                 output_dim=1, layers=layers, activation_fn=activation_fn,
                                                 dropout_rate=dropout_rate, regularization_pen=l2_reg)
                mcd_model = make_mcd_model(input_shape=X_train.shape[1:], output_dim=2,
                                        layers=layers, activation_fn=activation_fn,
                                        dropout_rate=dropout_rate, regularization_pen=l2_reg)

                # Make trainers
                if not isinstance(loss_fn, CoxPHLossLLA):
                    mlp_trainer = Trainer(model=mlp_model, model_name="MLP",
                                        train_dataset=train_ds, valid_dataset=None,
                                        test_dataset=test_ds, optimizer=optimizer,
                                        loss_function=loss_fn, num_epochs=N_EPOCHS,
                                        event_times=event_times)
                mlp_alea_trainer = Trainer(model=mlp_alea_model, model_name="MLP-ALEA",
                                        train_dataset=train_ds, valid_dataset=None,
                                        test_dataset=test_ds, optimizer=optimizer,
                                        loss_function=loss_fn, num_epochs=N_EPOCHS,
                                        event_times=event_times)
                vi_trainer = Trainer(model=vi_model, model_name="VI",
                                    train_dataset=train_ds, valid_dataset=None,
                                    test_dataset=test_ds, optimizer=optimizer,
                                    loss_function=loss_fn, num_epochs=N_EPOCHS,
                                    event_times=event_times)
                
                if not isinstance(loss_fn, CoxPHLossLLA):
                    vi_epi_trainer = Trainer(model=vi_epi_model, model_name="VI-EPI",
                                            train_dataset=train_ds, valid_dataset=None,
                                            test_dataset=test_ds, optimizer=optimizer,
                                            loss_function=loss_fn, num_epochs=N_EPOCHS,
                                            event_times=event_times)
                
                mcd_trainer = Trainer(model=mcd_model, model_name="MCD",
                                    train_dataset=train_ds, valid_dataset=None,
                                    test_dataset=test_ds, optimizer=optimizer,
                                    loss_function=loss_fn, num_epochs=N_EPOCHS,
                                    event_times=event_times)

                # Train models
                print(f"Started training models for {dataset_name}")
                trainers, model_names = [], []
                if not isinstance(loss_fn, CoxPHLossLLA):
                    mlp_trainer.train_and_evaluate()
                    trainers.append(mlp_trainer)
                    model_names.append("MLP")
                    
                mlp_alea_trainer.train_and_evaluate()
                trainers.append(mlp_alea_trainer)
                model_names.append("MLP-ALEA")
                                
                vi_trainer.train_and_evaluate()
                trainers.append(vi_trainer)
                model_names.append("VI")
                
                if not isinstance(loss_fn, CoxPHLossLLA):
                    vi_epi_trainer.train_and_evaluate()
                    trainers.append(vi_epi_trainer)
                    model_names.append("VI-EPI")
                
                mcd_trainer.train_and_evaluate()
                trainers.append(mcd_trainer)
                model_names.append("MCD")
                
                print(f"Finished training all models for {dataset_name}")
                # Save results per dataset
                for model_name, trainer in zip(model_names, trainers):
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
                    if model_name in ["MLP-ALEA", "VI", "VI-EPI", "MCD"]:
                        test_variance = trainer.test_variance
                    else:
                        test_variance = [0] * len(train_loss)

                    # Save to df
                    print(f"Creating dataframe for model {model_name} for dataset {dataset_name} with trainer {trainer.model_name}")
                    res_df = pd.DataFrame(np.column_stack([train_loss, train_ci, train_ctd, train_ibs, # train
                                                           test_loss, tests_ci, test_ctd, test_ibs, test_variance, # test
                                                           train_times, test_times]), # times
                                        columns=["TrainLoss", "TrainCI", "TrainCTD", "TrainIBS",
                                                "TestLoss", "TestCI", "TestCTD", "TestIBS", "TestVar",
                                                "TrainTime", "TestTime"])
                    res_df['ModelName'] = model_name
                    res_df['DatasetName'] = dataset_name
                    results = pd.concat([results, res_df], axis=0)
                    print(f"Completed dataframe for model {model_name} for dataset {dataset_name} with trainer {trainer.model_name}")

                    # Save model weights
                    model = trainer.model
                    if split == 1:
                        split_name = "full"
                    elif split == 2:
                        split_name = "half"
                    else:
                        split_name = "quarter"
                    
                    loss_fn_name = loss_fn.__class__.__name__
                    path = Path.joinpath(pt.MODELS_DIR, f"{dataset_name.lower()}_{model_name.lower()}_{split_name}_{loss_fn_name.lower()}/")
                    model.save_weights(path)

                # Save results
                print(results)
                results.to_csv(Path.joinpath(pt.RESULTS_DIR, f"baysurv_{dataset_name.lower()}_{split_name}_{loss_fn_name.lower()}_results.csv"), index=False)