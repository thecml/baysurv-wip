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
from tools.baysurv_builder import make_mlp_model, make_vi_model, make_mcd_model, make_sngp_model
from utility.risk import InputFunction
from utility.loss import CoxPHLoss, CoxPHLossLLA
from pathlib import Path
import paths as pt
from utility.survival import (calculate_event_times, calculate_percentiles, convert_to_structured,
                              compute_deterministic_survival_curve, compute_nondeterministic_survival_curve)
from utility.training import make_stratified_split
from time import time
from tools.evaluator import LifelinesEvaluator
from pycox.evaluation import EvalSurv
import math
from utility.survival import coverage
from scipy.stats import chisquare
import torch
from utility.survival import survival_probability_calibration
from tools.Evaluations.util import make_monotonic, check_monotonicity

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

np.seterr(divide ='ignore')
np.seterr(invalid='ignore')

np.random.seed(0)
tf.random.set_seed(0)
random.seed(0)

loss_fn = CoxPHLoss()
training_results, test_results = pd.DataFrame(), pd.DataFrame()

DATASETS = ["SUPPORT", "SEER", "METABRIC", "MIMIC"]
MODELS = ["MLP", "MCD", "SNGP"]
N_EPOCHS = 1

if __name__ == "__main__":
    # For each dataset, train models and plot scores
    for dataset_name in DATASETS:
        test_results = pd.DataFrame()
        
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
        n_samples_train = config['n_samples_train']
        n_samples_valid = config['n_samples_valid']
        n_samples_test = config['n_samples_test']

        # Load data
        dl = get_data_loader(dataset_name).load_data()
        num_features, cat_features = dl.get_features()
        df = dl.get_data()
        
        # Split data
        df_train, df_valid, df_test = make_stratified_split(df, stratify_colname='both', frac_train=0.7,
                                                            frac_valid=0.1, frac_test=0.2, random_state=0)
        X_train = df_train[cat_features+num_features]
        X_valid = df_valid[cat_features+num_features]
        X_test = df_test[cat_features+num_features]
        y_train = convert_to_structured(df_train["time"], df_train["event"])
        y_valid = convert_to_structured(df_valid["time"], df_valid["event"])
        y_test = convert_to_structured(df_test["time"], df_test["event"])
        
        # Scale data
        X_train, X_valid, X_test = scale_data(X_train, X_valid, X_test, cat_features, num_features)
        
        # Convert to array
        X_train = np.array(X_train)
        X_valid = np.array(X_valid)
        X_test = np.array(X_test)

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
        test_ds = InputFunction(X_test, t_test, e_test, batch_size=batch_size)()

        # Make models
        for model_name in MODELS:
            if model_name == "MLP":
                model = make_mlp_model(input_shape=X_train.shape[1:], output_dim=1,
                                       layers=layers, activation_fn=activation_fn,
                                       dropout_rate=dropout_rate, regularization_pen=l2_reg)
                loss_function = CoxPHLoss()
            elif model_name == "VI":
                model = make_vi_model(n_train_samples=X_train.shape[0], input_shape=X_train.shape[1:],
                                      output_dim=2, layers=layers, activation_fn=activation_fn,
                                      dropout_rate=dropout_rate)
                loss_function = CoxPHLossLLA()
            elif model_name == "SNGP":
                model = make_sngp_model(input_shape=X_train.shape[1:],
                                        output_dim=1, layers=layers, activation_fn=activation_fn,
                                        dropout_rate=dropout_rate, regularization_pen=l2_reg)
                loss_function = CoxPHLoss()
            else:
                model = make_mcd_model(input_shape=X_train.shape[1:], output_dim=2,
                                       layers=layers, activation_fn=activation_fn,
                                       dropout_rate=dropout_rate, regularization_pen=l2_reg)
                loss_function=CoxPHLossLLA()
            
            # Train model
            trainer = Trainer(model=model, model_name=model_name,
                              train_dataset=train_ds, valid_dataset=valid_ds,
                              test_dataset=None, optimizer=optimizer,
                              loss_function=loss_function, num_epochs=N_EPOCHS,
                              early_stop=early_stop, patience=patience,
                              n_samples_train=n_samples_train,
                              n_samples_valid=n_samples_valid,
                              n_samples_test=n_samples_test)
            train_start_time = time()
            trainer.train_and_evaluate()
            train_time = time() - train_start_time
            
            # Get model for best epoch
            best_ep = trainer.best_ep
            status = trainer.checkpoint.restore(Path.joinpath(pt.MODELS_DIR, f"ckpt-{best_ep}"))
            model = trainer.model

            # Compute survival function
            test_start_time = time()            
            if model_name in ["MLP", "SNGP"]:
                surv_preds = compute_deterministic_survival_curve(model, X_train, X_test,
                                                                  e_train, t_train, event_times, model_name)
            else:
                surv_preds = np.mean(compute_nondeterministic_survival_curve(model, np.array(X_train), np.array(X_test),
                                                                             e_train, t_train, event_times,
                                                                             n_samples_train, n_samples_test), axis=0)
            test_time = time() - test_start_time
            
            # Check monotonicity
            if not check_monotonicity(surv_preds):
                surv_preds = make_monotonic(surv_preds, event_times, method='ceil')
            
            # Make dataframe
            surv_preds = pd.DataFrame(surv_preds, dtype=np.float64, columns=event_times)
            
            # Sanitize
            surv_preds = surv_preds.fillna(0).replace([np.inf, -np.inf], 0)
            bad_idx = surv_preds[surv_preds.iloc[:,0] < 0.5].index # check we are above 0.5
            surv_preds = surv_preds.drop(bad_idx).reset_index(drop=True)
            y_test = np.delete(y_test, bad_idx)
            
            # Compute metrics
            lifelines_eval = LifelinesEvaluator(surv_preds.T, y_test["time"], y_test["event"], t_train, e_train)
            ibs = lifelines_eval.integrated_brier_score()
            mae_hinge = lifelines_eval.mae(method="Hinge")
            mae_pseudo = lifelines_eval.mae(method="Pseudo_obs")
            d_calib = 1 if lifelines_eval.d_calibration()[0] > 0.05 else 0
            km_mse = lifelines_eval.km_calibration()
            ev = EvalSurv(surv_preds.T, y_test["time"], y_test["event"], censor_surv="km")
            inbll = ev.integrated_nbll(event_times)
            ci = ev.concordance_td()
            
            # Calculate C-cal for BNN models
            if model_name in ["MLP-ALEA", "VI-EPI", "MCD-EPI", "MCD"]:
                surv_probs = compute_nondeterministic_survival_curve(model, X_train, X_test,
                                                                     e_train, t_train, event_times,
                                                                     n_samples_train, n_samples_test)
                credible_region_sizes = np.arange(0.1, 1, 0.1)
                surv_times = torch.from_numpy(surv_probs)
                coverage_stats = {}
                for percentage in credible_region_sizes:
                    drop_num = math.floor(0.5 * n_samples_test * (1 - percentage))
                    lower_outputs = torch.kthvalue(surv_times, k=1 + drop_num, dim=0)[0]
                    upper_outputs = torch.kthvalue(surv_times, k=n_samples_test - drop_num, dim=0)[0]
                    coverage_stats[percentage] = coverage(event_times, upper_outputs, lower_outputs, t_test, e_test)
                data = [list(coverage_stats.keys()), list(coverage_stats.values())]
                _, pvalue = chisquare(data)
                alpha = 0.05
                if pvalue[0] <= alpha:
                    c_calib = 0
                else:
                    c_calib = 1
            else:
                c_calib = 0
            
            # Compute calibration curves
            deltas = dict()
            for t0 in event_times_pct.values():
                _, _, _, deltas_t0 = survival_probability_calibration(surv_preds,
                                                                      y_test["time"],
                                                                      y_test["event"],
                                                                      t0)
                deltas[t0] = deltas_t0
            ici = deltas[t0].mean()
            
            # Save to df
            metrics = [ci, ibs, mae_hinge, mae_pseudo, d_calib, km_mse, inbll, c_calib, ici, train_time, test_time]
            res_df = pd.DataFrame(np.column_stack(metrics), columns=["CI", "IBS", "MAEHinge", "MAEPseudo", "DCalib", "KM",
                                                                     "INBLL", "CCalib", "ICI", "TrainTime", "TestTime"])
            res_df['ModelName'] = model_name
            res_df['DatasetName'] = dataset_name
            test_results = pd.concat([test_results, res_df], axis=0)
            
            # Save loss and variance from training
            train_loss = trainer.train_loss
            train_variance = trainer.train_variance
            valid_loss = trainer.valid_loss
            valid_variance = trainer.valid_variance
            res_df = pd.DataFrame(np.column_stack([train_loss, train_variance, valid_loss, valid_variance]),
                                  columns=["TrainLoss", "TrainVariance", "ValidLoss", "ValidVariance"])
            res_df['ModelName'] = model_name
            res_df['DatasetName'] = dataset_name
            training_results = pd.concat([training_results, res_df], axis=0)
            
            # Save model
            path = Path.joinpath(pt.MODELS_DIR, f"{dataset_name.lower()}_{model_name.lower()}/")
            model.save_weights(path)
            
    # Save results
    training_results.to_csv(Path.joinpath(pt.RESULTS_DIR, f"baysurv_training_results.csv"), index=False)
    test_results.to_csv(Path.joinpath(pt.RESULTS_DIR, f"baysurv_test_results.csv"), index=False)
        