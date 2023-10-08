import numpy as np
import tensorflow as tf

import random
import pandas as pd

from sklearn.model_selection import train_test_split
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw
from sksurv.metrics import integrated_brier_score
from utility.survival import convert_to_structured, make_time_bins
from utility.training import get_data_loader, scale_data, make_time_event_split
from tools.sota_builder import make_cox_model, make_coxnet_model, make_coxboost_model
from tools.sota_builder import make_rsf_model, make_dsm_model, make_dcph_model, make_dcm_model
from tools.sota_builder import make_baycox_model, make_baymtlr_model
from utility.risk import _make_riskset
from tools.bnn_isd_trainer import train_model
from utility.bnn_isd_models import make_ensemble_cox_prediction, make_ensemble_mtlr_prediction
from pathlib import Path
import paths as pt
import joblib
import os
from time import time
from utility.config import load_config
from sksurv.linear_model.coxph import BreslowEstimator
from utility.loss import CoxPHLoss
from pycox.evaluation import EvalSurv
import torch
from utility.survival import compute_unique_counts

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

np.random.seed(0)
tf.random.set_seed(0)
random.seed(0)

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

DATASETS = ["WHAS500"] #"SEER", "GBSG2", "FLCHAIN", "SUPPORT", "METABRIC"
MODEL_NAMES = ["Cox", "CoxNet", "CoxBoost", "RSF", "DSM", "DCPH", "DCM", "BayCox", "BayMTLR"]
results = pd.DataFrame()
loss_fn = CoxPHLoss()

if __name__ == "__main__":
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    
    # For each dataset, train three models (Cox, CoxNet, RSF)
    for dataset_name in DATASETS:
        print(f"Now training dataset {dataset_name}")
        
        # Get batch size for MLP to use for loss calculation
        mlp_config = load_config(pt.MLP_CONFIGS_DIR, f"{dataset_name.lower()}.yaml")
        batch_size = mlp_config['batch_size']
        
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
        
        # Make event times
        mtlr_times = make_time_bins(t_train, event=e_train)
        unique_times = compute_unique_counts(torch.Tensor(e_train), torch.Tensor(t_train))[0]
        if 0 not in unique_times:
            unique_times = torch.cat([torch.tensor([0]).to(unique_times.device), unique_times], 0)
        unique_times = unique_times.numpy()
        
        # Make data for BayCox/BayMTLR models
        data_train = X_train.copy()
        data_train["time"] = pd.Series(y_train['time'])
        data_train["event"] = pd.Series(y_train['event']).astype(int)
        data_test = X_test.copy()
        data_test["time"] = pd.Series(y_test['time'])
        data_test["event"] = pd.Series(y_test['event']).astype(int)
        num_features = X_train.shape[1]

        # Load training parameters
        rsf_config = load_config(pt.RSF_CONFIGS_DIR, f"{dataset_name.lower()}.yaml")
        cox_config = load_config(pt.COX_CONFIGS_DIR, f"{dataset_name.lower()}.yaml")
        coxnet_config = load_config(pt.COXNET_CONFIGS_DIR, f"{dataset_name.lower()}.yaml")
        coxboost_config = load_config(pt.COXBOOST_CONFIGS_DIR, f"{dataset_name.lower()}.yaml")
        dsm_config = load_config(pt.DSM_CONFIGS_DIR, f"{dataset_name.lower()}.yaml")
        dcph_config = load_config(pt.DCPH_CONFIGS_DIR, f"{dataset_name.lower()}.yaml")
        dcm_config = load_config(pt.DCM_CONFIGS_DIR, f"{dataset_name.lower()}.yaml")
        baycox_config = dotdict(load_config(pt.BAYCOX_CONFIGS_DIR, f"{dataset_name.lower()}.yaml"))
        baymtlr_config = dotdict(load_config(pt.BAYMTLR_CONFIGS_DIR, f"{dataset_name.lower()}.yaml"))

        # Make models
        rsf_model = make_rsf_model(rsf_config)
        cox_model = make_cox_model(cox_config)
        coxnet_model = make_coxnet_model(coxnet_config)
        coxboost_model = make_coxboost_model(coxboost_config)
        dsm_model = make_dsm_model(dsm_config)
        dcph_model = make_dcph_model(dcph_config)
        dcm_model = make_dcm_model(dcm_config)
        baycox_model = make_baycox_model(num_features, baycox_config)
        baymtlr_model = make_baymtlr_model(num_features, mtlr_times, baymtlr_config)

        # Train models
        print("Now training Cox")
        cox_train_start_time = time()
        cox_model.fit(X_train, y_train)
        cox_train_time = time() - cox_train_start_time
        print(f"Finished training Cox in {cox_train_time}")

        print("Now training CoxNet")
        coxnet_train_start_time = time()
        coxnet_model.fit(X_train, y_train)
        coxnet_train_time = time() - coxnet_train_start_time
        print(f"Finished training CoxNet in {coxnet_train_time}")

        print("Now training CoxBoost")
        coxboost_train_start_time = time()
        coxboost_model.fit(X_train, y_train)
        coxboost_train_time = time() - coxboost_train_start_time
        print(f"Finished training CoxBoost in {coxboost_train_time}")

        print("Now training RSF")
        rsf_train_start_time = time()
        rsf_model.fit(X_train, y_train)
        rsf_train_time = time() - rsf_train_start_time
        print(f"Finished training RSF in {rsf_train_time}")

        print("Now training DSM")
        dsm_train_start_time = time()
        dsm_model.fit(X_train, pd.DataFrame(y_train))
        dsm_train_time = time() - dsm_train_start_time
        print(f"Finished training DSM in {dsm_train_time}")

        print("Now training DCM")
        dcm_train_start_time = time()
        dcm_model.fit(X_train, pd.DataFrame(y_train))
        dcm_train_time = time() - dcm_train_start_time
        print(f"Finished training DCM in {dcm_train_time}")

        print("Now training DCPH")
        dcph_train_start_time = time()
        dcph_model.fit(np.array(X_train), t_train, e_train, batch_size=dcph_config['batch_size'],
                       iters=dcph_config['iters'], vsize=0.15, learning_rate=dcph_config['learning_rate'],
                       optimizer=dcph_config['optimizer'])
        dcph_train_time = time() - dcph_train_start_time
        print(f"Finished training DCPH in {dcph_train_time}")
        
        print("Now training BayCox")
        baycox_train_start_time = time()
        baycox_model = train_model(baycox_model, data_train, mtlr_times,
                                   config=baycox_config, random_state=0, reset_model=True, device=device)
        
        # Calculate baseline survival on whole train set
        x_train_te = torch.tensor(data_train.drop(["time", "event"], axis=1).values, dtype=torch.float)
        t_train_te = torch.tensor(t_train, dtype=torch.float)
        e_train_te = torch.tensor(e_train, dtype=torch.float)
        baycox_model.calculate_baseline_survival(x_train_te.to(device),
                                                 t_train_te.to(device),
                                                 e_train_te.to(device))
        
        baycox_train_time = time() - baycox_train_start_time
        print(f"Finished training BayCox in {baycox_train_time}")
        
        print("Now training BayMTLR")
        baymtlr_train_start_time = time()
        baymtlr_model = train_model(baymtlr_model, data_train, mtlr_times,
                                    config=baymtlr_config, random_state=0,
                                    reset_model=True, device=device)
        baymtlr_train_time = time() - baymtlr_train_start_time
        print(f"Finished training BayMTLR in {baymtlr_train_time}")
        
        trained_models = [cox_model, coxnet_model, coxboost_model, rsf_model,
                          dsm_model, dcph_model, dcm_model, baycox_model, baymtlr_model]
        train_times = [cox_train_time, coxnet_train_time, coxboost_train_time,
                       rsf_train_time, dsm_train_time, dcph_train_time,
                       dcm_train_time, baycox_train_time, baymtlr_train_time]

        # Compute scores
        for model, model_name, train_time in zip(trained_models, MODEL_NAMES, train_times):
            # Make predictions
            test_start_time = time()
            # Compute loss
            if model_name in ["Cox", "CoxNet", "CoxBoost"]:
                total_loss = list()
                from utility.risk import InputFunction
                X_test_arr = np.array(X_test)
                test_ds = InputFunction(X_test_arr, t_test, e_test, batch_size=batch_size)()
                for x, y in test_ds:
                    y_event = tf.expand_dims(y["label_event"], axis=1)
                    batch_preds = model.predict(x)
                    preds_tn = tf.convert_to_tensor(batch_preds.reshape(len(batch_preds), 1).astype(np.float32))
                    loss = loss_fn(y_true=[y_event, y["label_riskset"]], y_pred=preds_tn).numpy()
                    total_loss.append(loss)
                loss_avg = np.mean(total_loss)
            else:
                loss_avg = np.nan

            # Compute survival function
            if model_name == "DSM":
                surv_preds = pd.DataFrame(model.predict_survival(X_test, times=list(unique_times)), columns=unique_times)
            elif model_name == "DCPH":
                surv_preds = pd.DataFrame(model.predict_survival(X_test, t=list(unique_times)), columns=unique_times)
            elif model_name == "DCM":
                surv_preds = pd.DataFrame(model.predict_survival(X_test, times=list(unique_times)), columns=unique_times)
            elif model_name == "RSF": # uses KM estimator instead
                test_surv_fn = model.predict_survival_function(X_test)
                surv_preds = np.row_stack([fn(unique_times) for fn in test_surv_fn])
            elif model_name == "CoxBoost":
                test_surv_fn = model.predict_survival_function(X_test)
                surv_preds = np.row_stack([fn(unique_times) for fn in test_surv_fn])
            elif model_name == "BayCox":
                baycox_test_data = torch.tensor(data_test.drop(["time", "event"], axis=1).values,
                                                dtype=torch.float, device=device)
                survival_outputs, _, ensemble_outputs = make_ensemble_cox_prediction(model, baycox_test_data,
                                                                                     config=baycox_config)
                surv_preds = survival_outputs.numpy()
            elif model_name == "BayMTLR":
                baycox_test_data = torch.tensor(data_test.drop(["time", "event"], axis=1).values,
                                                dtype=torch.float, device=device)
                survival_outputs, _, ensemble_outputs = make_ensemble_mtlr_prediction(model,
                                                                                      baycox_test_data,
                                                                                      mtlr_times,
                                                                                      config=baymtlr_config)
                surv_preds = survival_outputs.numpy()
            else:
                train_predictions = model.predict(X_train).reshape(-1)
                test_predictions = model.predict(X_test).reshape(-1)
                breslow = BreslowEstimator().fit(train_predictions, e_train, t_train)
                test_surv_fn = breslow.get_survival_function(test_predictions)
                surv_preds = np.row_stack([fn(unique_times) for fn in test_surv_fn])
            
            # Compute CTD, IBS and INBLL
            if model_name == "BayMTLR":
                mtlr_times = torch.cat([torch.tensor([0]).to(mtlr_times.device), mtlr_times], 0)
                surv_test = pd.DataFrame(surv_preds, columns=mtlr_times.numpy())
            else:
                surv_test = pd.DataFrame(surv_preds, columns=unique_times)
            
            ev = EvalSurv(surv_test.T, y_test["time"], y_test["event"], censor_surv="km")
            ctd = ev.concordance_td()
            ibs = ev.integrated_brier_score(unique_times)
            inbll = ev.integrated_nbll(unique_times)
            test_time = time() - test_start_time
            
            # Save to df
            res_df = pd.DataFrame(np.column_stack([loss_avg, ctd, ibs, inbll, train_time, test_time]),
                                  columns=["TestLoss", "TestCTD", "TestIBS", "TestINBLL", "TrainTime", "TestTime"])
            res_df['ModelName'] = model_name
            res_df['DatasetName'] = dataset_name
            results = pd.concat([results, res_df], axis=0)

            # Save model
            if model_name in ["BayCox", "BayMTLR"]:
                path = Path.joinpath(pt.MODELS_DIR, f"{dataset_name.lower()}_{model_name.lower()}.pt")
                torch.save(model.state_dict(), path)
            else:
                path = Path.joinpath(pt.MODELS_DIR, f"{dataset_name.lower()}_{model_name.lower()}.joblib")
                joblib.dump(model, path)

    # Save results
    results.to_csv(Path.joinpath(pt.RESULTS_DIR, f"sota_results.csv"), index=False)
    