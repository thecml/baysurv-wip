import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sksurv.linear_model.coxph import BreslowEstimator
from utility.training import get_data_loader, scale_data, make_time_event_split
from utility.survival import survival_probability_calibration
from utility.model import load_mlp_model, load_sota_model, load_vi_model, load_mcd_model
from utility.survival import compute_survival_function
from utility.plot import plot_calibration_curves
from collections import defaultdict
from pathlib import Path
import paths as pt
from utility.survival import make_time_bins

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

DATASETS = ["WHAS500"]
RUNS = {'MLP': 1, 'VI': 100, 'MCD': 100}

if __name__ == "__main__":
    for dataset_name in DATASETS:
        # Load data
        dl = get_data_loader(dataset_name).load_data()
        X, y = dl.get_data()
        num_features, cat_features = dl.get_features()

        # Split data in train, valid and test set
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            train_size=0.7,
                                                            random_state=0)

        # Scale data
        X_train, X_test = scale_data(X_train, X_test, cat_features, num_features)
        X_train = np.array(X_train)
        X_test = np.array(X_test)

        # Make time/event split
        t_train, e_train = make_time_event_split(y_train)
        t_test, e_test = make_time_event_split(y_test)

        # Fit Breslow to get unique event times
        cox_model = load_sota_model(dataset_name, "Cox")
        train_predictions = cox_model.predict(X_train)
        breslow = BreslowEstimator().fit(train_predictions, e_train, t_train)
        event_times = breslow.unique_times_
            
        # Calculate quantiles
        percentiles = dict()
        for q in [25, 50, 75]:
            t = int(np.percentile(event_times, q))
            t_nearest = find_nearest(event_times, t)
            percentiles[q] = t_nearest

        # Load models
        n_input_dims = X_train.shape[1:]
        n_train_samples = X_train.shape[0]
        rsf_model = load_sota_model(dataset_name, "RSF")
        mlp_model = load_mlp_model(dataset_name, n_input_dims)
        vi_model = load_vi_model(dataset_name, n_train_samples, n_input_dims)
        mcd_model = load_mcd_model(dataset_name, n_input_dims)

        # Compute calibration curves
        pred_obs, predictions, deltas = defaultdict(dict), defaultdict(dict), defaultdict(dict)
        models = {'Cox': cox_model, 'RSF': rsf_model, 'MLP':mlp_model, 'VI': vi_model, 'MCD': mcd_model}
        for t0 in percentiles.values():
            for model_name, model in models.items():
                if type(model).__name__ in ["CoxPHSurvivalAnalysis", "RandomSurvivalForest"]:
                    surv_fn = model.predict_survival_function(X_test)
                    surv_preds = pd.DataFrame(np.row_stack([fn(event_times) for fn in surv_fn]), columns=event_times)
                else:
                    surv_fn = compute_survival_function(model, X_train, X_test, e_train, t_train, event_times, RUNS[model_name])
                    surv_preds = pd.DataFrame(np.mean(surv_fn, axis=0), columns=event_times)
                pred_t0, obs_t0, predictions_at_t0, deltas_t0 = survival_probability_calibration(surv_preds, t_test, e_test, t0)
                pred_obs[t0][model_name] = (pred_t0, obs_t0)
                predictions[t0][model_name] = predictions_at_t0
                deltas[t0][model_name] = deltas_t0
        
        # Compute calibration metrics
        calib_results = pd.DataFrame()
        for t0 in percentiles.values():
            for model_name in models.keys():
                ice = deltas[t0][model_name].mean()
                e50 = np.percentile(deltas[t0][model_name], 50)
                res_sr = pd.Series([model_name, dataset_name, ice, e50],
                                   index=["ModelName", "ICE", "E50", "CIndex"])
            calib_results = pd.concat([calib_results, res_sr.to_frame().T], ignore_index=True)
        calib_results.to_csv(Path.joinpath(pt.RESULTS_DIR, f"baysurv_{dataset_name.lower()}_calibration_results.csv"), index=False)
                            
        # Plot calibration curves
        model_names = models.keys()
        plot_calibration_curves(percentiles, pred_obs, predictions, model_names, dataset_name)