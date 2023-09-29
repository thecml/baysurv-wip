import pandas as pd
import paths as pt
import glob
import os
from utility import plot

matplotlib_style = 'fivethirtyeight'
import matplotlib.pyplot as plt; plt.style.use(matplotlib_style)
import numpy as np
import tensorflow as tf
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model.coxph import BreslowEstimator, CoxPHSurvivalAnalysis
matplotlib_style = 'fivethirtyeight'
import matplotlib.pyplot as plt; plt.style.use(matplotlib_style)
from sklearn.model_selection import train_test_split
from utility.training import get_data_loader, scale_data, make_time_event_split
from tools.model_builder import make_mcd_model
from utility.config import load_config
from utility.loss import CoxPHLoss
import paths as pt
from utility.risk import InputFunction
from tools.model_builder import make_cox_model, make_rsf_model
from utility.survival import survival_probability_calibration
import joblib
from tools.model_builder import make_mlp_model
from utility.model import load_mlp_model, load_sota_model, load_vi_model, load_mcd_model
from utility.survival import get_breslow_survival_times
from collections import defaultdict
import seaborn as sns

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

DATASET_NAME = "WHAS500"
RUNS = {'MLP': 1, 'VI': 100, 'MCD': 100}

class _TFColor(object):
    """Enum of colors used in TF docs."""
    red = '#F15854'
    blue = '#5DA5DA'
    orange = '#FAA43A'
    green = '#60BD68'
    pink = '#F17CB0'
    brown = '#B2912F'
    purple = '#B276B2'
    yellow = '#DECF3F'
    gray = '#4D4D4D'
    def __getitem__(self, i):
        return [
            self.red,
            self.orange,
            self.green,
            self.blue,
            self.pink,
            self.brown,
            self.purple,
            self.yellow,
            self.gray,
        ][i % 9]
TFColor = _TFColor()

if __name__ == "__main__":
    # Load data
    dl = get_data_loader(DATASET_NAME).load_data()
    X, y = dl.get_data()
    num_features, cat_features = dl.get_features()

    # Split data in train, valid and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7,
                                                        random_state=0)

    # Scale data
    X_train, X_test = scale_data(X_train, X_test, cat_features, num_features)
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    # Make time/event split
    t_train, e_train = make_time_event_split(y_train)
    t_test, e_test = make_time_event_split(y_test)

    # Fit Breslow to get event times
    cox_model = load_sota_model(DATASET_NAME, "Cox")
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
    rsf_model = load_sota_model(DATASET_NAME, "RSF")
    mlp_model = load_mlp_model(DATASET_NAME, n_input_dims)
    vi_model = load_vi_model(DATASET_NAME, n_train_samples, n_input_dims)
    mcd_model = load_mcd_model(DATASET_NAME, n_input_dims)

    pred_obs, predictions, deltas = defaultdict(dict), defaultdict(dict), defaultdict(dict)
    models = {'Cox': cox_model, 'RSF': rsf_model, 'MLP':mlp_model, 'VI': vi_model, 'MCD': mcd_model}
    for t0 in percentiles.values():
        for model_name, model in models.items():
            if type(model).__name__ in ["CoxPHSurvivalAnalysis", "RandomSurvivalForest"]:
                surv_fn = model.predict_survival_function(X_test)
                surv_preds = pd.DataFrame(np.row_stack([fn(event_times) for fn in surv_fn]), columns=event_times)
            else:
                surv_fn = get_breslow_survival_times(model, X_train, X_test, e_train, t_train,
                                                     event_times, RUNS[model_name])
                surv_preds = pd.DataFrame(np.mean(surv_fn, axis=1), columns=event_times)
            pred_t0, obs_t0, predictions_at_t0, deltas_t0 = survival_probability_calibration(surv_preds, t_test, e_test, t0)
            pred_obs[t0][model_name] = (pred_t0, obs_t0)
            predictions[t0][model_name] = predictions_at_t0
            deltas[t0][model_name] = deltas_t0
       
    '''
    for t0 in percentiles.values():
        cox_ice = deltas[t0]['Cox'].mean()
        mlp_ece = deltas[t0]['MLP'].mean()
        mcd_ece = deltas[t0]['MCD'].mean()
        cox_e50 = np.percentile(deltas[t0]['Cox'], 50)
        mlp_e50 = np.percentile(deltas[t0]['MLP'], 50)
        mcd_e50 = np.percentile(deltas[t0]['MCD'], 50)
        print(f"Cox/MLP/MCD ICI at {t0} = {cox_ice}/{mlp_ece}/{mcd_ece}")
        print(f"Cox/MLP/MCD E50 at {t0} = {cox_e50}/{mlp_e50}/{mcd_e50}")
    '''     
        
    # plot our results
    fig, axs = plt.subplots(1, 3, figsize=(14, 8))
    plt.rcParams.update({'axes.titlesize': 'small'})
    color = "tab:blue"
    labels = list()
    for (pct, t0), ax in zip(percentiles.items(), axs.ravel()):
        twin_ax = ax.twinx()
        twin_ax.tick_params(axis="y")
        for model_idx, model_name in enumerate(models.keys()):
            pred = pred_obs[t0][model_name][0]
            obs = pred_obs[t0][model_name][1]
            label, = ax.plot(pred, obs, color=TFColor[model_idx])
            labels.append(label)
            sns.histplot(predictions[t0][model_name],
                         bins="sqrt",
                         #stat="density",
                         color=TFColor[model_idx],
                         #kde=True,
                         alpha=0.3,
                         ax=twin_ax)
            
        # chart formatting
        ax.set_title(f'{str(pct).upper()}th percentile')
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Observed probability")
        ax.tick_params(axis="y")
        #pred = pred_obs[value]['MCD'][0]
        #ax.plot(pred, pred, c="k", ls="--")
        ax.plot([0, 1], [0, 1], c="k", ls="--", transform=ax.transAxes)

    fig.tight_layout()
    plt.legend([label for label in labels], ["Cox", "RSF", "MLP", "VI", "MCD"])
    plt.show()