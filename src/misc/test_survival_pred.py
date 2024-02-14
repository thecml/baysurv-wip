

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
from utility.training import get_data_loader, scale_data, split_time_event
from utility.survival import calculate_event_times, predict_median_survival_time, predict_mean_survival_time
from utility.config import load_config
import paths as pt
from utility.model import load_mlp_model
from utility.survival import compute_survival_function

DATASET_NAME = "SEER"

if __name__ == "__main__":
    # Load data
    dl = get_data_loader(DATASET_NAME).load_data()
    X, y = dl.get_data()
    num_features, cat_features = dl.get_features()

    # Split data in train, valid and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    X_train, X_valid, y_train, y_valid  = train_test_split(X_train, y_train, test_size=0.25, random_state=0)

    # Scale data
    X_train, X_valid, X_test = scale_data(X_train, X_valid, X_test, cat_features, num_features)

    # Make time/event split
    t_train, e_train = split_time_event(y_train)
    t_test, e_test = split_time_event(y_test)

    # Make event times
    event_times = calculate_event_times(t_train, e_train)

    # Load MLP model
    n_input_dims = X_train.shape[1:]
    model = load_mlp_model(DATASET_NAME, n_input_dims)

    # Select only test samples where event occurs
    test_idx = list(np.where(y_test['event'] == True)[0])
    X_test = X_test.iloc[test_idx]
    y_test = y_test[test_idx]
    
    # Compute surv func
    surv_preds = compute_survival_function(model, X_train, X_test, e_train, t_train, event_times)
    print(surv_preds)