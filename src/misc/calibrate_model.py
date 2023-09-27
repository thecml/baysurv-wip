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
from tools.model_builder import make_cox_model
from utility.survival import survival_probability_calibration

DATASET_NAME = "WHAS500"

if __name__ == "__main__":
    # Load data
    dl = get_data_loader(DATASET_NAME).load_data()
    X, y = dl.get_data()
    num_features, cat_features = dl.get_features()

    # Split data in train, valid and test set
    X_columns = X.columns
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7,
                                                        random_state=0)

    # Scale data
    X_train, X_test = scale_data(X_train, X_test, cat_features, num_features)
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    # Make time/event split
    t_train, e_train = make_time_event_split(y_train)
    t_test, e_test = make_time_event_split(y_test)

    # Make event times
    lower, upper = np.percentile(y['time'], [10, 90])
    event_times = np.arange(lower, upper+1)

    # Create Cox model
    cox_config = load_config(pt.COX_CONFIGS_DIR, f"{DATASET_NAME.lower()}.yaml")
    model = make_cox_model(cox_config)
    model.fit(X_train, y_train)

    # Compute surv func
    test_sample = X_test
    test_surv_fn = model.predict_survival_function(test_sample)
    surv_preds = np.row_stack([fn(event_times) for fn in test_surv_fn])
    
    from lifelines import CoxPHFitter
    cph = CoxPHFitter(penalizer=0.0001)
    data = pd.concat([pd.DataFrame(X_train),
                    pd.Series(y_train['time'], name="Survival_time"),
                    pd.Series(y_train['event'], name="Event")], axis=1)
    cph.fit(data, duration_col="Survival_time", event_col="Event")
    
    # Plot calibration
    survival_probability_calibration(model, X_test, event_times, t_test, e_test, t0=25)
    plt.show()
    
    
