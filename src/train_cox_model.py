import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from model_builder import TrainAndEvaluateModel, Predictor, make_cox_model
from utility import convert_to_structured
from sklearn.model_selection import train_test_split
from sksurv.linear_model.coxph import BreslowEstimator
import tensorflow_probability as tfp
from data_loader import load_cancer_ds, load_nhanes_ds, load_veterans_ds, prepare_veterans_ds
from sksurv.metrics import concordance_index_censored
import joblib
import os
from pathlib import Path
from sksurv.metrics import integrated_brier_score

DATASET = 'VETERANS'

if __name__ == "__main__":
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_veterans_ds()
    t_train, t_valid, t_test, e_train, e_valid, e_test  = prepare_veterans_ds(y_train, y_valid, y_test)

    # For nhanes only
    if DATASET == 'NHANES':
        y_train = convert_to_structured(y_train, np.ones(len(y_train)))
        y_test = convert_to_structured(y_test, np.ones(len(y_test)))

    model = make_cox_model()
    model.fit(X_train, y_train)

    # Calculate scores
    predictions = model.predict(X_test)
    c_index = concordance_index_censored(y_test["Status"], y_test["Survival_in_days"], predictions)[0] # veteran    
    lower, upper = np.percentile(t_test[t_test.dtype.names], [10, 90])
    times = np.arange(lower, upper+1)
    survs = model.predict_survival_function(X_test)
    preds = np.asarray([[fn(t) for t in times] for fn in survs])
    
    y_train_struc = convert_to_structured(t_train, e_train)
    y_test_struc = convert_to_structured(t_test, e_test)
    
    ibs = integrated_brier_score(y_train_struc, y_test_struc, preds, times)
    print(f"Training completed, test C-index/BS: {round(c_index, 4)}/{round(ibs, 4)}")
    
    #result = concordance_index_censored(y_test["cens"], y_test["time"], prediction) # cancer
    #result = concordance_index_censored(y_test["censor"], y_test["time"], prediction) # aids
    #result = concordance_index_censored(y_test["censor"], y_test["time"], prediction) # nhanes

    # Save model
    curr_dir = os.getcwd()
    root_dir = Path(curr_dir).absolute()
    joblib.dump(model, f'{root_dir}/models/cox.joblib')