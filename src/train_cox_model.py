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

    y_train_struc = convert_to_structured(t_train, e_train) # for convience
    y_valid_struc = convert_to_structured(t_valid, e_valid)

    # For nhanes only
    if DATASET == 'NHANES':
        y_train = convert_to_structured(y_train, np.ones(len(y_train)))
        y_test = convert_to_structured(y_test, np.ones(len(y_test)))

    model = make_cox_model()
    model.fit(X_train, y_train) # test
    predictions = model.predict(X_valid)

    print(model.coef_)
    print(predictions)

    # Calculate C-index
    c_index = concordance_index_censored(y_test["Status"], y_test["Survival_in_days"], predictions)[0] # veteran
    print(c_index)
    
    # Calculate Brier score
    lower, upper = np.percentile(t_valid[t_valid.dtype.names], [10, 90])
    times = np.arange(lower, upper+1)
    survs = model.predict_survival_function(X_valid)
    preds = np.asarray([[fn(t) for t in times] for fn in survs])
    ibs = integrated_brier_score(y_train_struc, y_valid_struc, preds, times)
    print(ibs)
    
    #result = concordance_index_censored(y_test["cens"], y_test["time"], prediction) # cancer
    #result = concordance_index_censored(y_test["censor"], y_test["time"], prediction) # aids
    #result = concordance_index_censored(y_test["censor"], y_test["time"], prediction) # nhanes

    # Save model
    curr_dir = os.getcwd()
    root_dir = Path(curr_dir).absolute()
    joblib.dump(model, f'{root_dir}/models/cox.joblib')