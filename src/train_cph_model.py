import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from neural_risk_model import TrainAndEvaluateModel, Predictor
from utility import convert_to_structured
from sklearn.model_selection import train_test_split
from sksurv.linear_model.coxph import BreslowEstimator
import tensorflow_probability as tfp
from data_loader import load_cancer_ds, load_nhanes_ds
from sksurv.metrics import concordance_index_censored
from sksurv.linear_model import CoxPHSurvivalAnalysis

DATASET = 'NHANES'

if __name__ == "__main__":
    X_train, _, X_test, y_train, _, y_test = load_nhanes_ds()

    # For nhanes only
    if DATASET == 'NHANES':
        y_train = convert_to_structured(y_train, np.ones(len(y_train)))
        y_test = convert_to_structured(y_test, np.ones(len(y_test)))
        
    estimator = CoxPHSurvivalAnalysis(alpha=0.0001)
    estimator.fit(X_train, y_train)
    prediction = estimator.predict(X_test)

    #result = concordance_index_censored(y_test["Status"], y_test["Survival_in_days"], prediction) # veteran
    #result = concordance_index_censored(y_test["cens"], y_test["time"], prediction) # cancer
    #result = concordance_index_censored(y_test["censor"], y_test["time"], prediction) # aids
    result = concordance_index_censored(y_test["censor"], y_test["time"], prediction) # nhanes

    print(result[0])