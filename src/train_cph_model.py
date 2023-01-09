import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from neural_risk_model import TrainAndEvaluateModel, Predictor
from utility import InputFunction, CindexMetric, CoxPHLoss
from sklearn.model_selection import train_test_split
from sksurv.linear_model.coxph import BreslowEstimator
import tensorflow_probability as tfp
from data_loader import load_veterans_ds, load_cancer_ds, load_aids_ds
from sksurv.metrics import concordance_index_censored
from sksurv.linear_model import CoxPHSurvivalAnalysis

if __name__ == "__main__":
    X_train, _, X_test, y_train, _, y_test = load_cancer_ds()
        
    estimator = CoxPHSurvivalAnalysis()
    estimator.fit(X_train, y_train)    
    prediction = estimator.predict(X_test)
    
    #result = concordance_index_censored(y_test["Status"], y_test["Survival_in_days"], prediction)
    result = concordance_index_censored(y_test["cens"], y_test["time"], prediction)
    #result = concordance_index_censored(y_test["censor"], y_test["time"], prediction)
    
    print(result[0])