import tensorflow as tf
import numpy as np
import pandas as pd
import random

from sklearn.model_selection import train_test_split
matplotlib_style = 'fivethirtyeight'
import matplotlib.pyplot as plt; plt.style.use(matplotlib_style)

from tools.model_trainer import Trainer
from utility.config import load_config
from utility.training import get_data_loader, scale_data, make_time_event_split
from utility.plotter import plot_training_curves
from tools.model_builder import make_mlp_model, make_vi_model, make_mcd_model, make_mlp_model
from utility.risk import InputFunction
from utility.loss import CoxPHLoss
from pathlib import Path
import paths as pt
import os
from tools.preprocessor import Preprocessor
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw

from auton_survival.estimators import SurvivalModel
from auton_survival.metrics import survival_regression_metric
from sklearn.model_selection import ParameterGrid
import pandas as pd

N_EPOCHS = 5
BATCH_SIZE = 32

if __name__ == "__main__":
    # Load data
    dataset_name = "WHAS500"
    dl = get_data_loader(dataset_name).load_data()
    X, y = dl.get_data()
    num_features, cat_features = dl.get_features()
    
    # Split data in train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=0)

    # Scale data
    preprocessor = Preprocessor(cat_feat_strat='mode', num_feat_strat='mean')
    transformer = preprocessor.fit(X_train, cat_feats=cat_features, num_feats=num_features,
                                   one_hot=True, fill_value=-1)
    X_train = transformer.transform(X_train)
    X_test = transformer.transform(X_test)
        
    # Define the times
    times = np.quantile(y_train['time'][y_train['event']==1], np.linspace(0.1, 1, 10)).tolist()
    
    # Make model
    model = SurvivalModel('dsm', random_seed=0, iters=N_EPOCHS,
                          layers=[32, 32], distribution='Weibull', max_features='sqrt')
    
    # Fit model
    model.fit(X_train, pd.DataFrame(y_train))
    
    # Evaluate
    predictions = model.predict_survival(X_test, times)
    predictions_test = predictions
    
    te_min, te_max = y_test['time'].min(), y_test['time'].max()
    unique_time_mask = (times>te_min)&(times<te_max)
    times = np.array(times)[unique_time_mask]
    predictions_test = predictions_test[:, unique_time_mask]
    
    ctd = concordance_index_ipcw(y_train, y_test,
                                 1-predictions_test, tau=times)[0]
    

    #model.predict_risk(X_test, times=y_test['time'].max())

    #ci = concordance_index_censored(y_test["event"], y_test["time"], predictions)[0]
    #print(ci)
    
