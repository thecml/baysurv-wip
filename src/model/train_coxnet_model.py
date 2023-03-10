import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tools.model_builder import make_coxnet_model
from sklearn.model_selection import train_test_split
from sksurv.linear_model.coxph import BreslowEstimator
import tensorflow_probability as tfp
from sksurv.metrics import concordance_index_censored
import joblib
import os
from pathlib import Path
from sksurv.metrics import integrated_brier_score
from tools import data_loader
from sklearn.preprocessing import StandardScaler
from utility.survival import compute_survival_times, convert_to_structured
from tools.preprocessor import Preprocessor

if __name__ == "__main__":
    # Load data
    dl = data_loader.SupportDataLoader().load_data()
    X, y = dl.get_data()
    num_features, cat_features = dl.get_features()
    
    # Split data in train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=0)
    
    # Scale data
    preprocessor = Preprocessor(cat_feat_strat='ignore', num_feat_strat='mean')
    transformer = preprocessor.fit(X_train, cat_feats=cat_features, num_feats=num_features,
                                   one_hot=True, fill_value=-1)
    X_train = np.array(transformer.transform(X_train))
    X_test = np.array(transformer.transform(X_test))

    # Make time/event split
    t_train = np.array(y_train['Time'])
    e_train = np.array(y_train['Event'])
    t_test = np.array(y_test['Time'])
    e_test = np.array(y_test['Event'])

    model = make_coxnet_model()
    model.fit(X_train, y_train)

    # Calculate scores
    predictions = model.predict(X_test)
    c_index = concordance_index_censored(e_test, t_test, predictions)[0] # veteran    
    lower, upper = np.percentile(t_test[t_test.dtype.names], [10, 90])
    times = np.arange(lower, upper+1)
    survs = model.predict_survival_function(X_test)
    preds = np.asarray([[fn(t) for t in times] for fn in survs])
    
    y_train_struc = convert_to_structured(t_train, e_train)
    y_test_struc = convert_to_structured(t_test, e_test)
    
    ibs = integrated_brier_score(y_train_struc, y_test_struc, preds, times)
    print(f"Training completed, test C-index/IBS: {round(c_index, 4)}/{round(ibs, 4)}")