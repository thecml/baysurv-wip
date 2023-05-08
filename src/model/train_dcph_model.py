import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
matplotlib_style = 'fivethirtyeight'
import matplotlib.pyplot as plt; plt.style.use(matplotlib_style)

from utility.training import get_data_loader
from tools.preprocessor import Preprocessor
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw, integrated_brier_score
from sksurv.linear_model.coxph import BreslowEstimator
from auton_survival.estimators import SurvivalModel
import pandas as pd
from utility.training import make_time_event_split
from auton_survival import DeepCoxPH

N_ITER = 10

if __name__ == "__main__":
    # Load data
    dataset_name = "FLCHAIN"
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
    lower, upper = np.percentile(y_train['time'][y_train['time'].dtype.names], [10, 90])
    times = np.arange(lower, upper+1)
    
    # Make model
    model = DeepCoxPH(layers=[100])
    
    # Make time/event split
    t_train, e_train = make_time_event_split(y_train)
    t_test, e_test = make_time_event_split(y_test)
    
    # Fit model
    model.fit(np.array(X_train), t_train, e_train,
              iters=N_ITER, vsize=0.15, learning_rate=0.001,
              optimizer="Adam", random_state=0)

    # Evaluate risk
    risk_pred = model.predict_risk(np.array(X_test), t=y_train['time'].max()).flatten()
    
    # Evaluate surv prob
    t_train = y_train['time']
    e_train = y_train['event']
    t_test = y_test['time']
    train_predictions = model.predict_risk(np.array(X_train), y_train['time'].max()).flatten()
    breslow = BreslowEstimator().fit(train_predictions, e_train, t_train)
    test_predictions = model.predict_risk(np.array(X_test), y_train['time'].max()).flatten()
    test_surv_fn = breslow.get_survival_function(test_predictions)
    surv_preds = np.row_stack([fn(times) for fn in test_surv_fn])
    
    ci = concordance_index_censored(y_test['event'], y_test['time'], risk_pred)[0]
    ctd = concordance_index_ipcw(y_train, y_test, risk_pred)[0]
    ibs = integrated_brier_score(y_train, y_test, surv_preds, list(times))
    
    print(ci)
    print(ctd)
    print(ibs)
