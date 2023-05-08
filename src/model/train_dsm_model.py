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
    model = SurvivalModel('dsm', random_seed=0, iters=N_ITER,
                          layers=[32, 32], distribution='Weibull')
    
    # Fit model
    model.fit(X_train, pd.DataFrame(y_train))
    
    # Evaluate risk
    risk_pred = model.predict_risk(X_test, times=y_train['time'].max()).flatten()
    
    # Evaluate surv prob
    t_train = y_train['time']
    e_train = y_train['event']
    t_test = y_test['time']
    train_predictions = model.predict_risk(X_train, times).reshape(-1)
    breslow = BreslowEstimator().fit(train_predictions, e_train, t_train)
    test_predictions = model.predict_risk(X_test, times).reshape(-1)
    test_surv_fn = breslow.get_survival_function(test_predictions)
    surv_preds = np.row_stack([fn(times) for fn in test_surv_fn])
    
    ci = concordance_index_censored(y_test['event'], y_test['time'], risk_pred)[0]
    ctd = concordance_index_ipcw(y_train, y_test, risk_pred)[0]
    ibs = integrated_brier_score(y_train, y_test, surv_preds, list(times))
    
    print(ci)
    print(ctd)
    print(ibs)
