from sksurv.ensemble import GradientBoostingSurvivalAnalysis
import numpy as np
import os
from pathlib import Path
from utility.tuning import get_coxboost_sweep_config
import argparse
from tools import data_loader
from sklearn.model_selection import train_test_split, KFold
from tools.preprocessor import Preprocessor
from sksurv.metrics import concordance_index_censored
import pandas as pd
from pycox.evaluation import EvalSurv

os.environ["WANDB_SILENT"] = "true"
import wandb

N_RUNS = 10
N_SPLITS = 5
PROJECT_NAME = "baysurv_bo_coxboost"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        required=True,
                        default=None)
    args = parser.parse_args()
    global dataset
    if args.dataset:
        dataset = args.dataset

    sweep_config = get_coxboost_sweep_config()
    sweep_id = wandb.sweep(sweep_config, project=PROJECT_NAME)
    wandb.agent(sweep_id, train_model, count=N_RUNS)

def train_model():
    config_defaults = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'loss': 'coxph',
            'min_samples_split': 2,
            'max_features': None,
            'dropout_rate': 0.0,
            'subsample': 1.0,
            'seed': 0,
            'test_size': 0.3,
        }

    # Initialize a new wandb run
    wandb.init(config=config_defaults, group=dataset)
    config = wandb.config

    # Load data
    if dataset == "SUPPORT":
        dl = data_loader.SupportDataLoader().load_data()
    elif dataset == "GBSG2":
        dl = data_loader.GbsgDataLoader().load_data()
    elif dataset == "WHAS500":
        dl = data_loader.WhasDataLoader().load_data()
    elif dataset == "FLCHAIN":
        dl = data_loader.FlchainDataLoader().load_data()
    elif dataset == "METABRIC":
        dl = data_loader.MetabricDataLoader().load_data()
    elif dataset == "SEER":
        dl = data_loader.SeerDataLoader().load_data()
    else:
        raise ValueError("Dataset not found")

    num_features, cat_features = dl.get_features()
    X, y = dl.get_data()

    # Split data in T1 and HOS
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=0)
    T1, HOS = (X_train, y_train), (X_test, y_test)

    # Perform K-fold cross-validation
    c_indicies = list()
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=0)
    for train, test in kf.split(T1[0], T1[1]):
        ti_X = T1[0].iloc[train]
        ti_y = T1[1][train]
        cvi_X = T1[0].iloc[test]
        cvi_y = T1[1][test]

        # Scale data split
        preprocessor = Preprocessor(cat_feat_strat='mode', num_feat_strat='mean')
        transformer = preprocessor.fit(ti_X, cat_feats=cat_features, num_feats=num_features,
                                       one_hot=True, fill_value=-1)
        ti_X = np.array(transformer.transform(ti_X))
        cvi_X = np.array(transformer.transform(cvi_X))

        # Make model
        model = GradientBoostingSurvivalAnalysis(n_estimators=config.n_estimators,
                                                 learning_rate=config.learning_rate,
                                                 max_depth=config.max_depth,
                                                 loss=config.loss,
                                                 min_samples_split=config.min_samples_split,
                                                 max_features=config.max_features,
                                                 dropout_rate=config.dropout_rate,
                                                 random_state=0)
        
        # Fit model
        model.fit(ti_X, ti_y)
        
        # Compute survival function
        lower, upper = np.percentile(y['time'], [10, 90])
        times = np.arange(lower, upper+1)
        test_surv_fn = model.predict_survival_function(X_test)
        surv_preds = np.row_stack([fn(times) for fn in test_surv_fn])

        # Compute CTD
        surv_test = pd.DataFrame(surv_preds, columns=times)
        ev = EvalSurv(surv_test.T, cvi_y["time"], cvi_y["event"], censor_surv="km")
        ctd = ev.concordance_td()
        c_indicies.append(ctd)

    mean_ci = np.nanmean(c_indicies)

    # Log to wandb
    wandb.log({"val_ci": mean_ci})

if __name__ == "__main__":
    main()

