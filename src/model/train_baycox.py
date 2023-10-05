import numpy as np
import random

from sklearn.model_selection import train_test_split
matplotlib_style = 'fivethirtyeight'
import matplotlib.pyplot as plt; plt.style.use(matplotlib_style)

from utility.training import get_data_loader, scale_data, make_time_event_split, train_val_test_stratified_split
from utility.survival import make_time_bins, cox_survival
from utility.torch_models import BayesCox, BayesEleCox, BayesLinCox, BayesianBaseModel
import math
import torch
import torch.optim as optim
import torch.nn as nn
from typing import List, Tuple, Optional, Union
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
from tqdm import trange
from pycox.evaluation import EvalSurv

random.seed(0)

DATASETS = ["WHAS500"]
N_EPOCHS = 10

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
Numeric = Union[float, int, bool]
NumericArrayLike = Union[List[Numeric], Tuple[Numeric], np.ndarray, pd.Series, pd.DataFrame, torch.Tensor]

def train_model(
        model: nn.Module,
        data_train: pd.DataFrame,
        time_bins: NumericArrayLike,
        config: dotdict,
        random_state: int,
        reset_model: bool = True,
        device: torch.device = torch.device("cuda")
) -> nn.Module:
    if config.verbose:
        print(f"Training {model.get_name()}: reset mode is {reset_model}, number of epochs is {config.num_epochs}, "
              f"learning rate is {config.lr}, C1 is {config.c1}, "
              f"batch size is {config.batch_size}, device is {device}.")
    data_train, _, data_val = train_val_test_stratified_split(data_train, stratify_colname='both',
                                                              frac_train=0.9, frac_test=0.1,
                                                              random_state=random_state)

    train_size = data_train.shape[0]
    val_size = data_val.shape[0]
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    if reset_model:
        model.reset_parameters()

    model = model.to(device)
    model.train()
    best_val_nll = np.inf
    best_ep = -1

    pbar = trange(config.num_epochs, disable=not config.verbose)

    start_time = datetime.now()
    if isinstance(model, BayesEleCox) or isinstance(model, BayesLinCox):
        x_train, t_train, e_train = (torch.tensor(data_train.drop(["time", "event"], axis=1).values, dtype=torch.float),
                                     torch.tensor(data_train["time"].values, dtype=torch.float),
                                     torch.tensor(data_train["event"].values, dtype=torch.float))
        x_val, t_val, e_val = (torch.tensor(data_val.drop(["time", "event"], axis=1).values, dtype=torch.float).to(device),
                               torch.tensor(data_val["time"].values, dtype=torch.float).to(device),
                               torch.tensor(data_val["event"].values, dtype=torch.float).to(device))

        train_loader = DataLoader(TensorDataset(x_train, t_train, e_train), batch_size=train_size, shuffle=True)
        model.config.batch_size = train_size

        for i in pbar:
            total_loss = 0
            total_log_likelihood = 0
            total_kl_divergence = 0
            for xi, ti, ei in train_loader:
                xi, ti, ei = xi.to(device), ti.to(device), ei.to(device)
                optimizer.zero_grad()
                loss, log_prior, log_variational_posterior, log_likelihood = model.sample_elbo(xi, ti, ei, train_size)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_log_likelihood += log_likelihood.item()
                total_kl_divergence += log_variational_posterior.item() - log_prior.item()

            val_loss, _, _, val_log_likelihood = model.sample_elbo(x_val, t_val, e_val, dataset_size=val_size)
            pbar.set_description(f"[epoch {i + 1: 4}/{config.num_epochs}]")
            pbar.set_postfix_str(f"Train: Total = {total_loss:.4f}, "
                                 f"KL = {total_kl_divergence:.4f}, "
                                 f"nll = {total_log_likelihood:.4f}; "
                                 f"Val: Total = {val_loss.item():.4f}, "
                                 f"nll = {val_log_likelihood.item():.4f}; ")
            if config.early_stop:
                if best_val_nll > val_loss:
                    best_val_nll = val_loss
                    best_ep = i
                if (i - best_ep) > config.patience:
                    print(f"Validation loss converges at {best_ep}-th epoch.")
                    break
    else:
        raise TypeError("Model type cannot be identified.")
    end_time = datetime.now()
    training_time = end_time - start_time
    print(f"Training time: {training_time.total_seconds()}")
    # model.eval()
    if isinstance(model, BayesEleCox) or isinstance(model, BayesLinCox):
        model.calculate_baseline_survival(x_train.to(device), t_train.to(device), e_train.to(device))
    return model

def make_ensemble_cox_prediction(
        model: BayesianBaseModel,
        x: torch.Tensor,
        config: dotdict
) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    model.eval()
    start_time = datetime.now()
    with torch.no_grad():
        logits_outputs = model.forward(x, sample=True, n_samples=config.n_samples_test)
        end_time = datetime.now()
        inference_time = end_time - start_time
        print(f"Inference time: {inference_time.total_seconds()}")
        survival_outputs = cox_survival(model.baseline_survival, logits_outputs)
        mean_survival_outputs = survival_outputs.mean(dim=0)

    time_bins = model.time_bins
    return mean_survival_outputs, time_bins, survival_outputs

if __name__ == "__main__":
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    
    # Load data
    dataset_name = DATASETS[0]
    dl = get_data_loader(dataset_name).load_data()
    X, y = dl.get_data()
    num_features, cat_features = dl.get_features()

    # Split data in train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=0)
    
    # Scale data
    X_train, X_test = scale_data(X_train, X_test, cat_features, num_features)
    
    # Make dataframe                                                                
    data_train = X_train
    data_train["time"] = pd.Series(y_train['time'])
    data_train["event"] = pd.Series(y_train['event']).astype(int)
    
    data_test = X_test
    data_test["time"] = pd.Series(y_test['time'])
    data_test["event"] = pd.Series(y_test['event']).astype(int)
    
    # Make times
    lower, upper = np.percentile(y['time'], [10, 90])
    time_bins = np.arange(lower, upper+1)
    
    # Make model
    config = {'hidden_size': 32, 'mu_scale': None, 'rho_scale': -5,
              'sigma1': 1, 'sigma2': math.exp(-6), 'pi': 0.5,
              'verbose': True, 'lr': 0.005, 'num_epochs': 1000,
              'dropout': 0, 'n_samples_train': 10,
              'n_samples_test': 10, 'batch_size': 32,
              'early_stop': True, 'patience': 100}
    config=dotdict(config)
    num_features = X_train.shape[1] - 2
    model = BayesCox(in_features=num_features, config=config)
    
    # Train model
    model = train_model(model, data_train, time_bins, config=config, random_state=0, reset_model=True, device=device)
    
    # Test model
    X_test = torch.tensor(data_test.drop(["time", "event"], axis=1).values, dtype=torch.float, device=device)
    survival_outputs, time_bins, ensemble_outputs = make_ensemble_cox_prediction(model, X_test, config=config)
    surv_test = pd.DataFrame(survival_outputs.numpy(), columns=time_bins.numpy())
    ev = EvalSurv(surv_test.T, np.array(data_test["time"]), np.array(data_test["event"]), censor_surv="km")
    print("CTD:", ev.concordance_td())
    print("IBS:", ev.integrated_brier_score(np.array(time_bins)))
    
    
    