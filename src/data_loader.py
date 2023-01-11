import numpy as np
import pandas as pd
from sksurv.datasets import load_veterans_lung_cancer, load_gbsg2, load_aids
from sksurv.preprocessing import OneHotEncoder, encode_categorical
from sklearn.model_selection import train_test_split
import shap

def split_data(X, y):
    X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.8, random_state=0)
    X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=0.5, random_state=0)
    return X_train, X_valid, X_test, y_train, y_valid, y_test

def load_veterans_ds():
    data_x, data_y = load_veterans_lung_cancer()
    data_x_numeric = encode_categorical(data_x)
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(data_x_numeric, data_y)
    return X_train, X_valid, X_test, y_train, y_valid, y_test

def load_cancer_ds():
    gbsg_X, gbsg_y = load_gbsg2()
    gbsg_X_numeric = encode_categorical(gbsg_X)
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(gbsg_X_numeric, gbsg_y)
    return X_train, X_valid, X_test, y_train, y_valid, y_test

def load_aids_ds():
    aids_X, aids_y = load_aids()
    aids_X_numeric = encode_categorical(aids_X)
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(aids_X_numeric, aids_y)
    return X_train, X_valid, X_test, y_train, y_valid, y_test

def load_nhanes_ds():
    nhanes_X, nhanes_y = shap.datasets.nhanesi()
    nhanes_X = nhanes_X.dropna(axis=1)
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(nhanes_X, nhanes_y)
    return X_train, X_valid, X_test, y_train, y_valid, y_test

def prepare_nhanes_ds(y_train, y_valid, y_test):
    t_train = np.array(y_train)
    t_valid = np.array(y_valid)
    t_test = np.array(y_test)
    e_train = np.ones(len(y_train)) # all observed
    e_valid = np.ones(len(y_valid))
    e_test = np.ones(len(y_test))
    return t_train, t_valid, t_test, e_train, e_valid, e_test

def prepare_veterans_ds(y_train, y_valid, y_test):
    t_train = np.array(y_train['Survival_in_days'])
    t_valid = np.array(y_valid['Survival_in_days'])
    t_test = np.array(y_test['Survival_in_days'])
    e_train = np.array(y_train['Status'])
    e_valid = np.array(y_valid['Status'])
    e_test = np.array(y_test['Status'])
    return t_train, t_valid, t_test, e_train, e_valid, e_test

def prepare_cancer_ds(y_train, y_valid, y_test):
    t_train = np.array(y_train['time'])
    t_valid = np.array(y_valid['time'])
    t_test = np.array(y_test['time'])
    e_train = np.array(y_train['cens'])
    e_valid = np.array(y_valid['cens'])
    e_test = np.array(y_test['cens'])
    return t_train, t_valid, t_test, e_train, e_valid, e_test

def prepare_aids_ds(y_train, y_valid, y_test):
    t_train = np.array(y_train['time'])
    t_valid = np.array(y_valid['time'])
    t_test = np.array(y_test['time'])
    e_train = np.array(y_train['censor'])
    e_valid = np.array(y_valid['censor'])
    e_test = np.array(y_test['censor'])
    return t_train, t_valid, t_test, e_train, e_valid, e_test