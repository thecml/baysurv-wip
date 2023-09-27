import numpy as np
import pandas as pd
from sksurv.linear_model.coxph import BreslowEstimator
import matplotlib.pyplot as plt
from lifelines.utils import CensoringType
from lifelines.fitters import RegressionFitter
from lifelines import CRCSplineFitter
import warnings

def survival_probability_calibration(model: RegressionFitter, df: pd.DataFrame,
                                     event_times: list, t, e,
                                     t0: float, ax=None):
    def ccl(p):
        return np.log(-np.log(1 - p))

    if ax is None:
        ax = plt.gca()

    T = "Survival_time"
    E = "Event"
    
    test_surv_fn = model.predict_survival_function(df)
    surv_preds = pd.DataFrame(np.row_stack([fn(event_times) for fn in test_surv_fn]), columns=event_times)
    predictions_at_t0 = np.clip(1 - surv_preds[t0].squeeze(), 1e-10, 1 - 1e-10)

    # create new dataset with the predictions
    prediction_df = pd.DataFrame({"ccl_at_%d" % t0: ccl(predictions_at_t0), T: t, E: e})

    # fit new dataset to flexible spline model
    # this new model connects prediction probabilities and actual survival. It should be very flexible, almost to the point of overfitting. It's goal is just to smooth out the data!
    knots = 3
    regressors = {"beta_": ["ccl_at_%d" % t0], "gamma0_": "1", "gamma1_": "1", "gamma2_": "1"}

    # this model is from examples/royson_crowther_clements_splines.py
    crc = CRCSplineFitter(knots, penalizer=0.000001)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        crc.fit_right_censoring(prediction_df, T, E, regressors=regressors) # only support right-censoring for now
        '''
        if CensoringType.is_right_censoring(model):
            crc.fit_right_censoring(prediction_df, T, E, regressors=regressors)
        elif CensoringType.is_left_censoring(model):
            crc.fit_left_censoring(prediction_df, T, E, regressors=regressors)
        elif CensoringType.is_interval_censoring(model):
            crc.fit_interval_censoring(prediction_df, T, E, regressors=regressors)
        '''

    # predict new model at values 0 to 1, but remember to ccl it!
    x = np.linspace(np.clip(predictions_at_t0.min() - 0.01, 0, 1), np.clip(predictions_at_t0.max() + 0.01, 0, 1), 100)
    y = 1 - crc.predict_survival_function(pd.DataFrame({"ccl_at_%d" % t0: ccl(x)}), times=[t0]).T.squeeze()

    # plot our results
    ax.set_title("Smoothed calibration curve of \npredicted vs observed probabilities of t ≤ %d mortality" % t0)

    color = "tab:red"
    ax.plot(x, y, label="smoothed calibration curve", color=color)
    ax.set_xlabel("Predicted probability of \nt ≤ %d mortality" % t0)
    ax.set_ylabel("Observed probability of \nt ≤ %d mortality" % t0, color=color)
    ax.tick_params(axis="y", labelcolor=color)

    # plot x=y line
    ax.plot(x, x, c="k", ls="--")
    ax.legend()

    # plot histogram of our original predictions
    color = "tab:blue"
    twin_ax = ax.twinx()
    twin_ax.set_ylabel("Count of \npredicted probabilities", color=color)  # we already handled the x-label with ax1
    twin_ax.tick_params(axis="y", labelcolor=color)
    twin_ax.hist(predictions_at_t0, alpha=0.3, bins="sqrt", color=color)

    plt.tight_layout()

    deltas = ((1 - crc.predict_survival_function(prediction_df, times=[t0])).T.squeeze() - predictions_at_t0).abs()
    ICI = deltas.mean()
    E50 = np.percentile(deltas, 50)
    print("ICI = ", ICI)
    print("E50 = ", E50)

    return ax, ICI, E50

def compute_survival_scale(risk_scores, t_train, e_train):
    # https://pubmed.ncbi.nlm.nih.gov/15724232/
    rnd = np.random.RandomState()

    # generate hazard scale
    mean_survival_time = t_train[e_train].mean()
    baseline_hazard = 1. / mean_survival_time
    scale = baseline_hazard * np.exp(risk_scores)
    return scale

def compute_survival_times(risk_scores, t_train, e_train):
    # https://pubmed.ncbi.nlm.nih.gov/15724232/
    rnd = np.random.RandomState(0)
        
    # generate survival time
    mean_survival_time = t_train[e_train].mean()
    baseline_hazard = 1. / mean_survival_time
    scale = baseline_hazard * np.exp(risk_scores)
    u = rnd.uniform(low=0, high=1, size=risk_scores.shape[0])
    t = -np.log(u) / scale

    return t

def compute_survival_times_with_censoring(risk_scores, t_train, e_train):
    # https://pubmed.ncbi.nlm.nih.gov/15724232/
    rnd = np.random.RandomState(0)

    # generate survival time
    mean_survival_time = t_train[e_train].mean()
    baseline_hazard = 1. / mean_survival_time
    scale = baseline_hazard * np.exp(risk_scores)
    u = rnd.uniform(low=0, high=1, size=risk_scores.shape[0])
    t = -np.log(u) / scale

    # generate time of censoring
    prob_censored = 1 - e_train.sum()/len(e_train)
    qt = np.quantile(t, 1.0 - prob_censored)
    c = rnd.uniform(low=t.min(), high=qt)

    # apply censoring
    observed_event = t <= c
    observed_time = np.where(observed_event, t, c)
    return observed_time, observed_event

def convert_to_structured(T, E):
    # dtypes for conversion
    default_dtypes = {"names": ("event", "time"), "formats": ("bool", "f8")}

    # concat of events and times
    concat = list(zip(E, T))

    # return structured array
    return np.array(concat, dtype=default_dtypes)

def get_breslow_survival_times(model, X_train, X_test, e_train, t_train, runs):
    train_predictions = model.predict(X_train, verbose=0).reshape(-1)
    breslow = BreslowEstimator().fit(train_predictions, e_train, t_train)
    model_cpd = np.zeros((runs, len(X_test)))
    for i in range(0, runs):
        model_cpd[i,:] = np.reshape(model.predict(X_test, verbose=0), len(X_test))
    event_times = breslow.get_survival_function(model_cpd[0])[0].x
    breslow_surv_times = np.zeros((len(X_test), runs, len(event_times)))
    for i in range(0, runs):
        surv_fns = breslow.get_survival_function(model_cpd[i,:])
        for j, surv_fn in enumerate(surv_fns):
            breslow_surv_times[j,i,:] = surv_fn.y
    return breslow_surv_times