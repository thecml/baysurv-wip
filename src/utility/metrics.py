import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from typing import Tuple, Dict, Iterable, Optional, Sequence
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw
from sksurv.metrics import integrated_brier_score
from utility.survival import convert_to_structured
from utility.survival import compute_survival_times
from sksurv.linear_model.coxph import BreslowEstimator
import pandas as pd
from pycox.evaluation import EvalSurv

class IbsMetric:
    """Computes integrated brier score across one epoch."""
    def __init__(self, event_times) -> None:
        self._event_times = event_times
        self._data = {
            "y_train": [],
            "y_test": [],
            "pred_train": [],
            "pred_test": [],
        }

    def reset_states(self) -> None:
        """Clear the buffer of collected values."""
        self._data = {
            "y_train": [],
            "y_test": [],
            "pred_train": [],
            "pred_test": [],
        }
        
    def update_train_state(self, y_train) -> None:
        self._data["y_train"].append(y_train)
        
    def update_test_state(self, y_test) -> None:
        self._data["y_test"].append(y_test)
    
    def update_train_pred(self, y_pred: tf.Tensor) -> None:
        self._data["pred_train"].append(tf.squeeze(y_pred).numpy())
    
    def update_test_pred(self, y_pred: tf.Tensor) -> None:
        self._data["pred_test"].append(tf.squeeze(y_pred).numpy())

    def result(self) -> Dict[str, float]:
        """Computes integrated brier score across collected values.

        Returns
        ----------
        metrics : dict
            Computed metrics.
        """
        data = {}
        for k, v in self._data.items():
            if len(v) > 0:
                data[k] = np.concatenate(v)
        
        t_train = data["y_train"]['time']
        e_train = data["y_train"]['event']
        if "pred_test" not in data: # no test preds
            train_predictions = data['pred_train'].reshape(-1)
            breslow = BreslowEstimator().fit(train_predictions, e_train, t_train)
            test_predictions = data['pred_train'].reshape(-1)
            test_surv_fn = breslow.get_survival_function(test_predictions)
            surv_preds = np.row_stack([fn(self._event_times) for fn in test_surv_fn])
            surv_test = pd.DataFrame(surv_preds, columns=self._event_times)
            ev = EvalSurv(surv_test.T, data["y_train"]["time"], data["y_train"]["event"], censor_surv="km")
            ibs = ev.integrated_brier_score(self._event_times)
        else:
            train_predictions = data['pred_train'].reshape(-1)
            breslow = BreslowEstimator().fit(train_predictions, e_train, t_train)
            test_predictions = data['pred_test'].reshape(-1)
            test_surv_fn = breslow.get_survival_function(test_predictions)
            surv_preds = np.row_stack([fn(self._event_times) for fn in test_surv_fn])
            surv_test = pd.DataFrame(surv_preds, columns=self._event_times)
            ev = EvalSurv(surv_test.T, data["y_test"]["time"], data["y_test"]["event"], censor_surv="km")
            ibs = ev.integrated_brier_score(self._event_times)
        return ibs

class CindexTdMetric:
    """Computes time-dependent concordance index across one epoch."""
    def __init__(self, event_times) -> None:
        self._event_times = event_times
        self._data = {
            "y_train": [],
            "y_test": [],
            "pred_train": [],
            "pred_test": [],
        }

    def reset_states(self) -> None:
        """Clear the buffer of collected values."""
        self._data = {
            "y_train": [],
            "y_test": [],
            "pred_train": [],
            "pred_test": [],
        }
        
    def update_train_state(self, y_train) -> None:
        self._data["y_train"].append(y_train)
        
    def update_test_state(self, y_test) -> None:
        self._data["y_test"].append(y_test)
    
    def update_train_pred(self, y_pred: tf.Tensor) -> None:
        self._data["pred_train"].append(tf.squeeze(y_pred).numpy())
    
    def update_test_pred(self, y_pred: tf.Tensor) -> None:
        self._data["pred_test"].append(tf.squeeze(y_pred).numpy())

    def result(self) -> Dict[str, float]:
        """Computes the concordance index across collected values.

        Returns
        ----------
        metrics : dict
            Computed metrics.
        """
        data = {}
        for k, v in self._data.items():
            if len(v) > 0:
                data[k] = np.concatenate(v)
        
        t_train = data["y_train"]['time']
        e_train = data["y_train"]['event']
        if "pred_test" not in data: # no test preds
            train_predictions = data['pred_train'].reshape(-1)
            breslow = BreslowEstimator().fit(train_predictions, e_train, t_train)
            test_predictions = data['pred_train'].reshape(-1)
            test_surv_fn = breslow.get_survival_function(test_predictions)
            surv_preds = pd.DataFrame(np.row_stack([fn(self._event_times) for fn in test_surv_fn]),
                                      columns=self._event_times)
            ev = EvalSurv(surv_preds.T, t_train, e_train, censor_surv="km")
            ctd = ev.concordance_td()
        else:
            train_predictions = data['pred_train'].reshape(-1)
            breslow = BreslowEstimator().fit(train_predictions, e_train, t_train)
            test_predictions = data['pred_test'].reshape(-1)
            test_surv_fn = breslow.get_survival_function(test_predictions)
            surv_preds = pd.DataFrame(np.row_stack([fn(self._event_times) for fn in test_surv_fn]),
                                      columns=self._event_times)
            t_test = data["y_test"]['time']
            e_test = data["y_test"]['event']
            ev = EvalSurv(surv_preds.T, t_test, e_test, censor_surv="km")
            ctd = ev.concordance_td()
        return ctd
    
class InbllMetric:
    """Computes integrated negative binomial log-likelihood across one epoch."""
    def __init__(self, event_times) -> None:
        self._event_times = event_times
        self._data = {
            "y_train": [],
            "y_test": [],
            "pred_train": [],
            "pred_test": [],
        }

    def reset_states(self) -> None:
        """Clear the buffer of collected values."""
        self._data = {
            "y_train": [],
            "y_test": [],
            "pred_train": [],
            "pred_test": [],
        }
        
    def update_train_state(self, y_train) -> None:
        self._data["y_train"].append(y_train)
        
    def update_test_state(self, y_test) -> None:
        self._data["y_test"].append(y_test)
    
    def update_train_pred(self, y_pred: tf.Tensor) -> None:
        self._data["pred_train"].append(tf.squeeze(y_pred).numpy())
    
    def update_test_pred(self, y_pred: tf.Tensor) -> None:
        self._data["pred_test"].append(tf.squeeze(y_pred).numpy())

    def result(self) -> Dict[str, float]:
        """Computes the integrated negative binomial log-likelihood across collected values.

        Returns
        ----------
        metrics : dict
            Computed metrics.
        """
        data = {}
        for k, v in self._data.items():
            if len(v) > 0:
                data[k] = np.concatenate(v)
        
        t_train = data["y_train"]['time']
        e_train = data["y_train"]['event']
        if "pred_test" not in data: # no test preds
            train_predictions = data['pred_train'].reshape(-1)
            breslow = BreslowEstimator().fit(train_predictions, e_train, t_train)
            test_predictions = data['pred_train'].reshape(-1)
            test_surv_fn = breslow.get_survival_function(test_predictions)
            surv_preds = pd.DataFrame(np.row_stack([fn(self._event_times) for fn in test_surv_fn]),
                                      columns=self._event_times)
            ev = EvalSurv(surv_preds.T, t_train, e_train, censor_surv="km")
            inbll = ev.integrated_nbll(self._event_times)
        else:
            train_predictions = data['pred_train'].reshape(-1)
            breslow = BreslowEstimator().fit(train_predictions, e_train, t_train)
            test_predictions = data['pred_test'].reshape(-1)
            test_surv_fn = breslow.get_survival_function(test_predictions)
            surv_preds = pd.DataFrame(np.row_stack([fn(self._event_times) for fn in test_surv_fn]),
                                      columns=self._event_times)
            t_test = data["y_test"]['time']
            e_test = data["y_test"]['event']
            ev = EvalSurv(surv_preds.T, t_test, e_test, censor_surv="km")
            inbll = ev.integrated_nbll(self._event_times)
        return inbll
    
class CindexMetric:
    """Computes concordance index across one epoch."""
    def __init__(self) -> None:
        self._data = {
            "label_time": [],
            "label_event": [],
            "prediction": []
        }

    def reset_states(self) -> None:
        """Clear the buffer of collected values."""
        self._data = {
            "label_time": [],
            "label_event": [],
            "prediction": []
        }

    def update_state(self, y_true: Dict[str, tf.Tensor], y_pred: tf.Tensor) -> None:
        """Collect observed time, event indicator and predictions for a batch.

        Parameters
        ----------
        y_true : dict
            Must have two items:
            `label_time`, a tensor containing observed time for one batch,
            and `label_event`, a tensor containing event indicator for one batch.
        y_pred : tf.Tensor
            Tensor containing predicted risk score for one batch.
        """
        self._data["label_time"].append(y_true["label_time"].numpy())
        self._data["label_event"].append(y_true["label_event"].numpy())
        self._data["prediction"].append(tf.squeeze(y_pred).numpy())

    def result(self) -> Dict[str, float]:
        """Computes the concordance index across collected values.

        Returns
        ----------
        metrics : dict
            Computed metrics.
        """
        data = {}
        for k, v in self._data.items():
            data[k] = np.concatenate(v)

        results = concordance_index_censored(
            data["label_event"] == 1,
            data["label_time"],
            data["prediction"])

        result_data = {}
        names = ("cindex", "concordant", "discordant", "tied_risk")
        for k, v in zip(names, results):
            result_data[k] = v

        return result_data