import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from typing import Tuple, Dict, Iterable, Optional, Sequence
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw
from sksurv.metrics import integrated_brier_score
from utility.survival import convert_to_structured
from utility.survival import compute_survival_times

class IbsMetric:
    """Computes concordance index across one epoch."""
    def __init__(self) -> None:
        self._data = {
            "y_train": [],
            "y_test": [],
            "prediction": []
        }

    def reset_states(self) -> None:
        """Clear the buffer of collected values."""
        self._data = {
            "y_train": [],
            "y_test": [],
            "prediction": []
        }
        
    def update_train_state(self, y_train) -> None:
        self._data["y_train"].append(y_train)
        
    def update_test_state(self, y_test) -> None:
        self._data["y_test"].append(y_test)
        
    def update_pred_state(self, y_pred: tf.Tensor) -> None:
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
        
        t_train = data["y_train"]['Time']
        e_train = data["y_train"]['Event']
        t_test = data["y_test"]['Time']
        lower, upper = np.percentile(t_test[t_test.dtype.names], [10, 90])
        times = np.arange(lower, upper+1)
        estimate = np.zeros((len(data["y_test"]), len(times)))
        surv_times = compute_survival_times(data["prediction"], t_train, e_train)
        for i, surv_time in enumerate(surv_times):
            surv_prob = np.exp(-times/surv_time)
            estimate[i] = surv_prob
            
        ibs = integrated_brier_score(data["y_train"], data["y_test"], estimate, times)
        
        return ibs

class CindexTdMetric:
    """Computes concordance index across one epoch."""
    def __init__(self) -> None:
        self._data = {
            "y_train": [],
            "y_test": [],
            "prediction": []
        }

    def reset_states(self) -> None:
        """Clear the buffer of collected values."""
        self._data = {
            "y_train": [],
            "y_test": [],
            "prediction": []
        }
        
    def update_train_state(self, y_train) -> None:
        self._data["y_train"].append(y_train)
        
    def update_test_state(self, y_test) -> None:
        self._data["y_test"].append(y_test)
        
    def update_pred_state(self, y_pred: tf.Tensor) -> None:
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

        results = concordance_index_ipcw(data["y_train"],
                                         data["y_test"],
                                         data["prediction"])

        result_data = {}
        names = ("cindex", "concordant", "discordant", "tied_risk", "tied_time")
        for k, v in zip(names, results):
            result_data[k] = v

        return result_data
    
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