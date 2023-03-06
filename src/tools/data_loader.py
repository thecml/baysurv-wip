import numpy as np
import pandas as pd
from sksurv.datasets import load_veterans_lung_cancer, load_gbsg2, load_aids, load_whas500, load_flchain
from sklearn.model_selection import train_test_split
from auton_survival import datasets
import shap
from abc import ABC, abstractmethod
from typing import Tuple, List
from tools.preprocessor import Preprocessor
import paths as pt
from pathlib import Path
from pycox import datasets
from utility.survival import convert_to_structured

class BaseDataLoader(ABC):
    """
    Base class for data loaders.
    """
    def __init__(self):
        """Initilizer method that takes a file path, file name,
        settings and optionally a converter"""
        self.X: pd.DataFrame = None
        self.y: np.ndarray = None
        self.num_features: List[str] = None
        self.cat_features: List[str] = None

    @abstractmethod
    def load_data(self) -> None:
        """Loads the data from a data set at startup"""

    @abstractmethod
    def make_time_event_split(self, y_train, y_valid, y_test) -> None:
        """Makes time/event split of y"""

    def get_data(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        This method returns the features and targets
        :return: X and y
        """
        return self.X, self.y

    def get_features(self) -> List[str]:
        """
        This method returns the names of numerical and categorial features
        :return: the columns of X as a list
        """
        return self.num_features, self.cat_features

    def _get_num_features(self, data) -> List[str]:
        return data.select_dtypes(include=np.number).columns.tolist()
    
    def _get_cat_features(self, data) -> List[str]:
        return data.select_dtypes(['category']).columns.tolist() \
            + data.select_dtypes(['object']).columns.tolist()

    def prepare_data(self, train_size: float = 0.7) -> Tuple[np.ndarray, np.ndarray,
                                                             np.ndarray, np.ndarray]:
        """
        This method prepares and splits the data from a data set
        :param train_size: the size of the train set
        :return: a split train and test dataset
        """
        X = self.X
        y = self.y
        cat_features = self.cat_features
        num_features = self.num_features

        X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=train_size, random_state=0)
        X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=0.5, random_state=0)

        preprocessor = Preprocessor(cat_feat_strat='ignore', num_feat_strat='mean')
        transformer = preprocessor.fit(X_train, cat_feats=cat_features, num_feats=num_features,
                                       one_hot=True, fill_value=-1)
        X_train = transformer.transform(X_train)
        X_valid = transformer.transform(X_valid)
        X_test = transformer.transform(X_test)

        X_train = np.array(X_train, dtype=np.float32)
        X_valid = np.array(X_valid, dtype=np.float32)
        X_test = np.array(X_test, dtype=np.float32)

        return X_train, X_valid, X_test, y_train, y_valid, y_test

class SupportDataLoader(BaseDataLoader):
    """
    Data loader for SUPPORT dataset
    """
    def load_data(self):
        data = datasets.support.read_df()
        
        outcomes = data.copy()
        outcomes['event'] =  data['event']
        outcomes['time'] = data['duration']
        outcomes = outcomes[['event', 'time']]
        
        num_feats =  ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6',
                      'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13']
        
        self.num_features = num_feats
        self.cat_features = []
        self.X = pd.DataFrame(data[num_feats])
        self.y = convert_to_structured(outcomes['time'], outcomes['event'])
        
        return self

    def make_time_event_split(self, y_train, y_valid, y_test) -> None:
        t_train = np.array(y_train[:,1])
        t_valid = np.array(y_valid[:,1])
        t_test = np.array(y_test[:,1])
        e_train = np.array(y_train[:,0])
        e_valid = np.array(y_valid[:,0])
        e_test = np.array(y_test[:,0])
        return t_train, t_valid, t_test, e_train, e_valid, e_test

class NhanesDataLoader(BaseDataLoader):
    """
    Data loader for NHANES dataset
    """
    def load_data(self):
        nhanes_X, nhanes_y = shap.datasets.nhanesi()
        self.X = pd.DataFrame(nhanes_X)
        
        event = np.array([True if x > 0 else False for x in nhanes_y])
        time = np.array(abs(nhanes_y))
        self.y = convert_to_structured(time, event)
        
        self.num_features = self._get_num_features(self.X)
        self.cat_features = self._get_cat_features(self.X)
        return self

    def make_time_event_split(self, y_train, y_valid, y_test) -> None:
        t_train = np.array(y_train)
        t_valid = np.array(y_valid)
        t_test = np.array(y_test)
        e_train = np.array([True if x > 0 else False for x in y_train])
        e_valid = np.array([True if x > 0 else False for x in y_valid])
        e_test = np.array([True if x > 0 else False for x in y_test])
        return t_train, t_valid, t_test, e_train, e_valid, e_test

class AidsDataLoader(BaseDataLoader):
    def load_data(self) -> None:
        aids_X, aids_y = load_aids()
        self.X = aids_X
        self.y = convert_to_structured(aids_y['time'], aids_y['censor'])
        self.num_features = self._get_num_features(self.X)
        self.cat_features = self._get_cat_features(self.X)
        return self

    def make_time_event_split(self, y_train, y_valid, y_test) -> None:
        t_train = np.array(y_train['time'])
        t_valid = np.array(y_valid['time'])
        t_test = np.array(y_test['time'])
        e_train = np.array(y_train['censor'])
        e_valid = np.array(y_valid['censor'])
        e_test = np.array(y_test['censor'])
        return t_train, t_valid, t_test, e_train, e_valid, e_test

class GbsgDataLoader(BaseDataLoader):
    def load_data(self) -> BaseDataLoader:
        gbsg_X, gbsg_y = load_gbsg2()
        self.X = gbsg_X
        self.y = convert_to_structured(gbsg_y['time'], gbsg_y['cens'])
        self.num_features = self._get_num_features(self.X)
        self.cat_features = self._get_cat_features(self.X)
        return self

    def make_time_event_split(self, y_train, y_valid, y_test) -> Tuple[np.ndarray, np.ndarray,
                                                                       np.ndarray, np.ndarray,
                                                                       np.ndarray, np.ndarray]:
        t_train = np.array(y_train['time'])
        t_valid = np.array(y_valid['time'])
        t_test = np.array(y_test['time'])
        e_train = np.array(y_train['cens'])
        e_valid = np.array(y_valid['cens'])
        e_test = np.array(y_test['cens'])
        return t_train, t_valid, t_test, e_train, e_valid, e_test

class WhasDataLoader(BaseDataLoader):
    def load_data(self) -> None:
        data_x, data_y = load_whas500()
        self.X = data_x
        self.y = convert_to_structured(data_y['lenfol'], data_y['fstat'])
        self.num_features = self._get_num_features(self.X)
        self.cat_features = self._get_cat_features(self.X)
        return self

    def make_time_event_split(self, y_train, y_valid, y_test) -> None:
        t_train = np.array(y_train['lenfol'])
        t_valid = np.array(y_valid['lenfol'])
        t_test = np.array(y_test['lenfol'])
        e_train = np.array(y_train['fstat'])
        e_valid = np.array(y_valid['fstat'])
        e_test = np.array(y_test['fstat'])
        return t_train, t_valid, t_test, e_train, e_valid, e_test

class FlchainDataLoader(BaseDataLoader):
    def load_data(self) -> None:
        data_x, data_y = load_flchain()
        self.X = data_x
        self.y = convert_to_structured(data_y['futime'], data_y['death'])
        self.num_features = self._get_num_features(self.X)
        self.cat_features = self._get_cat_features(self.X)
        return self

    def make_time_event_split(self, y_train, y_valid, y_test) -> None:
        t_train = np.array(y_train['futime'])
        t_valid = np.array(y_valid['futime'])
        t_test = np.array(y_test['futime'])
        e_train = np.array(y_train['death'])
        e_valid = np.array(y_valid['death'])
        e_test = np.array(y_test['death'])
        return t_train, t_valid, t_test, e_train, e_valid, e_test

class MetabricDataLoader(BaseDataLoader):
    def load_data(self) -> None:
        data = datasets.metabric.read_df()
        
        outcomes = data.copy()
        outcomes['event'] =  data['event']
        outcomes['time'] = data['duration']
        outcomes = outcomes[['event', 'time']]
        
        num_feats =  ['x0', 'x1', 'x2', 'x3', 'x8'] \
                     + ['x4', 'x5', 'x6', 'x7']
        
        self.num_features = num_feats
        self.cat_features = []
        self.X = pd.DataFrame(data[num_feats])
        self.y = convert_to_structured(outcomes['time'], outcomes['event'])
        
        return self

    def make_time_event_split(self, y_train, y_valid, y_test) -> None:
        t_train = np.array(y_train[:,1])
        t_valid = np.array(y_valid[:,1])
        t_test = np.array(y_test[:,1])
        e_train = np.array(y_train[:,0])
        e_valid = np.array(y_valid[:,0])
        e_test = np.array(y_test[:,0])
        return t_train, t_valid, t_test, e_train, e_valid, e_test

