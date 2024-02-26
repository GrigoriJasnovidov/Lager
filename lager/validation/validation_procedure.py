import numpy as np
from sklearn.base import BaseEstimator

from lager.ml_forecasters.recursive_forecaster import RecursiveForecaster
from lager.ml_forecasters.simple_forecaster import SimpleForecaster
from lager.validation.val_utils import make_ts_folds


def validation_procedure(ts: np.array,
                         regressor: BaseEstimator,
                         num_features: int,
                         model_type: str,
                         metric: str,
                         test_size: int,
                         train_size: int,
                         fold_choice_type: str):
    """Evaluate performance of RecursiveForecaster over several train/test folds.

    Args: ts - given ts
          regressor - regressor having fit/predict methods to build RecursiveForecaster
          num_features - number features for RecursiveForecaster
          metric - metric to evaluate performance
          test_size - size for test folds
          train_size - size for train folds
          folds_choice_type - mode to choose folds.

    Returns: np.array consisting of metrics for each test fold.
    """
    folds = make_ts_folds(ts=ts, test_size=test_size, train_size=train_size, mode=fold_choice_type)
    metric_values = []
    for f in folds:
        ts_test = f['test']
        ts_train = f['train']
        if model_type == 'recursive':
            R = RecursiveForecaster(ts=ts_train, num_features=num_features, regressor=regressor)
            R.forecast(length=len(ts_test))
            metric_values.append(R.get_metric(ts_test=ts_test, metric=metric))
        elif model_type == 'simple':
            S = SimpleForecaster(ts=ts_train, future_ts=ts_test, num_features=num_features, regressor=regressor)
            S.forecast()
            metric_values.append(S.get_metric(metric=metric))
    return np.array(metric_values)
