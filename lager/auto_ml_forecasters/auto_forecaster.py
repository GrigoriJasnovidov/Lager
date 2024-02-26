from typing import Iterable
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from lightgbm import LGBMRegressor

from lager.auto_ml_forecasters.utils import get_best_model, get_validation_value


class ParamsAutoForecaster:
    """Class containing parameters for AutoRecursiveForecaster class.

    Args: num_features - list of possible lags for RecursiveForecaster
          regressors - list of regressors. Each regressor must have fit and predict methods
          metric - metrics to optimize performance
          test_size - size of test sample during validation procedure
          train_size - size during validation procedure
          fold_choice_type - mode of choosing folds. Options: 'rolling_start' - all folds have the same train and test
                             sizes; 'initial_start' - all train parts starts at the first point of given ts
          verbose - whether to show progress
    """
    default_num_features = (1, 2, 3, 5, 10, 24)
    default_regressors = (RandomForestRegressor(), LGBMRegressor(verbose=-1, n_jobs=-1),
                          DecisionTreeRegressor(), LinearRegression())

    def __init__(self,
                 num_features: Iterable[int] = default_num_features,
                 regressors: Iterable[BaseEstimator] = default_regressors,
                 metric: str = 'mse',
                 test_size: int = 100,
                 train_size: int = 400,
                 fold_choice_type: str = 'rolling_start',
                 verbose: bool = True):
        self.num_features = num_features
        self.regressors = regressors
        self.metric = metric
        self.test_size = test_size
        self.train_size = train_size
        self.fold_choice_type = fold_choice_type
        self.verbose = verbose


class AutoForecaster(ParamsAutoForecaster):
    """Class to find best parameters for RecursiveForecaster or SimpleForecaster.

    Args: ts - given train series
          model_type - which model to solve. Options: 'recursive' find RecursiveForecaster;
                                                      'simple' find SimpleClassifier.
    """

    def __init__(self, ts: np.array, model_type: str, **kwargs):
        super().__init__(**kwargs)
        if model_type not in ['simple', 'recursive']:
            raise ValueError("Incorrect 'model_type' argument! Must be 'simple' or 'recursive'.")
        self.model_type = model_type
        self.ts = ts

        self.best_regressor = None
        self.best_num_features = None
        self.best_metric = None

    def get_best_model(self, search: str = 'all_models', optuna_n_trials: int = 20):
        """Find best model.

        Args: search - how to find models. Options: 'all_models' look over all possible models;
                                                    'optuna' use optuna to find the best model;
              optuna_n_trial - number trials for optuna.
        """
        best_model = get_best_model(model_type=self.model_type,
                                    ts=self.ts,
                                    regressors=self.regressors,
                                    num_features=self.num_features,
                                    train_size=self.train_size,
                                    test_size=self.test_size,
                                    fold_choice_type=self.fold_choice_type,
                                    search=search,
                                    optuna_n_trials=optuna_n_trials,
                                    metric=self.metric,
                                    verbose=self.verbose)

        self.best_regressor = best_model['regressor']
        self.best_num_features = best_model['num_features']
        self.best_metric = best_model['metric']

    def get_validation_value(self, ts_val: np.array, num_iterations: int = 10):
        """Get validation metric for the choice of get_best_model method.

               Args: ts_val - validation series
                     num_iterations - number of repetitions for validation procedure.
               Returns: metric value for performance of the best RecursiveRegressor over validation ts.
        """

        return get_validation_value(ts=self.ts,
                                    ts_val=ts_val,
                                    model_type=self.model_type,
                                    regressor=self.best_regressor,
                                    num_features=self.best_num_features,
                                    num_iterations=num_iterations,
                                    verbose=self.verbose)
