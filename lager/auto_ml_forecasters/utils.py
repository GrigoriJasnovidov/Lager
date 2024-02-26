import numpy as np
import optuna
from typing import Iterable
from sklearn.base import BaseEstimator

from lager.validation.validation_procedure import validation_procedure
from lager.ml_forecasters.recursive_forecaster import RecursiveForecaster
from lager.ml_forecasters.simple_forecaster import SimpleForecaster
from lager.baseline_forecasters.baseline_forecast import get_baseline_forecast_recursive, get_baseline_forecast_simple


def get_best_model(model_type: str,
                   ts: np.array,
                   regressors: Iterable[BaseEstimator],
                   num_features: Iterable[int],
                   train_size: int,
                   test_size: int,
                   fold_choice_type: str,
                   search: str,
                   optuna_n_trials: int,
                   metric: str,
                   verbose: bool):
    """Find best RecursiveForecaster or SimpleForecaster.

    Main method for AutoForecaster class. Chooses the best possible model.
    Args: model_type - which model to solve. Options: 'recursive' find RecusrsiveForecaster;
                                                      'simple' find SimpleClassifier
          ts - given ts
          regressors - possible regressors to choose from
          num_features - possible num_features to choose from
          train_size - size of train part for folds
          test_size - size of test part for folds
          fold_choice_type - way to build folds, see make_ts_folds doc for details
          search - how to search best model; Options: 'optuna' or 'all_models'
          optuna_n_trials - number of trials for optuna
          metric - metric to evaluate performance
          verbose - whether to show training process.
    Returns: dictionary with best regressor, num_features and metric.
    """
    regressors_dict = {r.__class__.__name__: r for r in regressors}
    if search == 'all_models':
        results = []
        for r in regressors:
            for n in num_features:
                if verbose:
                    print(r.__class__.__name__, n)
                vp = validation_procedure(ts=ts, regressor=r, num_features=n, model_type=model_type,
                                          metric=metric, test_size=test_size,
                                          train_size=train_size,
                                          fold_choice_type=fold_choice_type)
                results.append({'regressor': r.__class__.__name__, 'num_features': n, 'metric': vp.mean()})
        best_trial = sorted(results, key=lambda x: x['metric'])[0]

        res = {'regressor': regressors_dict[best_trial['regressor']],
               'num_features': best_trial['num_features'],
               'metric': best_trial['metric']}

    elif search == 'optuna':
        regressors_list = [r.__class__.__name__ for r in regressors]

        def objective(trial):
            regressor_name = trial.suggest_categorical("regressor", regressors_list)
            regressor = regressors_dict[regressor_name]
            possible_number_features = trial.suggest_categorical("num_features", num_features)

            vp_optuna = validation_procedure(ts=ts, regressor=regressor, num_features=possible_number_features,
                                             model_type=model_type,
                                             metric=metric,
                                             test_size=test_size, train_size=train_size,
                                             fold_choice_type=fold_choice_type)
            return np.mean(vp_optuna)

        study = optuna.create_study()
        study.optimize(objective, n_trials=optuna_n_trials)

        res = {'regressor': regressors_dict[study.best_params['regressor']],
               'num_features': study.best_params['num_features'],
               'metric': study.best_value}

    else:
        raise ValueError("Incorrect value of argument 'search'. Must be either 'all_models' or 'optuna'.")

    if verbose:
        print(f"Best regressor {res['regressor']} with {res['num_features']} num_features; "
              f"value of {metric} metric is {res['metric']}")

    return res


def get_validation_value(ts: np.array,
                         ts_val: np.array,
                         model_type: str,
                         regressor: BaseEstimator,
                         num_features: int,
                         num_iterations: int,
                         verbose: bool):
    """Get validation metric for the choice of get_best_model method.

    Args: ts - initial training ts
          ts_val - validation series
          model_type - type of the model; 'simple' or 'recursive'
          regressor - regressor to be evaluated
          num_features - num_features or regressor
          num_iterations - number of repetitions for validation procedure
          verbose - weather to show results.
    Returns: metric value for performance of the best RecursiveRegressor over validation ts.
    """
    validation_values = []
    if model_type == 'recursive':
        for i in range(num_iterations):
            R = RecursiveForecaster(ts=ts, regressor=regressor,
                                    num_features=num_features)
            R.forecast(length=len(ts_val))
            validation_values.append(R.get_metric(ts_test=ts_val))
        baseline = get_baseline_forecast_recursive(ts_train=ts, ts_test=ts_val)
    elif model_type == 'simple':
        for i in range(num_iterations):
            S = SimpleForecaster(ts=ts, future_ts=ts_val, regressor=regressor,
                                 num_features=num_features)
            S.forecast()
            validation_values.append(S.get_metric())
        baseline = get_baseline_forecast_simple(ts_train=ts, ts_test=ts_val)

    regressor_metric = np.array(validation_values).mean()
    if verbose:
        print(f"Baseline metric: {baseline}; regressor average metric after {num_iterations} "
              f"trials: {regressor_metric}.")
    return regressor_metric
