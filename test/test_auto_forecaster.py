import pandas as pd
import numpy as np
import pytest
import os

from lager.auto_ml_forecasters.auto_forecaster import AutoForecaster
from lager.baseline_forecasters.baseline_forecast import get_baseline_forecast_recursive


@pytest.fixture
def get_ts_auto():
    """Get ts and params for testing AutoForecaster."""
    # name = '/home/grisha/PycharmProjects/TimeSeries/test'
    name = os.getcwd()
    f_ts = np.log(np.array(pd.read_csv(name + '/data/crypto_usd.csv')['volume']))
    w_ts = np.array(pd.read_csv(name + '/data/temperature.csv')['temperature'].dropna(how='any'))
    f_split = 100
    w_split = 50
    params = {}
    for (ts, split, key) in [(f_ts, f_split, 'finance'), (w_ts, w_split, 'weather')]:
        ts_train = ts[:-split]
        ts_test = ts[-split:]
        baseline_metric = get_baseline_forecast_recursive(ts_train=ts_train, ts_test=ts_test)
        params[key] = {'ts_test': ts_test, 'ts_train': ts_train, 'baseline': baseline_metric}
    return params


@pytest.mark.parametrize("model_type, series, search, error, train_size, test_size",
                         [
                             ('recursive', 'finance', 'optuna', 2, 400, 100),
                             ('recursive', 'finance', 'all_models', 2, 400, 100),
                             ('recursive', 'weather', 'optuna', 1, 200, 50),
                             ('recursive', 'weather', 'all_models', 1, 200, 50),
                             ('simple', 'finance', 'optuna', 2, 400, 100),
                             ('simple', 'finance', 'all_models', 2, 400, 100),
                             ('simple', 'weather', 'optuna', 1, 200, 50),
                             ('simple', 'weather', 'all_models', 1, 200, 50)
                         ]
                         )
def test_auto_forecaster(get_ts_auto: dict,
                         model_type: str,
                         series: str,
                         search: str,
                         error: float,
                         train_size: int,
                         test_size: int):
    """Unit test for AutoForecaster."""
    fixture = get_ts_auto[series]

    A = AutoForecaster(ts=fixture['ts_train'], train_size=train_size, test_size=test_size, model_type=model_type)
    A.get_best_model(search=search, optuna_n_trials=20)
    auto_metric = A.get_validation_value(ts_val=fixture['ts_test'], num_iterations=10)

    assert auto_metric < error * fixture['baseline'], "AutoForecaster is too bad..."
