import pandas as pd
import numpy as np
import pytest
import os

from lager.baseline_forecasters.baseline_forecast import get_baseline_forecast_recursive, get_baseline_forecast_simple
from lager.ml_forecasters.recursive_forecaster import RecursiveForecaster
from lager.ml_forecasters.simple_forecaster import SimpleForecaster


@pytest.fixture
def get_ts():
    """Get ts and params for testing RecursiveForecaster and SimpleForecaster."""
    name = os.getcwd()
    # name = '/home/grisha/PycharmProjects/TimeSeries/test'
    f_ts = np.log(np.array(pd.read_csv(name + '/data/crypto_usd.csv')['volume']))
    w_ts = np.array(pd.read_csv(name + '/data/temperature.csv')['temperature'].dropna(how='any'))
    split = 100
    params = {}
    for (ts, key) in [(f_ts, 'finance'), (w_ts, 'weather')]:
        ts_train = ts[:-split]
        ts_test = ts[-split:]
        baseline_recursive = get_baseline_forecast_recursive(ts_train=ts_train, ts_test=ts_test)
        baseline_simple = get_baseline_forecast_simple(ts_train=ts_train, ts_test=ts_test)
        params[key] = {'ts_test': ts_test, 'ts_train': ts_train,
                       'baseline_recursive': baseline_recursive, 'baseline_simple': baseline_simple}
    return params


@pytest.mark.parametrize("series, error",
                         [
                             ('finance', 10),
                             ('weather', 1)
                         ])
def test_recursive_forecaster(get_ts: dict, series: str, error: float):
    """Unit test for RecursiveForecaster."""
    fixture = get_ts[series]

    R = RecursiveForecaster(ts=fixture['ts_train'])
    R.forecast(length=len(fixture['ts_test']))
    metric = R.get_metric(ts_test=fixture['ts_test'])

    assert metric < fixture['baseline_recursive'] * error, "RecursiveForecaster is too bad..."

@pytest.mark.parametrize("series, error",
                         [
                             ('finance', 10),
                             ('weather', 1)
                         ])
def test_simple_forecaster(get_ts: dict, series: str, error: float):
    """Unit test for SimpleForecaster."""
    fixture = get_ts[series]

    S = SimpleForecaster(ts=fixture['ts_train'], future_ts=fixture['ts_test'])
    S.forecast()
    metric = S.get_metric()

    assert metric < fixture['baseline_simple'] * error, "SimpleForecaster is too bad..."
