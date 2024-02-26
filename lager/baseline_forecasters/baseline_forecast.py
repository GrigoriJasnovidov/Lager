import numpy as np
from sklearn.metrics import mean_squared_error


def get_baseline_forecast_recursive(ts_train: np.array, ts_test: np.array, metric: str = 'mse'):
    """Get metric of forecast consisting of the last value of given ts.

    Args: ts_train - given ts for forecasting task
          ts_tes - actual future values of given ts
          metric - metric to evaluate performance.
    Returns: value of metric.
    """
    value = ts_train[-1]
    base_forecast = np.full(len(ts_test), value)
    if metric == 'mse':
        return mean_squared_error(base_forecast, ts_test)


def get_baseline_forecast_simple(ts_train: np.array, ts_test: np.array, metric: str = 'mse'):
    """Get metric of forecast that consists of previous value for each step.

    Args: ts_train - given ts for forecasting task
          ts_tes - actual future values of given ts
          metric - metric to evaluate performance.
    Returns: value of metric.
    """
    base_forecast = np.append(ts_train[-1], ts_test[:-1])
    if metric == 'mse':
        return mean_squared_error(base_forecast, ts_test)
