import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def make_table_ts(ts: np.array, num_features: int):
    """Make from ts table consisting of lagged ts. Used later for ML regressors.
    
    Args: ts - given ts
          num_features - num_features (number of lags) for ML regressor.
    Returns: pd.DataFrame with columns consisting of lagged ts.
    """
    features = []
    target = []
    for i in range(num_features, len(ts)):
        features.append(ts[i - num_features: i])
        target.append(ts[i])
    features = np.array(features)
    feature_cols = [f"lag {num_features - i}" for i in range(num_features)]
    df = pd.DataFrame(features, columns=feature_cols)
    df['target'] = target
    return df


def make_multidim_table_ts(df: pd.DataFrame, num_features: int):
    """Make from multidimensional ts table consisting of lagged ts. Used later for ML regressors.

    Args: df -given df
          num_features - num_features (number of lags) for ML regressor.
    Returns: pd.DataFrame with columns consisting of lagged ts.
    """
    df = df.reset_index()
    features = []
    target = []
    col_names = [f'{name} lag {i + 1}' for name in df.columns for i in range(num_features)]

    for i in range(num_features, df.shape[0]):
        features.append(df[i - num_features: i].reset_index())
        target.append(df['target'][i])

    np_features = []
    for f in features:
        z = []
        for name in df.columns:
            for i in range(num_features):
                z.append(f[name][i])
        np_features.append(np.array(z))
    np_features = np.array(np_features)
    table_data = pd.DataFrame(np_features, columns=col_names)
    table_data['target'] = target
    return table_data


def plot_forecast(ts_train: np.array, ts_test: np.array, forecast: np.array):
    """Plot ts_train, forecast and if given ts_test.

    Args: ts_train - train ts
          ts_test - test ts. Options: None - plot only forecast and train ts; np.array of the same length as forecast
          forecast - forecast.
    """
    if ts_test is not None and len(ts_test) != len(forecast):
        raise ValueError("Forecast and ts_test must have the same length! Change ts_test of length of forecast.")

    train_len = len(ts_train)
    train_range = range(1, train_len + 1)
    forecast_len = len(forecast)
    forecast_range = range(train_len + 1, train_len + forecast_len + 1)

    fig, ax = plt.subplots()
    ax.plot(train_range, ts_train, label='Train TS')
    ax.plot(forecast_range, forecast, label='Forecast')
    if ts_test is not None:
        ax.plot(forecast_range, ts_test, label='Test TS')
    ax.legend()
    plt.show()
