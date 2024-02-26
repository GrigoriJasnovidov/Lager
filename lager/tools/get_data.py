import pandas as pd
import numpy as np


def get_ts(name: str = 'data/usd_crypto/STRAX-USD.csv', train_test_ratio: float = 0.8, log: bool = True):
    """Get ts. Default is some crypto/usd data.

    Args: name - .csv file containing data
          train_test_ratio - proportion train part
          log - whether to take log of ts.
    Returns: dict with ts, ts_train and ts_test in np.array type.
    """
    data = pd.read_csv(name)
    ts = np.array(data['volume'])
    if log:
        ts = np.log(ts)

    partition_index = int(len(ts) * train_test_ratio)
    ts_train = ts[:partition_index]
    ts_test = ts[partition_index:]

    return {'ts_train': ts_train, 'ts_test': ts_test, 'ts': ts}


def get_table_ts(name: str = 'data/temperature/ts.csv', train_test_ratio: float = 0.8):
    """Get data for multidimensional ts. Default is some weather data.

    Args: name - .csv file containing data
          train_test_ratio - proportion of train part.
    Returns: dict with multidimensional ts, ts_train and ts_test in pd.DataFrame type.
    """
    data = pd.read_csv(name).dropna(how='any')
    data = data.drop(['datetime_local', 'icon', 'ozone', 'precip_intensity'], axis=1).rename(
        columns={'temperature': 'target'})
    data_split = int(data.shape[0] * train_test_ratio)
    data_train = data[:data_split]
    data_test = data[data_split:]

    return {'data_train': data_train, 'data_test': data_test, 'data': data}
