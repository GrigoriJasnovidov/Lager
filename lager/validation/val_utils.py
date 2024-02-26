import numpy as np


def make_ts_folds(ts: np.array, test_size: int, train_size: int, mode: str):
    """Make train/test for one-dimensional ts.

    Args: ts - given ts.
          test_size - size of test part
          train_size - size of train part
          mode - way to build folds. Options: 'rolling start' - all folds consist of fixed size train and test parts.
                 First test_fold consist of final test_size values and first train_fold consist of next train_size
                 values. The second fold is shifted on test_size values to direction of ts beginning.
                 'initial_start' test folds are the same as in 'rolling start mode, while train folds complete them
                 to the beginning of ts.
    Returns: list of dicts; each dict consist of train and test folds.'"""

    folds = []
    s = 0
    if mode == 'rolling_start':
        while train_size + test_size + s <= len(ts):
            if s == 0:
                folds.append({'train': ts[-s - test_size - train_size: -s - test_size], 'test': ts[-s - test_size:]})
            else:
                folds.append({'train': ts[-s - test_size - train_size: -s - test_size], 'test': ts[-s - test_size:-s]})
            s += test_size
    elif mode == 'initial_start':
        while train_size + test_size + s <= len(ts):
            if s == 0:
                folds.append({'train': ts[:-s - test_size], 'test': ts[-s - test_size:]})
            else:
                folds.append({'train': ts[:-s - test_size], 'test': ts[-s - test_size:-s]})
            s += test_size
    else:
        raise ValueError("Incorrect mode. Possible options: 'rolling_start' or 'initial_start'.")

    return folds
