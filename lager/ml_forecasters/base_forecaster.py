from lightgbm import LGBMRegressor
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split

from lager.tools.utils import make_table_ts


class BaseForecaster:
    """Base class for one-dimensional forecasters.

    Args: ts - given ts series for forecasting task
          num_features - num_features (number of lags) to give as input for ml regressor
          regressor - regressor that performs fit/predict methods over data from lagged ts."""

    def __init__(self, ts: np.array, num_features: int = 5,
                 regressor: BaseEstimator = LGBMRegressor(verbose=-1, n_jobs=-1)):
        self.regressor = regressor
        self.num_features = num_features
        self.ts = ts
        self.table_ts = None
        self.col_names = [f"lag {num_features - i}" for i in range(num_features)]
        self.predictions = None
        self.metric = None

    def _preprocess_data(self):
        """Make ts to table data consisting of lagged ts."""
        self.table_ts = make_table_ts(ts=self.ts, num_features=self.num_features)

    def _fit(self):
        """Fit ML classifier over table data obtained in _preprocess_data method."""
        x = self.table_ts.drop(['target'], axis=1)
        y = self.table_ts['target']
        x_train, x_test, y_train, y_test = train_test_split(x, y)
        self.regressor.fit(x_train, y_train)
