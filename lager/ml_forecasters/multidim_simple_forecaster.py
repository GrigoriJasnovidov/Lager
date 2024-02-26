import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator

from lager.tools.utils import make_multidim_table_ts, plot_forecast


class MultiDimSimpleForecaster:
    """Forecaster to make one-step-forward predictions for multidimensional ts.

    Args: df - DataFrame with multidimensional ts
          future_df DaaFrame with future series
          num_features - number lags for each component of series
          regressor - ML regressor supporting fit/predict methods.
    """

    def __init__(self, df: pd.DataFrame, future_df: pd.DataFrame, num_features: int = 5,
                 regressor: BaseEstimator = LGBMRegressor(verbose=-1, n_jobs=-1)):
        self.df = df
        self.future_df = future_df
        self.full_df = pd.concat([self.df, future_df])
        self.ts = df['target']
        self.future_ts = future_df['target']
        self.num_features = num_features
        self.regressor = regressor

        self.table_ts = None
        self.predictions = None
        self.metric = None

    def _preprocess_data(self):
        """Make data to be a table."""
        self.table_ts = make_multidim_table_ts(df=self.df, num_features=self.num_features)

    def _fit(self):
        """Fit ML regressor on table preprocessed ts."""
        x = self.table_ts.drop(['target'], axis=1)
        y = self.table_ts['target']
        x_train, x_test, y_train, y_test = train_test_split(x, y)
        self.regressor.fit(x_train, y_train)

    def forecast(self):
        """Make forecast.

        Length of forecast is the same as length of future_df.
        Returns: prediction for target series."""
        self._preprocess_data()
        self._fit()
        all_features = make_multidim_table_ts(df=self.full_df, num_features=self.num_features).drop(['target'], axis=1)
        features_for_predictions = all_features[-self.future_df.shape[0]:]
        self.predictions = self.regressor.predict(features_for_predictions)
        return self.predictions

    def get_metric(self, metric='mse'):
        """Get metric over future df.

        Args: metric - name of metric.
        Returns: value of chosen metric."""
        if metric == 'mse':
            self.metric = mean_squared_error(self.predictions, self.future_ts)
        return self.metric

    def plot(self):
        """Plot forecast."""
        plot_forecast(ts_train=self.ts, ts_test=self.future_ts, forecast=self.predictions)
