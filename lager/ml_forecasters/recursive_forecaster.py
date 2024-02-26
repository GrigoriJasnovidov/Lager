import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from lager.tools.utils import plot_forecast
from lager.ml_forecasters.base_forecaster import BaseForecaster


class RecursiveForecaster(BaseForecaster):
    """Class to forecast one-dimensional ts on several steps forward."""

    def forecast(self, length: int):
        """Build forecast.

        Args: length - length of forecast.
        Returns: np.array of predictions.
        """
        self._preprocess_data()
        self._fit()
        init_features = self.ts[-self.num_features:]
        current_features = init_features.reshape(1, -1)
        predictions = []
        for i in range(length):
            current_features = pd.DataFrame(current_features, columns=self.col_names)
            new_prediction = self.regressor.predict(current_features)
            predictions.append(new_prediction)
            current_features = np.array(current_features)
            current_features = np.append(current_features[0, 1:], new_prediction).reshape(1, -1)

        predictions = np.array(predictions)
        self.predictions = predictions

        return predictions

    def get_metric(self, ts_test: np.array, metric: str = 'mse'):
        """Evaluate performance of RecursiveRegressor.

        Args: ts_test - actual time series. Must have the same length as forecast
              metric - metric to evaluate performance of RecursiveRegressor.
        Returns: metric value evaluating performance of RecursiveRegressor.
        """
        if len(self.predictions) != len(ts_test):
            raise ValueError("ts_test and forecast must have the same length!")
        if metric == 'mse':
            self.metric = mean_squared_error(self.predictions, ts_test)
        return self.metric

    def plot(self, ts_test: np.array = None):
        """Plot forecast and (if given) actual ts.

        Args: ts_test - actual future ts. Options: None - plot only forecast;
                                                  np.array of the same length as length in 'forecast' method.
        """
        plot_forecast(ts_train=self.ts, ts_test=ts_test, forecast=self.predictions)
