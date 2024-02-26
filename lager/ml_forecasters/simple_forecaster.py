import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from lager.ml_forecasters.base_forecaster import BaseForecaster
from lager.tools.utils import plot_forecast


class SimpleForecaster(BaseForecaster):
    """Class to forecast one-dimensional ts on one step forward.

    Args: ts - given ts
          future_ts - future values.
    """

    def __init__(self, ts: np.array, future_ts: np.array, **kwargs):
        super().__init__(ts=ts, **kwargs)
        self.future_ts = future_ts

    def forecast(self):
        """Make one-step-forward forecast for initial ts and consequently given future_ts. After fitting process
        on train ts each one-step forecast is build over last actual values of ts.

        Returns: np.array of predictions with the same length as future_ts.
        """
        self._preprocess_data()
        self._fit()
        init_features = self.ts[-self.num_features:]
        current_features = init_features.reshape(1, -1)
        predictions = []
        for i in range(len(self.future_ts)):
            current_features = pd.DataFrame(current_features, columns=self.col_names)
            new_prediction = self.regressor.predict(current_features)
            predictions.append(new_prediction)
            current_features = np.array(current_features)
            current_features = np.append(current_features[0, 1:], self.future_ts[i]).reshape(1, -1)
        predictions = np.array(predictions)
        self.predictions = predictions

        return predictions

    def get_metric(self, metric: str = 'mse'):
        """Evaluate performance.

        Args: metric - metric to evaluate performance over.
        Returns - metric value.
        """
        if metric == 'mse':
            self.metric = mean_squared_error(self.predictions, self.future_ts)
        return self.metric

    def plot(self):
        plot_forecast(ts_train=self.ts, ts_test=self.future_ts, forecast=self.predictions)
