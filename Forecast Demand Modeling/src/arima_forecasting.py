import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

class ARIMAForecaster:
    def __init__(self, order=(5, 1, 0)):
        """
        Initialize the ARIMA model with the specified order.
        :param order: tuple (p, d, q)
        """
        self.order = order
        self.model = None

    def fit(self, data, target_column):
        """
        Fit the ARIMA model to the given time series data.
        :param data: Pandas DataFrame containing time series data
        :param target_column: Column name of the target variable
        :return: Fitted model
        """
        series = data[target_column]
        self.model = ARIMA(series, order=self.order).fit()
        print(f"Model Summary:\n{self.model.summary()}")
        return self.model

    def forecast(self, steps=10):
        """
        Forecast future values based on the fitted model.
        :param steps: Number of future time steps to forecast
        :return: Forecasted values as a Pandas DataFrame
        """
        if not self.model:
            raise Exception("Model not fitted yet. Call the fit method first.")
        forecast = self.model.get_forecast(steps=steps)
        forecast_df = forecast.summary_frame()
        return forecast_df[['mean', 'mean_ci_lower', 'mean_ci_upper']]

    def plot_forecast(self, forecast_df, historical_data):
        """
        Plot the forecast alongside the historical data.
        :param forecast_df: DataFrame with forecasted values and confidence intervals
        :param historical_data: Historical data for reference
        """
        plt.figure(figsize=(12, 6))
        plt.plot(historical_data, label='Historical Data', color='blue')
        plt.plot(forecast_df['mean'], label='Forecast', color='orange')
        plt.fill_between(
            forecast_df.index, 
            forecast_df['mean_ci_lower'], 
            forecast_df['mean_ci_upper'], 
            color='orange', alpha=0.2, label='Confidence Interval'
        )
        plt.legend()
        plt.title("ARIMA Forecast")
        plt.xlabel("Time")
        plt.ylabel("Values")
        plt.show()
