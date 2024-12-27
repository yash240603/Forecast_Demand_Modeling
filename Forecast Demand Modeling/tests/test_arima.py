import pandas as pd
import pytest
from src.arima_forecasting import ARIMAForecaster

@pytest.fixture
def sample_data():
    # Create a sample time series dataset
    dates = pd.date_range(start="2023-01-01", periods=100, freq='D')
    values = pd.Series(range(100)) + pd.Series([10]*100)
    return pd.DataFrame({'date': dates, 'demand': values}).set_index('date')

def test_arima_fit(sample_data):
    forecaster = ARIMAForecaster(order=(2, 1, 2))
    model = forecaster.fit(sample_data, target_column='demand')
    assert model is not None, "Model fitting failed."

def test_arima_forecast(sample_data):
    forecaster = ARIMAForecaster(order=(2, 1, 2))
    forecaster.fit(sample_data, target_column='demand')
    forecast = forecaster.forecast(steps=5)
    assert len(forecast) == 5, "Forecast length mismatch."

def test_arima_plot(sample_data):
    forecaster = ARIMAForecaster(order=(2, 1, 2))
    forecaster.fit(sample_data, target_column='demand')
    forecast = forecaster.forecast(steps=10)
    try:
        forecaster.plot_forecast(forecast_df=forecast, historical_data=sample_data['demand'])
    except Exception as e:
        pytest.fail(f"Plotting failed: {e}")
