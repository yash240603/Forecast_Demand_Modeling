from src.preprocess import Preprocessor
from src.arima_forecasting import ARIMAForecaster
import pandas as pd

def main():
    # Load and preprocess data
    data = pd.read_csv('data/sample_data.csv', parse_dates=['date'])
    data.set_index('date', inplace=True)
    preprocessor = Preprocessor()
    data = preprocessor.clean_data(data)

    # Apply ARIMA for forecasting
    arima_forecaster = ARIMAForecaster(order=(2, 1, 2))
    arima_forecaster.fit(data, target_column='demand')
    forecast = arima_forecaster.forecast(steps=10)
    print(forecast)
    
    # Plot the forecast
    arima_forecaster.plot_forecast(forecast_df=forecast, historical_data=data['demand'])

if __name__ == "__main__":
    main()
