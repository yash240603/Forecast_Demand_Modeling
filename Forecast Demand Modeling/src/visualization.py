import matplotlib.pyplot as plt

def plot_results(predictions):
    """
    Plot the forecasted demand predictions.
    :param predictions: Array of demand predictions.
    """
    plt.plot(predictions, label='Predicted Demand')
    plt.title('Demand Forecasting Results')
    plt.xlabel('Time')
    plt.ylabel('Demand')
    plt.legend()
    plt.show()
