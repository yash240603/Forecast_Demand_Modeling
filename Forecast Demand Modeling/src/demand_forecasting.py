def predict_demand(model, features):
    """
    Predict demand using the trained model.
    :param model: Trained machine learning model.
    :param features: DataFrame with features.
    :return: Array of demand predictions.
    """
    X = features.drop('Sales', axis=1)
    return model.predict(X)
