def generate_features(data):
    """
    Generate features from the cleaned dataset.
    :param data: Cleaned DataFrame.
    :return: DataFrame with new features.
    """
    data['Month'] = data['Date'].dt.month
    data['Year'] = data['Date'].dt.year
    return data.drop(['Date'], axis=1)
