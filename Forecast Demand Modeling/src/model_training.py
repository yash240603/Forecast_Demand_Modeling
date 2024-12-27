from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def train_model(features):
    """
    Train a machine learning model using the provided features.
    :param features: DataFrame with features and target variable.
    :return: Trained model.
    """
    X = features.drop('Sales', axis=1)
    y = features['Sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f"Mean Squared Error: {mean_squared_error(y_test, predictions)}")
    return model
