from src import model_training
import pandas as pd

def test_train_model():
    sample_data = {
        'Sales': [50, 60, 70, 80],
        'Month': [1, 2, 3, 4],
        'Year': [2023, 2023, 2023, 2023]
    }
    df = pd.DataFrame(sample_data)
    model = model_training.train_model(df)
    assert hasattr(model, 'predict')
