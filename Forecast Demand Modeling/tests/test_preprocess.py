import pandas as pd
from src import preprocess

def test_clean_data():
    sample_data = {'Date': ['2023-01-01', '2023-01-02'], 'Sales': [50, None]}
    df = pd.DataFrame(sample_data)
    df.to_csv('data/sample_test.csv', index=False)
    cleaned = preprocess.clean_data('data/sample_test.csv')
    assert cleaned.isna().sum().sum() == 0
