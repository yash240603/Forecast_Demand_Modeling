import pandas as pd

class Preprocessor:
    def clean_data(self, data):
        """
        Basic data cleaning.
        :param data: Pandas DataFrame
        :return: Cleaned DataFrame
        """
        data = data.dropna()
        return data
