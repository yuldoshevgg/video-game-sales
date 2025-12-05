import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

def load_data(path):
    return pd.read_csv(path)

class ReleaseYearExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, date_col='release_date'):
        self.date_col = date_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = X.copy()
        X_[self.date_col] = pd.to_datetime(X_[self.date_col], errors='coerce')
        X_['release_year'] = X_[self.date_col].dt.year.fillna(0).astype(int)
        return X_
