import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class WordCounter(BaseEstimator, TransformerMixin):
    def fit(self, X, Y):
        return self

    def transform(self, X):
        return pd.Series(X).apply(lambda x: len(str(x).split(" ")))