import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class WordCounter(BaseEstimator, TransformerMixin):
    def wordlength(self, text):
        length = 0
        for word in text.split(" "):
            length += 1
        return length

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        X_tagged = pd.Series(x).apply(self.wordlength)
        df = pd.DataFrame(X_tagged)
        return df
        