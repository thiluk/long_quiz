from typing import List
import sys
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder


class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Change the outlier values: 
        - to upper-bound, if the value is higher than upper-bound, or
        - to lower-bound, if the value is lower than lower-bound respectively.
    """

    def __init__(self, variable:str):

        if not isinstance(variable, str):
            raise ValueError("variable name should be a string")

        self.variable = variable

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        X = X.copy()
        q1 = X.describe()[self.variable].loc['25%']
        q3 = X.describe()[self.variable].loc['75%']
        iqr = q3 - q1
        self.lower_bound = q1 - (1.5 * iqr)
        self.upper_bound = q3 + (1.5 * iqr)
        
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        
        for i in X.index:
            if X.loc[i, self.variable] > self.upper_bound:
                X.loc[i, self.variable]= self.upper_bound
            if X.loc[i, self.variable] < self.lower_bound:
                X.loc[i, self.variable]= self.lower_bound

        return X

class NullImputer(BaseEstimator, TransformerMixin):
    """ Impute missing values in column by replacing them with the most frequent category value """

    def __init__(self, feature=None):
        self.feature = feature

    def transform(self, X):
        X[self.feature].fillna(self.params_, inplace=True)
        return X

    def fit(self, X,y=None):
        df = X.copy()
        feature_mode = df[self.feature].value_counts().index[0]
        self.params_ = feature_mode
        return self

class columnDropperTransformer():
    def __init__(self,columns):
        self.columns=columns

    def transform(self,X,y=None):
        df = X.copy()
        df.drop(columns=self.columns,inplace=True)
        return df


    def fit(self, X, y=None):
        return self

