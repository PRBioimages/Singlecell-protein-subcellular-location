# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 19:23:11 2019
@author: nbszg
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2,f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LassoCV


# feature selector
class Feature_Selector(object):

    # RFE based on logistic
    def rfe_selector(self, df, feature_col, label_col, select_num, scale=True):

        if select_num > len(feature_col):
            raise ValueError("rfe_selector's select_num greater than length of feature_col")
        df_copy = df[feature_col].copy()
        df_copy[label_col] = df[label_col]
        if scale == True:
            for col in feature_col:
                df_copy[col] = (df_copy[col] - df_copy[col].min()) / (
                            df_copy[col].max() - df_copy[col].min())
        df_copy = df_copy.fillna(0)
        model = LogisticRegression(solver='lbfgs', max_iter=200)
        rfe = RFE(model, select_num)
        rfe = rfe.fit(df_copy[feature_col], df_copy[label_col])
        rfe_col = list(np.array(feature_col)[rfe.support_])

        return rfe_col
