##!/usr/bin/env python3
'''standardscaler'''
from sklearn import preprocessing


def Standardize(X):
    '''standardize'''
    scaler = preprocessing.StandardScaler()
    return scaler.fit_transform(X)
