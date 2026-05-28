#!usr/bin/env python3
'''principal component analysis'''
from sklearn.decomposition import PCA


def Apply_PCA(X, n_components, random_state):
    '''apply pca'''
    pca = PCA(n_components=n_components, random_state=random_state)
    return pca.fit_transform(X)
