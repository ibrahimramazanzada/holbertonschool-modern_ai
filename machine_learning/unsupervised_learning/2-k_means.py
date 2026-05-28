#!/usr/bin/env python3
'''kmeans clustering'''
from sklearn import cluster


def K_Means(X, n_clusters, random_state):
    '''kmeans clustering'''
    kmeans = cluster.KMeans(n_clusters=n_clusters,
                            random_state=random_state)
    return kmeans.fit(X)
