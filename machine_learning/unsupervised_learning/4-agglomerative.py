#!/usr/bin/env python3
'''Agglomerative Clustering and silhouette score'''
from sklearn import cluster
from sklearn import metrics
Apply_PCA = __import__('1-pca').Apply_PCA


def Agglomerative_Clustering(X, n_clusters, random_state,
                             n_components, use_pca_data=True):
    '''Agglomerative Clustering and silhouette score'''
    if use_pca_data:
        X, _ = Apply_PCA(X, n_components, random_state)
    model = cluster.AgglomerativeClustering(n_clusters=n_clusters)
    labels = model.fit_predict(X)
    return model, X, metrics.silhouette_score(X, labels)
