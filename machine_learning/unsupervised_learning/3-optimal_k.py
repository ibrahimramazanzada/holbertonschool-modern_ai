#!/usr/bin/env python3
'''elbow vs silhouette'''
from sklearn import metrics
K_Means = __import__('2-k_means').K_Means


def optimal_k(X, max_clusters, random_state):
    '''optimal k'''

    elbow = []
    silhouette = []
    for i in range(2, max_clusters + 1):
        model = K_Means(X, i, random_state)
        elbow.append(model.inertia_)
        silhouette.append(metrics.silhouette_score(X, model.labels_))
    clusters = range(2, max_clusters + 1)
    return clusters, elbow, silhouette
