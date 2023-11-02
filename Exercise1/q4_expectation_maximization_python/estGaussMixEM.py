import numpy as np
from EStep import EStep
from MStep import MStep
from regularize_cov import regularize_cov
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances


def estGaussMixEM(data, K, n_iters, epsilon):
    # EM algorithm for estimation gaussian mixture mode
    #
    # INPUT:
    # data           : input data, N observations, D dimensional
    # K              : number of mixture components (modes)
    #
    # OUTPUT:
    # weights        : mixture weights - P(j) from lecture
    # means          : means of gaussians
    # covariances    : covariancesariance matrices of gaussians

    N, D = data.shape
    
    # Initialize weights uniformly
    weights = np.ones(K) / K
    
    # Initialize means and covariances using K-Means
    kmeans = KMeans(n_clusters=K, n_init=10).fit(data)
    means = kmeans.cluster_centers_
    covariances = np.zeros((K, D, D))
    
    # Create initial covariance matrices
    for j in range(K):
        data_cluster = data[kmeans.labels_ == j]
        min_dist = np.inf
        for i in range(K):
            dist = np.mean(euclidean_distances(data_cluster, [means[i]], squared=True))
            if dist < min_dist:
                min_dist = dist
        covariances[j] = np.eye(D) * min_dist
    
    for _ in range(n_iters):
        # E-Step
        gamma = EStep(means, covariances, weights, data)[1]
        
        # M-Step
        new_weights, new_means, new_covariances, logLikelihood = MStep(gamma, data)
        
        new_covariances = [regularize_cov(covariance) for covariance in new_covariances]
        
        # Update parameters
        weights, means, covariances = new_weights, new_means, new_covariances
        
    return [weights, means, covariances]
