import numpy as np
from getLogLikelihood import getLogLikelihood


def MStep(gamma, X):
    # Maximization step of the EM Algorithm
    #
    # INPUT:
    # gamma          : NxK matrix of responsibilities for N datapoints and K Gaussians.
    # X              : Input data (NxD matrix for N datapoints of dimension D).
    #
    # N is number of data points
    # D is the dimension of the data points
    # K is number of Gaussians
    #
    # OUTPUT:
    # logLikelihood  : Log-likelihood (a scalar).
    # means          : Mean for each gaussian (KxD).
    # weights        : Vector of weights of each gaussian (1xK).
    # covariances    : Covariance matrices for each component(DxDxK).

    N, K = np.shape(gamma)
    D = np.shape(X)[1]  # Dimensionality of the data
    
    # Calculate N_hat
    NHat = np.sum(gamma, axis=0)  # Sum along columns
    
    # Calculate weights
    weights = NHat / N
    
    # Calculate means
    means = np.zeros((K, D))
    for j in range(K):
        means[j] = np.sum(gamma[:, j].reshape(-1, 1) * X, axis=0) / NHat[j]
    
    # Calculate covariances
    covariances = np.zeros((K, D, D))
    for j in range(K):
        for n in range(N):
            diff = (X[n] - means[j]).reshape(-1, 1)
            covariances[j] += gamma[n, j] * np.dot(diff, diff.T)
        covariances[j] /= NHat[j]
    
    logLikelihood = getLogLikelihood(means, weights, covariances, X)
    
    return weights, means, covariances, logLikelihood
