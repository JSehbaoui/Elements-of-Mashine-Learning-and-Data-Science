import numpy as np
from scipy.stats import multivariate_normal
from getLogLikelihood import getLogLikelihood

def EStep(means, covariances, weights, X):
    # Expectation step of the EM Algorithm
    #
    # INPUT:
    # means          : Mean for each Gaussian KxD
    # weights        : Weight vector 1xK for K Gaussians
    # covariances    : Covariance matrices for each Gaussian DxDxK
    # X              : Input data NxD
    #
    # N is number of data points
    # D is the dimension of the data points
    # K is number of Gaussians
    #
    # OUTPUT:
    # logLikelihood  : Log-likelihood (a scalar).
    # gamma          : NxK matrix of responsibilities for N datapoints and K Gaussians.

    N = len(X)
    K = len(means)
    gamma = np.zeros((N, K))
    
    for n, x in enumerate(X):
        for k, (mean, weight) in enumerate(zip(means, weights)):
            gamma[n, k] = weight * multivariate_normal.pdf(x, mean, covariances[:, :, k])
        gamma[n] /= gamma[n].sum()
            
    return [getLogLikelihood(means, weights, covariances, X), gamma]
