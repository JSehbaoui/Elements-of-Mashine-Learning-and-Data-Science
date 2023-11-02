import numpy as np
from gaussian import ND
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
        for j, (mean, covariance, weight) in enumerate(zip(means, covariances, weights)):
            numerator = weight * ND(mean, covariance, x)
            denominator = sum([weightK * ND(meanK, covarianceK, x) for meanK, covarianceK, weightK in zip(means, covariances, weights)])
            gamma[n, j] = numerator / denominator
            

        
    return [getLogLikelihood(means, weights, covariances, X), gamma]
