import numpy as np
import math
from gaussian import ND

def getLogLikelihood(means, weights, covariances, X):
    # Log Likelihood estimation
    #
    # INPUT:
    # means          : Mean for each Gaussian KxD
    # weights        : Weight vector 1xK for K Gaussians
    # covariances    : Covariance matrices for each gaussian DxDxK
    # X              : Input data NxD
    # where N is number of data points
    # D is the dimension of the data points
    # K is number of gaussians
    #
    # OUTPUT:
    # logLikelihood  : log-likelihood

    logLikelihood = 0
    for x in X: 
        logLikelihood += sum([math.log(weight * ND(mean, covariance, x)) for mean, weight, covariance in zip(means, weights, covariances)])
    return logLikelihood

