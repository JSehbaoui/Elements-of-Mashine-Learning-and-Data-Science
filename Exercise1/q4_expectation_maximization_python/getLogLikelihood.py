import math
from scipy.stats import multivariate_normal

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
       logLikelihood += math.log(
           sum(
               [weight * multivariate_normal.pdf(x, mean, covariances[:, :, j])
                for j, (mean, weight) in enumerate(zip(means, weights))
                ]
               )
           )
       
    return logLikelihood
