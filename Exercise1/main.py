"""main.py"""
"""
authors: Julian Sehbaoui, Tobias <>, <> <>
date: 30.10.2023
"""
import numpy as np
import math

def ND(mean, covariance, x):
    K = len(mean)
    det_covariance = np.linalg.det(covariance)
    inv_covariance = np.linalg.inv(covariance) #  throws an error?

    if K <= 0:
        raise ValueError()
    
    normalization_factor = 1 / (math.pow(2 * math.pi, K/2) * math.sqrt(det_cov))
    exponent = np.matmul(np.matmul((x - mean).T, inv_cov), (x - mean))
    
    return normalization_factor * math.exp(-0.5 * exponent)


def getLogLikelihood(means, weights, covariances, X):
    logLikelihood = 0
    for x in X:
        logLikelihood = sum([math.log(weight * ND(mean, covariance, x)) for mean, weight, covariance in zip(means, weights, covariances)])
    return logLikelihood
        

def EStep(means, covariances, weights, X):
    N = len(X)
    K = len(means)
    gamma = np.zeros((N, K))
    logLikelihood = 0
    
    for n, x in enumerate(X):
        for j, (mean, covariance, weight) in enumerate(zip(means, covariances, weights)):
            numerator = weight * ND(mean, covariance, x)
            denominator = sum([weightK * ND(meanK, covarianceK, x) for meanK, covarianceK, weightK in zip(means, covariances, weights)])
            gamma[n, j] = numerator / denominator
            
        # Compute log-likelihood
        logLikelihood += np.log(np.sum(gamma[n, :]))
        
    return [logLikelihood, gamma]
