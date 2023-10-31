"""main.py"""
"""
authors: Julian Sehbaoui, Tobias <>, <> <>
date: 30.10.2023
"""
import numpy as np
import math

def getColumn(list, n):
    if n > np.shape(list)[1]: raise ValueError()
    return [row[n] for row in list]

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

def MStep(gamma, X):
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
    
    return [weights, means, covariances, logLikelihood]
