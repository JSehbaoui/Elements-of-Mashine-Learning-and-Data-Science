import numpy as np
import math

def ND(mean, covariance, x):
    K = len(mean)
    det_covariance = np.linalg.det(covariance)
    inv_covariance = np.linalg.inv(covariance) #  throws an error?

    if K <= 0:
        raise ValueError()
    
    normalization_factor = 1 / (math.pow(2 * math.pi, K/2) * math.sqrt(det_covariance))
    exponent = np.matmul(np.matmul((x - mean).T, inv_covariance), (x - mean))
    
    return normalization_factor * math.exp(-0.5 * exponent)