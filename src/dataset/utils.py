import numpy as np
from scipy.stats import multivariate_normal


def get_keypoint_mask():
    x, y = np.mgrid[0:1:20j, 0:1:20j]
    xy = np.column_stack([x.flat, y.flat])

    mu = np.array([0.5, 0.5])
    sigma = np.array([0.25, 0.25])
    covariance = np.diag(sigma ** 2)

    z = multivariate_normal.pdf(xy, mean=mu, cov=covariance)
    z = z.reshape(x.shape)
    z = z*(1. / np.max(z))

    return z
