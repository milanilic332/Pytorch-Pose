import numpy as np
from scipy.stats import multivariate_normal


def get_keypoint_mask(kp_size=32, s=0.2):
    """Creates a 2d Gaussian mask representing one keypoint

    :param kp_size:     size of square containing gaussian (ex. 32x32)
    :param s:           sigma - Gaussian noise
    :return:            2d Gaussian mask
    """
    x, y = np.mgrid[0:1:(kp_size * 1j), 0:1:(kp_size * 1j)]
    xy = np.column_stack([x.flat, y.flat])

    mu = np.array([0.5, 0.5])
    sigma = np.array([s, s])
    covariance = np.diag(sigma ** 2)

    z = multivariate_normal.pdf(xy, mean=mu, cov=covariance)
    z = z.reshape(x.shape)
    z = z*(1. / np.max(z))

    return z
