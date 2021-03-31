from typing import Tuple

import numpy as np


class MovingMeanVar(object):
    def __init__(self, shape, alpha):
        """Computes the exponentially weighted moving mean and variance.

        Implemented according to:
        https://en.wikipedia.org/wiki/Moving_average#Exponentially_weighted_moving_variance_and_standard_deviation

        Args:
           shape: dim of data
           alpha: weighting factor
        """
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.alpha = alpha


    def update(self, sample: np.ndarray) -> None:
        delta = sample - self.mean
        self.mean = self.mean + self.alpha * delta
        self.var = ((1 - self.alpha)
                           * (self.var + self.alpha * (delta**2)))

