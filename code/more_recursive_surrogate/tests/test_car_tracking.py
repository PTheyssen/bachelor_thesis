import numpy as np
import math
import unittest
import matplotlib.pylab as plt
import numpy.random as random
from more.kalman_filter import *

class TestKalmanFilter(unittest.TestCase):

    def test_kf(self):
        # set the parameters
        q = 1
        dt = 0.1
        s = 0.5
        A = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        Q = q * np.array([[dt**3/3, 0, dt**2/2, 0],
                          [0, dt**3/3, 0, dt**2/2],
                          [dt**2/2, 0, dt, 0],
                          [0, dt**2/2, 0, dt]])
        
        H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]])
        R = s ** 2 * np.eye(2)
        m0 = np.array([0, 0, 1, -1]).reshape(4,1)
        P0 = np.eye(4)

        # simulate data
        np.random.seed(1)
        Arow, _ = np.shape(A)
        Hrow, _ = np.shape(H)

        steps = 100
        X = np.zeros(Arow).reshape(Arow,1)
        X = np.tile(X, (1, steps))
        Y = np.zeros(Hrow).reshape(Hrow,1)
        Y = np.tile(Y, (1, steps))
        x = m0

        for k in range(steps):
            q = np.linalg.cholesky(Q).conj() @ np.random.randn(Arow, 1)
            x = A @ x + q
            y = H @ x + s * np.random.randn(2,1)

            X[:,k] = x.reshape(1, Arow)
            Y[:,k] = y.reshape(1, Hrow)


        # Kalman Filter
        kf = KalmanFilter(m0, P0)
        kf_m = np.zeros((4,100))
        for k in range(steps):
            kf.predict(A, Q)
            m, _ = kf.update(H, R, Y[:,[k]])
            kf_m[:, [k]] = m

            
        plt.plot(X[0,:], X[1,:], label='Trajectory')
        plt.plot(Y[0,:], Y[1,:], '.', label='Measurements')
        plt.plot(kf_m[0,:], kf_m[1,:], '--', label='Prediction')        
        plt.legend()
        plt.show()


if __name__ == '__main__':
    unittest.main()


