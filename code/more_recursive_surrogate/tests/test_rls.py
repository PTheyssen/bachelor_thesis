import numpy as np
import math
import unittest
import matplotlib.pylab as plt
from more.rls_algo import *


class TestRLSConstantModel(unittest.TestCase):

    def test_init_prior(self):
        n = 5
        dim_tri = int(n * (n + 1) / 2)
        model_dim = 1 + n + dim_tri
        lower_tri = np.tril_indices(n)        

        rls = RLSDriftModel(10, model_dim, dim_tri, True)
        par = rls.m_old
        a_0 = par[0]
        a_lin = par[1:n + 1]
        a_quad = np.zeros((n, n))
        a_tri = par[n + 1:].flatten()
        a_quad[lower_tri] = a_tri
        # a_quad = -1 * (a_quad + a_quad.T)
        a_quad = -1/2 * (a_quad + a_quad.T)

        print(f'quadratic_term:\n{a_quad}')
        print(f'linear_term:\n{a_lin}')
        print(f'scalar_term:\n{a_0}')        

    def test_step(self):
        n = 20
        x = np.linspace(0.1, 20, n)
        true_signal = 0.6 + 0.3 * x
        noise = np.random.normal(0, 0.4, n)
        y = true_signal + noise

        rls = RLSConstantModel(100, 2, 0, True)
        predictions = []
        for i in range(n):
            mean, _ = rls.step(np.hstack([1, x[i]]), y[i], 0.1)
            predictions.append(mean)

        # plot predicted parameters    
        plt.subplot(2,1,1)
        plt.plot(x, [0.6]*n, color='r', label='true par_0')
        plt.plot(x, [a[0] for a in predictions], '--', color='r', label='predicted par_0')
        plt.plot(x, [0.3]*n, color='b', label='true par_1')
        plt.plot(x, [a[1] for a in predictions], '--', color='b', label='predicted par_1')
        plt.title("RLS estimating two constant parameters from a "\
                      "signal with white noise (20 samples)")
        plt.legend()


        # plot true signal and predicted signal and measurements
        plt.subplot(2,1,2)
        plt.scatter(x, y, s=8, c='blue', label='measurements')
        plt.plot(x, true_signal, c='blue', label='true signal')
        p = predictions[-1]
        pred = [p[0] + p[1] * a for a in x]
        plt.plot(x, pred, '--', color='r', label='final prediction')
        plt.legend()
        # plt.show()


class TestRLSDriftModel(unittest.TestCase):

    def test_step(self):
        """RLS estimating a linear signal with white noise (var = 0.2).
           The parameters perform a gaussian random walk each step.
           RLS uses one sample at each step, the covariance of the gaussian random walk
           and the variance of the white noise to estimate the 2 parameters.
        """
        n = 100
        x = np.linspace(0.1, 1, n)

        par = [0.6, 0.3]
        cov = [[0.2, 0.01], [0.01, 0.5]]
        parameters = []
        for i in range(n):
            # perform gaussian random walk
            par = np.random.multivariate_normal(par, cov)
            parameters.append(par)


        # get targets by evaluating parameters
        true_signal = []
        for i, p in enumerate(parameters):
            true_signal.append(p[0] + p[1] * x[i])
        noise = 0.2
        y = true_signal + np.random.normal(0, noise, n)


        rls = RLSDriftModel(10, 2, 0, True)
        predictions = []
        for i in range(n):
            mean, _ = rls.step(np.hstack([1, x[i]]), y[i], noise, cov)
            predictions.append(mean)

        plt.subplot(2,1,1)
        plt.plot(x, [p[0] for p in parameters], color='r', label='true par_0')
        plt.plot(x, [a[0] for a in predictions], '--', color='r', label='estimated par_0')
        plt.title("RLS estimating parameters of a linear signal that peform "\
                 "gaussian random walk,\n using one sample per step")                 
        plt.legend()
        
        plt.subplot(2,1,2)        
        plt.plot(x, [p[1] for p in parameters], color='b', label='true par_1')
        plt.plot(x, [a[1] for a in predictions], '--', color='b', label='estimated par_1')
        plt.legend()

        # plt.show()

    def test_sin_curve(self):
        n = 150
        x = np.linspace(0.1, 5, n)
        true_signal = [math.sin(a) for a in x]
        y = true_signal + np.random.normal(0, 0.15, n)

        rls = RLSDriftModel(1, 2, 0, True)
        predictions = []
        cov = [[0.0003, 0], [0, 0.0003]]
        for i in range(n):
            mean, _ = rls.step(np.hstack([1, x[i]]), y[i], 0.15, cov)
            predictions.append(mean[0] + mean[1] * x[i])


        plt.plot(x, true_signal, linestyle='--', color='b', label='true signal')
        plt.scatter(x, y, s=8, c='blue', label='measurements')
        plt.plot(x, predictions, color='r', label='estimated signal')
        plt.title("sine signal with white noise estimated by RLS dynamic model (150 samples)")
        plt.legend()
        # plt.show()

if __name__ == '__main__':
    unittest.main()
