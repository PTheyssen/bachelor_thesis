import numpy as np

class RLS:
    """Recursive Least Squares Algorithm, implemented in bayesian filtering
       sense, based on "Bayesian Estimation of time-varying systems"
       by Simo Särkkä.

       Subclasses implement different step methods assuming that the estimated
       parameters are constant or not (drift model).
    """

    def __init__(self, delta, model_dim, dim_tri, norm_prior):
        """Initialize RLS algorithm.

           Args:
               delta: variance of prior
               model_dim: number of parameters to be estimated
               dim_tri: number of parameters of the quadratic term
               norm_prior: whether prior for normalized rewards should be used
        """
        self.dim = model_dim
        self.dim_tri = dim_tri
        org_dim = model_dim - dim_tri - 1 # dimension of the original problem

        self.p_old = delta * np.eye(model_dim)
        if model_dim == 2:
            # for test cases
            self.m_old = np.random.randn(self.dim, 1)
        else:
            if norm_prior:
                self.m_old = self.init_prior(np.sqrt(2) / np.sqrt(org_dim))
            else:
                self.m_old = self.init_prior(1.)
        self.prior = self.m_old

        # for plotting
        self.kalman_gain = None
        self.s = None


    def init_prior(self, entry):
        """Initialize the prior so that the quadratic term has ones on the diagonal and
           all other values are zero
        """
        counter = self.dim - self.dim_tri - 2
        quad = []
        while counter:
            quad.append(entry)
            for i in range(counter):
                quad.append(0.)
            counter -= 1
        quad.append(entry)
        q = np.flip(np.array(quad))
        return np.hstack((np.zeros(self.dim - self.dim_tri), q)).reshape(self.dim,1)

    def step(self):
        raise NotImplementedError("has to be implemented in subclass")


class RLSConstantModel(RLS):
    """Recursive Least Squares Algorithm implemented in bayesian filtering sense
       assuming the parameters to be estimated are constant
       (static linear regregression model).
    """
    def __init__(self, delta, dim, dim_tri, norm_prior):
        super().__init__(delta, dim, dim_tri, norm_prior)

    def step(self, h, y, std):
        """Calculates one step of recursive regression problem assuming the
           estimated parameters are constant.

        Args:
            h: new sample (regressor value) used to compute
               new estimate of parameters
            y: target (response) of regression problem
            std: standard deviation of the measurements

        Returns:
            Mean and covariance of updated posterior distribution (gaussian)
            of the parameters.
        """


        # temporary variables
        s = h @ self.p_old @ h.T + (std ** 2)
        k = (self.p_old @ h.T * (s ** -1)).reshape(self.dim, 1)

        # mean of new posterior distribution of the parameters
        m = self.m_old + k * (y - h @ self.m_old)
        self.m_old = m

        # covariance of new posterior distribution of the parameters
        p = self.p_old - k @ (s * k.T)
        self.p_old = p
        return m, p


class RLSDriftModel(RLS):
    """Recursive Least Squares Algorithm implemented in bayesian filtering sense
       assuming the parameters change between measurements.

       The parameter of the linear regression model perform Gaussian random walk
       between measurements with covariance Q.

       In MORE context we estimate the parameters of a quadratic surrogate model
       of an objective function. We draw samples from a  search distribution
       (multivariate gaussian) and locally evaluate the objective function.
       Therefore the covariance Q should depend on the covariance of the search
       distribution and the KL-divergence bound.
    """
    def __init__(self, delta, dim, dim_tri, norm_prior):
        super().__init__(delta, dim, dim_tri, norm_prior)

    def step(self, h, y, std, par_covar, weighting, K=None, final_step=False, count=None):
        """Calculates one step of recursive regression problem assuming the
           estimated parameters perform Gaussian random walk between
           measurements.

        Args:
            h: measurement sample
            y: target (response) of regression problem
            std: standard deviation of the measurement
            par_covar: covariance of gaussian walk performed by parameters
            weighting (num) : factor to increase model noise for older samples
            K: parameter for transition step for weighting old prior
            count (int) : counter for how many times the sample has been used when using a
                          sample pool
        Returns:
            Mean and covariance of updated posterior distribution (gaussian)
            of the parameters.
        """
        cov = par_covar
        # increase model noise for older samples, based on counter (how often they have been used)
        if count is not None:
            cov = par_covar + weighting * count * np.eye(np.shape(par_covar)[0])
            # TODO: try different weightings
            # cov = par_covar + (count ** weighting) * np.eye(np.shape(par_covar)[0])

        # Add model noise to covariance matrix
        self.p_old = self.p_old + cov

        s = h @  self.p_old @ h.T + (std ** 2)
        self.s = s

        k = (self.p_old @ h.T * (s ** -1)).reshape(self.dim, 1)
        self.kalman_gain = k

        # mean of new posterior distribution of the parameters
        m = self.m_old + k * (y - h @ self.m_old)
        self.m_old = m

        # covariance of new posterior distribution of the parameters
        p = self.p_old - k @ (s * k.T)
        self.p_old = p


        if K is not None:
            self.m_old = K * (self.m_old - self.prior) + self.prior

        return m, p
