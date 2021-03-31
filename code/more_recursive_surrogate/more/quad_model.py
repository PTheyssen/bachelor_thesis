from collections import deque
import numpy as np
from types import SimpleNamespace
from cma.evolution_strategy import RecombinationWeights
from joblib import Parallel, delayed
from more.rls_algo import RLSDriftModel
from more.kalman_filter import KalmanFilter
from more.running_mean_std import RunningMeanStd
from more.moving_average import MovingMeanVar


def default_config_ls():
    config = {"output_weighting": "min_max",
              "whiten_input": True,
              "normalize_features": True,
              "normalize_output": "mean_std",
              "unnormalize_output": False,
              "ridge_factor": 1e-12,
              "limit_model_opt": True,
              "refit": False,
              "seed": None}

    return config


def default_config_ls_rank():
    config = {"output_weighting": None,
              "whiten_input": True,
              "normalize_features": True,
              "normalize_output": "rank",
              "unnormalize_output": False,
              "ridge_factor": 1e-12,
              "limit_model_opt": True,
              "refit": False,
              "seed": None}

    return config


class MoreModel:
    """ Quadratic surrogate model for MORE algorithm.
    """
    def __init__(self, n, config_dict):
        self.n = n
        self.options = SimpleNamespace(**config_dict)

        self.data_x_org = None
        self._data_x_mean = None
        self._data_x_inv_std = None

        self.data_y_org = None
        self._data_y_mean = None
        self._data_y_std = None

        self.data_y_min = None
        self.data_y_max = None

        self._a_quad = np.eye(self.n)
        self._a_lin = np.zeros(shape=(self.n, 1))
        self._a_0 = np.zeros(shape=(1, 1))

        # for plotting
        self.means_batch_norm_samples = []
        self.means_batch_norm_out = []

    @property
    def r(self):
        return self._a_lin

    @property
    def R(self):
        return self._a_quad

    def get_model_params(self):
        return self._a_quad, self._a_lin

    def fit(self, data_x, data_y, **fit_args):
        raise NotImplementedError("has to be implemented in subclass")

    def preprocess_data(self, data_x, data_y, dist, **kwargs):
        raise NotImplementedError

    def postprocess_params(self, params):
        raise NotImplementedError

    def poly_feat(self, data_x):
        """Creates design matrix for polynomial regression problem.
        """
        lin_feat = data_x
        quad_feat = np.transpose((data_x[:, :, None] @ data_x[:, None, :]),
                                 [1, 2, 0])[self.square_feat_lower_tri_ind].T

        phi = np.hstack([np.ones([data_x.shape[0], 1]), lin_feat, quad_feat])

        return phi

    def normalize_features(self, phi):
        """ Batch normalization of samples, using the standard score.
        """
        phi_mean = np.mean(phi[:, 1:], axis=0, keepdims=True)
        phi_std = np.std(phi[:, 1:], axis=0, keepdims=True, ddof=1)
        phi[:, 1:] = phi[:, 1:] - phi_mean  # or only linear part? use theta_mean?
        phi[:, 1:] = phi[:, 1:] / phi_std
        self._phi_mean = phi_mean
        self._phi_std = phi_std
        self.means_batch_norm_samples.append(phi_mean)
        return phi

    def denormalize_features(self, par):
        par[1:] = par[1:] / self._phi_std.T
        par[0] = par[0] - self._phi_mean @ par[1:]
        return par

    def output_weighting(self, y):
        output_weighting = self.options.output_weighting
        if output_weighting is None or not output_weighting:
            weighting = np.ones(shape=(y.size, 1))
        elif output_weighting == "rank":
            cma_weights = RecombinationWeights(y.size * 2)
            cma_weights.finalize_negative_weights(cmu=1, dimension=y.size * 2, c1=1)
            ind = np.argsort(y.flatten())
            weighting = np.zeros(shape=y.shape)
            weighting[ind] = cma_weights.positive_weights[::-1, None]
        elif output_weighting == "min_max":
            weighting = (y - np.min(y)) / (np.max(y) - np.min(y))
        elif output_weighting == "linear":
            ind = np.argsort(y.flatten())
            weighting = np.zeros(shape=y.shape)
            weighting[ind] = np.linspace(0, 20, num=y.size)[:, None]
        else:
            raise NotImplementedError

        return weighting

    def normalize_output(self, y):
        """Different methods for normalizing the reward.
        """
        norm_type = self.options.normalize_output
        if norm_type == "mean_std":
            data_y_mean = np.mean(y)
            data_y_std = np.std(y, ddof=1)
            self._data_y_mean = data_y_mean
            new_y = (y - data_y_mean) / data_y_std

        elif norm_type == "mean_std_clipped":
            ind = np.argsort(y, axis=0)[int((1 - self.options.top_data_fraction) * len(y)):]
            top_data_y_mean = np.mean(y[ind])
            top_data_y_std = np.std(y[ind])
            new_y = (y - top_data_y_mean) / top_data_y_std
            new_y[new_y < self.options.min_clip_value] = self.options.min_clip_value

        elif norm_type == "min_max":
            new_y = (y - np.min(y)) / (np.max(y) - np.min(y))

        elif norm_type == "rank_linear":
            ind = np.argsort(y.flatten())
            new_y = np.zeros(shape=y.shape)
            new_y[ind] = np.linspace(0, 1, num=y.size)[:, None]

        elif norm_type == "rank_cma":
            cma_weights = RecombinationWeights(2 * y.size)[0:y.size]
            ind = np.argsort(y.flatten())
            new_y = np.zeros(shape=y.shape)
            new_y[ind] = ((cma_weights - np.min(cma_weights)) / (np.max(cma_weights) - np.min(cma_weights)))[::-1,
                           None]
        elif norm_type is None or not norm_type:
            new_y = y
        else:
            raise NotImplementedError

        return new_y

    def unnormalize_output(self, a_quad, a_lin, a_0):
        norm_type = self.options.normalize_output
        if norm_type == "mean_std":
            # std_mat = np.diag(self.data_y_std)
            new_a_quad = self._data_y_std * a_quad
            new_a_lin = self._data_y_std * a_lin
            new_a_0 = self._data_y_std * a_0 + self._data_y_mean
        elif norm_type == "min_max":
            new_a_quad = (self.data_y_max - self.data_y_min) * a_quad
            new_a_lin = (self.data_y_max - self.data_y_min) * a_lin
            new_a_0 = (self.data_y_max - self.data_y_min) * a_0 + self.data_y_min
        else:
            return a_quad, a_lin, a_0

        return new_a_quad, new_a_lin, new_a_0

    def whiten_input(self, x, dist):
        data_x_mean = np.mean(x, axis=0, keepdims=True)

        try:
            data_x_inv_std = np.linalg.inv(np.linalg.cholesky(np.cov(x, rowvar=False))).T
        except np.linalg.LinAlgError:
            data_x_inv_std = dist.sqrt_prec # achieves var = 1 but does not remove covariance
        finally:
            if np.any(np.isnan(data_x_inv_std)):
                data_x_inv_std = dist.sqrt_prec

        self.data_x_org = x
        self._data_x_mean = data_x_mean
        self._data_x_inv_std = data_x_inv_std
        x = x - data_x_mean
        x = x @ data_x_inv_std
        return x

    def unwhiten_params(self, a_quad, a_lin, a_0):
        int_a_quad = self._data_x_inv_std @ a_quad @ self._data_x_inv_std.T
        int_a_lin = self._data_x_inv_std @ a_lin
        a_quad = - 2 * int_a_quad  # to achieve -1/2 xMx + xm form
        a_lin = - 2 * (self._data_x_mean @ int_a_quad).T + int_a_lin
        a_0 = a_0 + self._data_x_mean @ (int_a_quad @ self._data_x_mean.T - int_a_lin)

        return a_quad, a_lin, a_0

    def refit_pos_def(self, data_x, data_y, M, weights):
        w, v = np.linalg.eig(M)
        w[w > 0.0] = -1e-8
        M = v @ np.diag(np.real(w)) @ v.T

        # refit quadratic
        aux = data_y - np.einsum('nk,kh,nh->n', data_x, M, data_x)[:, None]
        lin_feat = data_x
        phi = np.hstack([np.ones([data_x.shape[0], 1]), lin_feat])

        phi_weighted = phi * weights

        phi_t_phi = phi_weighted.T @ phi

        par = np.linalg.solve(phi_t_phi + self.options.ridge_factor * np.eye(self.n + 1),
                              phi_weighted.T @ aux)

        return par


class QuadModelLS(MoreModel):
    """ Estimate surrogate model with a form of ridge regression.
    """
    def __init__(self, dim, config_dict):
        super().__init__(dim, config_dict)

        self.dim_tri = int(self.n * (self.n + 1) / 2)

        self.model_dim = 1 + self.n + self.dim_tri

        self.model_params = np.zeros(self.model_dim)
        self.square_feat_lower_tri_ind = np.tril_indices(self.n)

        self._phi_mean = np.zeros(shape=(1, self.model_dim - 1))
        self._phi_std = np.ones(shape=(1, self.model_dim - 1))

        self.ridge_factor = self.options.ridge_factor

        self.phi = None
        self.targets = None
        self.targets_org = None
        self.weights = None
        self.scalar = None

    def preprocess_data(self, data_x, data_y, dist, imp_weights=None):
        if len(data_y.shape) < 2:
            data_y = data_y[:, None]

        data_y_org = np.copy(data_y)
        self.targets_org = data_y_org

        if imp_weights is None:
            imp_weights = np.ones(data_y_org.shape)

        self._data_y_std = np.std(data_y_org, ddof=1)

        if self._data_y_std == 0:
            return False

        if self.options.whiten_input:
            data_x = self.whiten_input(data_x, dist)
        else:
            self.data_x_mean = np.zeros((1, self.n))
            self.data_x_inv_std = np.eye(self.n)

        phi = self.poly_feat(data_x)

        if self.options.normalize_features:
            self.normalize_features(phi)

        data_y = self.normalize_output(data_y_org)

        weights = self.output_weighting(data_y_org, )

        self.targets = data_y
        self.phi = phi
        self.weights = weights

        return True

    def postprocess_params(self, par):
        if self.options.normalize_features:
            par = self.denormalize_features(par)

        a_0 = par[0]
        a_lin = par[1:self.n + 1]
        a_quad = np.zeros((self.n, self.n))
        a_tri = par[self.n + 1:].flatten()
        a_quad[self.square_feat_lower_tri_ind] = a_tri
        a_quad = 1 / 2 * (a_quad + a_quad.T)

        if self.options.whiten_input:
            a_quad, a_lin, a_0 = self.unwhiten_params(a_quad, a_lin, a_0)
        else:
            a_quad = - 2 * a_quad

        if self.options.unnormalize_output:
            a_quad, a_lin, a_0 = self.unnormalize_output(a_quad, a_lin, a_0)

        self.scalar = a_0
        return a_quad, a_lin, a_0

    def state_transition_prior(self, K, a_quad, a_lin, a_0, norm_reward):
        # TODO: add gaussian noise
        # TODO: add version for standard normalization prior
        I = np.eye(np.shape(a_quad)[0], np.shape(a_quad)[0])
        if norm_reward:
            I = (np.sqrt(2) / np.sqrt(self.n)) * I
        a_quad = K * (a_quad - I) + I
        a_lin =  K * a_lin
        a_0 = K * a_0
        return a_quad, a_lin, a_0

    def fit(self, data_x, data_y, dist=None):
        success = self.preprocess_data(data_x, data_y, dist)
        if not success:
            return False

        phi_weighted = self.phi * self.weights

        reg_mat = np.eye(self.model_dim)
        reg_mat[0, 0] = 0

        phi_t_phi = phi_weighted.T @ self.phi

        par = np.linalg.solve(phi_t_phi + self.options.ridge_factor * reg_mat, phi_weighted.T @ self.targets)

        a_quad, a_lin, a_0 = self.postprocess_params(par)

        if self.options.limit_model_opt:
            try:
                model_opt = np.linalg.solve(a_quad, a_lin)
            except:
                model_opt = np.zeros_like(a_lin)

            if np.any(np.abs(model_opt) > 20):
                return False, None, None, None, None

        # K = self.options.K
        # a_quad, a_lin, a_0 = self.state_transition_prior(K, a_quad, a_lin, a_0, True)

        self._a_quad = a_quad
        self._a_lin = a_lin
        self._a_0 = a_0

        self.model_params = np.vstack([a_quad[self.square_feat_lower_tri_ind][:, None], a_lin, a_0])

        return True, self.model_params, par, self.targets_org, self.targets


class QuadModelSubBLR(MoreModel):
    """Estimate surrogate model with Bayesian dimensionality reduction and linear regression.
    """
    def __init__(self, dim_x, config_dict):
        super().__init__(dim_x, config_dict)

        self.tau_squared = 100
        self.sigma_squared = 100
        self.sub_space_dim = 5
        self.k = 1000



        self.model_dim_lin = 1 + self.sub_space_dim
        self.model_dim_diag = self.model_dim_lin + self.sub_space_dim
        self.dim_tri = int(self.sub_space_dim * (self.sub_space_dim + 1) / 2)
        self.model_dim_full = 1 + self.sub_space_dim + self.dim_tri

        self.model_dim = 1 + self.sub_space_dim + self.dim_tri

        self.beta_prior_prec = 1 / self.tau_squared * np.eye(self.model_dim_full)
        # self.beta_prior_prec[0, 0] = 1e-10

        self.beta_prior_cov = self.tau_squared * np.eye(self.model_dim_full)
        # self.beta_prior_cov[0, 0] = 1e10

        self.square_feat_lower_tri_ind = np.tril_indices(self.sub_space_dim)

        self.data_x = None
        self.data_y = None
        self.weights = None

        self._a_quad = np.eye(self.n)
        self._a_lin = np.zeros(shape=(self.n, 1))
        self._a_0 = np.zeros(shape=(1, 1))

    def beta_post(self, phi):
        phi_weighted = phi * self.weights

        phi_t_phi = phi_weighted.T @ phi

        beta_mat = 1 / self.sigma_squared * phi_t_phi + self.beta_prior_prec

        mu_beta = np.linalg.solve(beta_mat, 1 / self.sigma_squared * phi_weighted.T @ self.data_y)

        return mu_beta

    def p_d_w(self, phi, log=True):
        cov_p_d_w = self.sigma_squared * np.diag(np.abs(self.data_y).flatten()) + phi @ self.beta_prior_cov @ phi.T
        # cov_p_d_w = self.sigma_squared * np.eye(len(self.data_y)) + phi @ self.beta_prior_cov @ phi.T

        chol_cov = np.linalg.cholesky(cov_p_d_w)
        log_det = 2 * np.sum(np.log(np.diag(chol_cov)))

        quad_term_half = np.linalg.solve(chol_cov, self.data_y)

        likelihood = -0.5 * (log_det + quad_term_half.T @ quad_term_half + len(self.data_y) * np.log(2 * np.pi))

        if log:
            return likelihood.flatten()
        else:
            return np.exp(likelihood).flatten()

    def create_sub_space_model(self, _):
        w = np.random.randn(self.n, self.sub_space_dim)

        phi = self.poly_feat(self.data_x @ w)
        if self.options.normalize_features:
            self.normalize_features(phi)

        mu_beta = self.beta_post(phi)

        if self.options.normalize_features:
            mu_beta[1:] = mu_beta[1:] / self._phi_std.T
            mu_beta[0] = mu_beta[0] - self._phi_mean @ mu_beta[1:]

        p_d_w_i_log = self.p_d_w(phi, log=True)

        a_0 = mu_beta[0]

        a_lin = mu_beta[1:self.sub_space_dim + 1]

        a_quad_sub = np.zeros((self.sub_space_dim, self.sub_space_dim))
        a_tri = mu_beta[self.sub_space_dim + 1:].flatten()
        a_quad_sub[self.square_feat_lower_tri_ind] = a_tri
        a_quad_sub = 1 / 2 * (a_quad_sub + a_quad_sub.T)

        return w, p_d_w_i_log, a_0, a_lin, a_quad_sub

    def fit(self, data_x, data_y, dist=None, imp_weights=None, objective=None):

        if len(data_x) < 0.1 * self.model_dim_full:
            return False

        if self._data_y_std == 0:
            return False

        # data_y_org = np.copy(data_y)
        # data_y = self.normalize_output(data_y_org)

        self.data_x = data_x
        self.data_y = data_y
        self.weights = 1 / np.abs(data_y)

        w_all, p_d_w_all_log, a_0_all, a_lin_all, a_quad_sub_all = zip(*Parallel(n_jobs=8)(delayed(self.create_sub_space_model)(i) for i in range(self.k)))

        p_max = np.max(p_d_w_all_log)
        exp = [np.exp(p - p_max) for p in p_d_w_all_log]
        log_p_d = p_max + np.log(np.sum(exp)) - np.log(self.k)

        imp_weights = np.exp([pdwi - log_p_d for pdwi in p_d_w_all_log]) #/ self.k
        a_quad = np.mean([w_i @ a_q_i @ w_i.T * iw_i for a_q_i, w_i, iw_i in zip(a_quad_sub_all, w_all, imp_weights)],
                         axis=0)

        a_lin = np.mean([w_i @ a_l_i * iw_i for a_l_i, w_i, iw_i in zip(a_lin_all, w_all, imp_weights)],
                         axis=0)

        a_0 = np.mean([a_0_i * iw_i for a_0_i, iw_i in zip(a_0_all, imp_weights)],
                         axis=0)

        a_quad = -2 * a_quad

        self._a_quad = a_quad
        self._a_lin = a_lin
        self._a_0 = a_0

        self.model_params = np.vstack([a_quad[self.square_feat_lower_tri_ind][:, None], a_lin, a_0])

        return True


class QuadModelRLS(MoreModel):
    """
    Recursive Least Squares for parameter estimation of quadratic surrogate model.

    """
    def __init__(self, dim, config_dict, logger=None):
        super().__init__(dim, config_dict)

        self.logger = logger
        self.dim_tri = int(self.n * (self.n + 1) / 2)

        self.model_dim = 1 + self.n + self.dim_tri

        self.model_params = np.zeros(self.model_dim)
        self.square_feat_lower_tri_ind = np.tril_indices(self.n)

        self.delta = self.options.delta
        self.std = self.options.std

        self.cov = self.options.cov * np.eye(self.model_dim)
        self.iteration = 0
        self.final_step = False
        self.rls = RLSDriftModel(self.delta, self.model_dim, self.dim_tri, self.options.normalize_output)

        self.running_mean_std_feat = RunningMeanStd()
        self.running_mean_std_out = RunningMeanStd()
        self.moving_average_feat = MovingMeanVar(self.model_dim - 1, self.options.alpha)
        self.moving_average_output = MovingMeanVar(1, self.options.alpha)
        self.out_buffer = []
        self.phi = None
        self.targets = None
        self.targets_org = None
        self.weights = None

        # these variables are used for plotting
        self.out_mean = None
        self.out_var = None
        self.out_mean_pool = None
        POOL_SIZE = 527 # used for plotting mean of certain poolsize, for comparison of different normalization methods
        self.out_buffer = deque(maxlen=POOL_SIZE)
        self.feat_means_running = []
        self.feat_means_moving = []
        self.out_means_running = []
        self.out_means_moving = []
        self.out_means_iter = []
        self.out_means_all = []
        self.out_norm_moving = []
        self.out_norm_running = []
        self.s_scalars = []
        self.p_matrix = []        


    def normalize_features_running(self, phi) -> np.ndarray:
        """Online calculation of standard score, this is the same
           as having one large sample buffer and normalizing with
           the mean and covariance of that buffer
        """
        if self.iteration == 0:
            # warm start
            self.running_mean_std_feat.update(phi[:, 1:])
        else:
            new_data = phi[-self.options.spi:, 1:]
            self.running_mean_std_feat.update(new_data)

        mean = self.running_mean_std_feat.mean
        self.feat_means_running.append(mean)
        var = self.running_mean_std_feat.var
        phi[:, 1:] = phi[:, 1:] - np.atleast_2d(mean)
        phi[:, 1:] = phi[:, 1:] / np.sqrt(np.atleast_2d(var))
        # for denormalization
        self._phi_mean = np.atleast_2d(mean)
        self._phi_std = np.sqrt(np.atleast_2d(var))
        return phi


    def normalize_features_moving(self, phi) -> np.ndarray:
        """Using a exponential moving average for normalization.
        """
        if self.iteration == 0:
            # warm start
            for i in range(np.shape(phi)[0]):
                self.moving_average_feat.update(phi[i, 1:])
        else:
            new_data = phi[-self.options.spi:, 1:]
            for i in range(self.options.spi):
                self.moving_average_feat.update(new_data[i, :])

        mean = self.moving_average_feat.mean
        self.feat_means_moving.append(mean)
        var = self.moving_average_feat.var
        phi[:, 1:] = phi[:, 1:] - np.atleast_2d(mean)
        phi[:, 1:] = phi[:, 1:] / np.sqrt(np.atleast_2d(var))
        # for denormalization
        self._phi_mean = np.atleast_2d(mean)
        self._phi_std = np.sqrt(np.atleast_2d(var))
        return phi


    def normalize_output_running(self, data_y) -> np.ndarray:
        """online calculation of standard score, this is the same
           as having one large buffer and normalizing with
           the mean and covariance of that buffer
        """
        if self.iteration == 0:
            # warm start
            self.running_mean_std_out.update(data_y)
        else:
            new_data = data_y[-self.options.spi:]
            self.running_mean_std_out.update(new_data)


        mean = self.running_mean_std_out.mean
        self.out_means_running.append(mean)
        var = self.running_mean_std_out.var
        self.out_norm_running.extend( (data_y - mean) / np.sqrt(var))
        return (data_y - mean) / np.sqrt(var)

    def normalize_output_moving(self, data_y) -> np.ndarray:
        """Normalize rewards with exponential moving mean.

        In the first iteration all rewards from warm start are processed,
        than in each following iteration only the new samples
        are used for the exponential moving mean calculation.
        """
        # TODO implement clipping here
        
        if self.iteration == 0:
            # warm start
            for i in range(np.shape(data_y)[0]):
                self.moving_average_output.update(data_y[i])
        else:
            new_data = data_y[-self.options.spi:]
            for i in range(np.shape(new_data)[0]):
                self.moving_average_output.update(new_data[i])

        mean = self.moving_average_output.mean
        var = self.moving_average_output.var

        # for logging
        self.out_means_moving.append(mean)
        self.out_mean = mean
        self.out_var = var
        self.out_mean_pool = np.mean(self.out_buffer)
        self.out_var_pool = np.var(self.out_buffer)

        self.out_norm_moving.extend( (data_y - mean) / np.sqrt(var))
        return (data_y - mean) / np.sqrt(var)

    def normalize_output_moving_clipped(self, data_y) -> np.ndarray:
        """Normalize rewards with exponential moving mean.

        In the first iteration all rewards from warm start are processed,
        than in each following iteration only the new samples
        are used for the exponential moving mean calculation.
        """
        data_y_org = data_y
        # take only top 50% (example value) of data for mean
        ind = np.argsort(data_y_org, axis=0)[int((1 - self.options.top_data_fraction) * len(data_y_org)):]
        data_y = data_y[ind]
        for i in range(np.shape(data_y)[0]):
            self.moving_average_output.update(data_y[i])
        top_data_y_mean = self.moving_average_output.mean
        top_data_y_std = np.sqrt(self.moving_average_output.var)
        new_y = (data_y_org - top_data_y_mean) / top_data_y_std
        new_y[new_y < self.options.min_clip_value] = self.options.min_clip_value
        return new_y


    def preprocess_data(self, data_x, data_y, dist, **kwargs):
        if len(data_y.shape) < 2:
            data_y = data_y[:, None]

        data_y_org = np.copy(data_y)
        self.targets_org = data_y_org
        self.out_buffer.extend(data_y)
        self.out_means_all.append(np.mean(self.out_buffer))
        weights = np.ones_like(data_y)  # may weight with absolute values of reward

        if self.options.whiten_input:
            data_x = self.whiten_input(data_x, dist)
        else:
            self.data_x_mean = np.zeros((1, self.n))
            self.data_x_inv_std = np.eye(self.n)

        phi = self.poly_feat(data_x)

        if self.options.norm_feat == "running":
            phi = self.normalize_features_running(phi)
        elif self.options.norm_feat == "moving":
            phi = self.normalize_features_moving(phi)
        elif self.options.norm_feat == "batch":
            phi = self.normalize_features(phi)

        if self.options.normalize_output == "running":
            data_y = self.normalize_output_running(data_y_org)
        elif self.options.normalize_output == "moving":
            data_y = self.normalize_output_moving(data_y_org)
        elif self.options.normalize_output == "moving_clipped":
            data_y = self.normalize_output_moving_clipped(data_y_org)            
        else:
            self.normalize_output_moving(data_y_org) # for plotting
            data_y = self.normalize_output(data_y_org)

        self.targets = data_y
        self.phi = phi
        self.weights = weights

        return True


    def unwhiten_params(self, a_quad, a_lin, a_0):
        int_a_quad = self._data_x_inv_std @ a_quad @ self._data_x_inv_std.T
        int_a_lin = self._data_x_inv_std @ a_lin
        a_quad = int_a_quad
        a_lin = (self._data_x_mean @ int_a_quad).T + int_a_lin
        a_0 = a_0 + self._data_x_mean @ (int_a_quad @ self._data_x_mean.T - int_a_lin)

        return a_quad, a_lin, a_0


    def postprocess_params(self, par):
        if self.options.unnorm_feat:
            par = self.denormalize_features(par)
        a_0 = par[0]
        a_lin = par[1:self.n + 1]
        a_quad = np.zeros((self.n, self.n))
        a_tri = par[self.n + 1:].flatten()
        a_quad[self.square_feat_lower_tri_ind] = a_tri
        # a_quad = -1 * (a_quad + a_quad.T)
        a_quad = -1/2 * (a_quad + a_quad.T)
        if self.options.whiten_input:
            a_quad, a_lin, a_0 = self.unwhiten_params(a_quad, a_lin, a_0)

        if self.options.unnormalize_output:
            a_quad, a_lin, a_0 = self.unnormalize_output(a_quad, a_lin, a_0)
        return a_quad, a_lin, a_0

    def fit(self, data_x, data_y, dist=None, counter=None):
        success = self.preprocess_data(data_x, data_y, dist)
        if not success:
            return False

        # process each sample separably
        for i in range(len(data_y)):
            phi_i = self.phi[i,:].reshape(1, self.model_dim)
            targets_i = self.targets[i]

            if i == len(data_y)-1:
                self.final_step = True

            if counter is not None:
                self.rls.step(
                    phi_i, targets_i, self.std, self.cov, self.options.weighting, self.options.K, self.final_step, count=counter[i])
            else:
                self.rls.step(phi_i, targets_i, self.std, self.cov, self.options.K, self.final_step)

            self.final_step = False

            # for plotting
            self.s_scalars.append(self.rls.s.item())
            self.p_matrix.append(self.rls.p_old.diagonal().tolist())


        self.iteration += 1
        a_quad, a_lin, a_0 = self.postprocess_params(self.rls.m_old)
        old_quad = a_quad

        self._a_quad = a_quad
        self._a_lin = a_lin
        self._a_0 = a_0

        self.model_params = np.vstack([a_quad[self.square_feat_lower_tri_ind][:, None], a_lin, a_0])

        return True, self.model_params, self.rls.m_old, self.targets_org, self.targets


class QuadModelKF(MoreModel):
    """
    Kalman filter for parameter estimation of quadratic surrogate model.

    """
    def __init__(self, dim, config_dict):
        super().__init__(dim, config_dict)
        self.dim = dim
        self.dim_tri = int(self.n * (self.n + 1) / 2)
        self.model_dim = 1 + self.n + self.dim_tri

        self.model_params = np.zeros((self.model_dim, 1))
        self.model_params_prev = np.zeros((self.model_dim, 1))

        self.y_prev = 0

        self.square_feat_lower_tri_ind = np.tril_indices(self.n)

        self.phi = None
        self.targets = None
        self.weights = None

        self.kf = KalmanFilter(self.options.m0, self.options.P0)

    def preprocess_data(self, data_x, data_y, dist, **kwargs):
        if len(data_y.shape) < 2:
            data_y = data_y[:, None]

        weights = np.ones_like(data_y)  # may weight with absolute values of reward

        if self.options.whiten_input:
            data_x = self.whiten_input(data_x, dist)
        else:
            self.data_x_mean = np.zeros((1, self.n))
            self.data_x_inv_std = np.eye(self.n)

        phi = self.poly_feat(data_x)

        if self.options.normalize_output:
            data_y = self.normalize_output(data_y)

        self.targets = data_y
        self.phi = phi
        self.weights = weights

        return True

    # Overwrite for different coefficients for the surrogate parameters
    def unwhiten_params(self, a_quad, a_lin, a_0):
        int_a_quad = self._data_x_inv_std @ a_quad @ self._data_x_inv_std.T
        int_a_lin = self._data_x_inv_std @ a_lin
        a_quad = int_a_quad
        a_lin = (self._data_x_mean @ int_a_quad).T + int_a_lin
        a_0 = a_0 + self._data_x_mean @ (int_a_quad @ self._data_x_mean.T - int_a_lin)

        return a_quad, a_lin, a_0

    def postprocess_params(self, par):
        a_0 = par[0]
        a_lin = par[1:self.n + 1]
        a_quad = np.zeros((self.n, self.n))
        a_tri = par[self.n + 1:].flatten()
        a_quad[self.square_feat_lower_tri_ind] = a_tri
        a_quad = -1 * (a_quad + a_quad.T)

        if self.options.whiten_input:
            a_quad, a_lin, a_0 = self.unwhiten_params(a_quad, a_lin, a_0)

        if self.options.unnormalize_output:
            a_quad, a_lin, a_0 = self.unnormalize_output(a_quad, a_lin, a_0)
        return a_quad, a_lin, a_0



    def fit(self, data_x, data_y, dist=None):

        success = self.preprocess_data(data_x, data_y, dist)
        if not success:
            return False

        # process each sample separably
        for i in range(len(data_y)):

            y = self.targets[i]
            phi_i = self.phi[i,:].reshape(1, self.model_dim)
            H = phi_i


            A = np.eye(self.model_dim, self.model_dim) # same as RLS

            Q = self.options.Q # covariance for parameters
            R = self.options.R # covariance for measurements

            self.kf.predict(A, Q)
            self.kf.update(H, R, y)


        a_quad, a_lin, a_0 = self.postprocess_params(self.kf.m_old)

        self._a_quad = a_quad
        self._a_lin = a_lin
        self._a_0 = a_0

        self.model_params_prev = self.model_params
        self.y_prev = y
        self.model_params = np.vstack([a_quad[self.square_feat_lower_tri_ind][:, None], a_lin, a_0])
        return True, self.model_params, self.kf.m_old
