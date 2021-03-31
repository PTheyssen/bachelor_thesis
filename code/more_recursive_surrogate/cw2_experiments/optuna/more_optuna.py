import numpy as np
import matplotlib.pyplot as plt

import cw2.cluster_work
import cw2.cw_data.cw_wandb_logger
import cw2.experiment

from cma.bbobbenchmarks import nfreefunclasses
from more.more_algo import MORE
from more.gauss_full_cov import GaussFullCov
from more.quad_model import QuadModelRLS, QuadModelLS, QuadModelSubBLR
from more.sample_db import SimpleSampleDatabase
from alr_envs.classic_control.utils import make_viapointreacher_env
from alr_envs.utils.dmp_async_vec_env import DmpAsyncVectorEnv

class CWMORE(cw2.experiment.AbstractIterativeExperiment):
    """Implementation of MORE for running clusterwork 2 Experiments.

    Making it possible to use yaml configuration for running experiments
    and doing grid search for hyperparameters.
    """
    def __init__(self):
        super().__init__()
        self.problem = None
        self.optimizer = None
        self.search_dist = None
        self.surrogate = None
        self.sample_db = None
        self.total_samples = 0
        self.samples_per_iter = None
        self.pool_size = None
        self.warm_start = None
        self.lin_terms = []
        self.quad_terms = []
        self.params = []


    def init_problem(self, config: dict) -> None:
        """Initialize the problem, either a 2D reaching task or optimization
           test function.

        Args:
            config (dict) : dict containing parameters for configuration
        """
        problem = config["params"]["problem"]
        n_samples = self.samples_per_iter
        n_cpus = 16 # reaching task uses parallelization

        if problem == "via":
            self.problem = DmpAsyncVectorEnv([make_viapointreacher_env(i) for i in range(n_cpus)],
                                    n_samples=n_samples)
        elif problem == "hole":
            self.problem = DmpAsyncVectorEnv([make_holereacher_env(i) for i in range(n_cpus)],
                                    n_samples)
        elif problem == "test_func":
            dim = config["params"]["dim"]
            self.problem = nfreefunclasses[config["params"]["objective"]](0, zerof=True, zerox=True)
            self.problem.initwithsize(curshape=(1, dim), dim=dim)


    def init_more(self, config: dict) -> None:
        """Initializes the MORE algorithm.

        When optimizing parameters we have to update the corresponding values.
        Args:
            config (dict) : configuration from yaml file
        """
        more_config = {"epsilon": config["params"]["kl_bound"],
                       "dim": config["params"]["dim"],
                       "gamma": config["params"]["gamma"],
                       "beta_0": config["params"]["entropy_loss_bound"],
                       "minimize": config["params"]["minimize"]}

        self.optimizer = MORE(config["params"]["dim"], more_config, logger=None)
        x_start = 0.5 * np.random.randn(config["params"]["dim"])
        init_sigma = 1
        self.search_dist = GaussFullCov(
            x_start, init_sigma * np.eye(config["params"]["dim"]))


    def init_surrogate(self, config: dict):
        """Initializes the surrogate model.

        When optimizing parameters (for example with grid search) we have to
        update the corresponding values in the options for the surrogate models.

        Args:
            config (dict) : configuration from yaml file
        """
        sur = config["params"]["surrogate"]
        if sur == "BLR":
            blr_options = {"normalize_features": True,
                           "normalize_output": None}
            self.surrogate = QuadModelSubBLR(config["params"]["dim"], blr_options)
        elif sur == "LS":
            buffer_fac = 1.5
            ls_options = {"max_samples": self.pool_size,
                          "output_weighting": "rank",  # "rank",
                          "whiten_input": True,
                          "normalize_features": True,
                          "normalize_output": config["params"]["normalize_output"],
                          # "mean_std", "mean_std_clipped",  # "rank", "mean_std", "min_max",
                          "top_data_fraction": 0.5,
                          "min_clip_value": -3.,
                          "unnormalize_output": False,  # "rank",
                          "ridge_factor": 1e-12,
                          "limit_model_opt": True,
                          "refit": False,
                          "buffer_fac": buffer_fac,
                          "seed": None}
            self.surrogate = QuadModelLS(config["params"]["dim"], ls_options)
        elif sur == "RLS":
            rls_options = {"whiten_input": config["params"]["whiten"],
                           "norm_feat": False,
                           "unnorm_feat": False,
                           "norm_out": False,
                           "norm_type": "moving", # moving (average with weighting), running (mean of all samples)
                           "weighting": config["params"]["weighting"],
                           "cov": config["params"]["cov"],
                           "std": config["params"]["std"],
                           "delta": config["params"]["delta"],
                           "alpha": config["params"]["alpha"],
                           "spi": config["params"]["samples_per_iter"],
                           "K": config["params"]["K"],
                           "unnormalize_output": False,
                           "output_weighting": False}
            self.surrogate = QuadModelRLS(config["params"]["dim"], rls_options)


    def initialize(self, config: dict, rep: int, logger: cw2.cw_data.cw_pd_logger.PandasLogger) -> None:
        """Initialize the MORE algorithm, surrogate model and problem.

        The MORE algorithm has a search distribution that has to be initialized and
        a surrogate model, which uses a sample database.

        Args:
            config (dict) : configuration dict from yaml file
        """
        self.warm_start = config["params"]["warm_start"]
        self.pool_size = config["params"]["sample_pool"]
        self.sample_db = SimpleSampleDatabase(self.pool_size)

        if config["params"]["seed"]:
            np.random.seed(12312)

        self.samples_per_iter = config["params"]["samples_per_iter"]

        self.init_problem(config)
        self.init_more(config)
        self.init_surrogate(config)


    def generate_data(self, config: dict, n: int):
        """Generate samples and reward for one MORE iteration.

        Returns:
            samples (np.ndarrray)
            reward (np.ndarray)
            counter (list) : indicates how many times a sample was used
        """
        amount = self.samples_per_iter
        if n == 0:
            amount = self.warm_start
        samples = self.search_dist.sample(amount)
        new_samples_counter = [[s, 0] for s in samples]
        self.total_samples += amount

        # different behaviour for test functions and reaching tasks
        if config["params"]["problem"] != "test_func":
            rewards = self.problem(samples)[0]
        else:
            rewards = self.problem(samples)
        if config["params"]["minimize"]:
            rewards = -rewards

        self.sample_db.add_data_with_counter(new_samples_counter, rewards)
        return self.sample_db.get_data_with_counter()


    def iterate(self, config: dict, rep: int, n: int) -> dict:
        """Do one iteration of MORE algorithm.

        Draw samples from search distribution, evaluate the samples
        on the problem and get corresponding reward. Use the samples and
        reward to estimate surrogate model. Do one MORE step, using the
        surrogate model to solve the optimization problem and updating
        the search distribution.

        Args:
            config (dict) : configuration dict from yaml file
            rep (int) : current repition of the experiment
            n (int) : current iteration of MORE algorithm

        Returns:
            (dict) : results (for example loss) from this iteration
        """
        samples, rewards, counter = self.generate_data(config, n)

        if config["params"]["surrogate"] == "RLS":
            success = self.surrogate.fit(samples, rewards, self.search_dist, counter)
        else:
            success = self.surrogate.fit(samples, rewards, self.search_dist)
        if not success:
            return {"success": False}

        new_mean, new_cov, success = self.optimizer.step(self.search_dist, self.surrogate)
        if success:
            try:
                self.search_dist.update_params(new_mean, new_cov)
            except Exception as E:
                print(E)

        lam = self.problem(self.search_dist.mean.T)
        # different reward for test functions and reaching tasks
        if config["params"]["problem"] == "test_func":
            lam = np.abs((self.problem._fopt - lam))
        else:
            lam = lam[0].item()

        results_dict = {"loss_at_mean": lam,
                        "kl": self.optimizer._kl,
                        "parameter": self.search_dist.mean.T,
                        "entropy": self.search_dist.entropy,
                        "total_samples": self.total_samples,}
        return results_dict


    def save_state(self, config: dict, rep: int, n: int) -> None:
        pass


    def finalize(self, surrender = None, crash: bool = False) -> dict:
        pass


if __name__ == "__main__":
    import sys
    from optuna_work.experiment_wrappers import wrap_iterative_experiment
    sys.argv.append("rls_rosenbrock.yml")
    
    cw = cw2.cluster_work.ClusterWork(
        wrap_iterative_experiment(
            CWMORE, loss_key="loss_at_mean", loss_mode="last"))
    
    # cw.add_logger(cw2.cw_data.cw_wandb_logger.WandBLogger())
    cw.run()
