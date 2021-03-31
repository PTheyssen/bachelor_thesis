import numpy as np
import matplotlib.pylab as plt
import logging
import wandb

import util.plotting_more as plotting
from more.gauss_full_cov import GaussFullCov
from more.quad_model import QuadModelRLS, QuadModelLS, QuadModelSubBLR
from more.more_algo import MORE
from more.sample_db import SimpleSampleDatabase
from cma.bbobbenchmarks import nfreefunclasses

from alr_envs.classic_control.utils import make_viapointreacher_env, make_holereacher_env
from alr_envs.mujoco.ball_in_a_cup.utils import make_simple_env
from alr_envs.utils.dmp_async_vec_env import DmpAsyncVectorEnv

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('MORE')
logger.setLevel("INFO")

log_wandb = False
plot_custom = False

if __name__ == "__main__":
    task = "via" # "via", "hole", "cup"
    
    dim = 25    
    if task == "cup":
        dim = 15
    if task == "hole":
        dim = 30
        
    buffer_fac = 1.5
    max_samples_ = int(np.ceil((buffer_fac * (1 + dim + int(dim * (dim + 1) / 2)))))
    samples_per_iter_ = int(4 + np.floor(3 * np.log(dim)))


    # config for DMP
    allow_self_collision = True
    weights_scale = 100
    penalty = 10

    max_iters = 5
    max_samples = 150 # 500 for ls model
    no_pool = False
    samples_per_iter = 13
    kl_bound = 0.2
    gamma = 0.99
    entropy_loss_bound = 0.1
    minimize = False
    warm_start = 50
    sur = "RLS" # "RLS", "BLR", "LS"
    rls_norm = "moving_clipping"
    name = "_same_samples_in_pool"
    alpha = 0.05
    new_samples_multiple_times = True
    # np.random.seed(1221)

    n_samples = samples_per_iter # Anzahl samples die man mit einem Call ziehen will
    n_cpus = 4
    more_config = {"epsilon": kl_bound,
                   "gamma": gamma,
                   "beta_0": entropy_loss_bound}

    config = {"max_iters": max_iters,
              "max_samples": max_samples,
              "samples_per_iter": samples_per_iter,
              "kl_bound": kl_bound,
              "gamma": gamma,
              "entropy_loss_bound": entropy_loss_bound,
              "minimize": minimize,
              "warm_start": warm_start,
              "sur": sur}

    ################################################################################
    # Surrogate Model options
    ################################################################################
    model_options_rls = {"whiten_input": True,
                         "norm_feat": False, # moving, running, batch
                         "unnorm_feat": False,
                         "normalize_output": rls_norm, # "moving"
                         "top_data_fraction": 0.5,
                         "min_clip_value": -3.,
                         "weighting": 0,
                         "unnormalize_output": False,
                         "output_weighting": False,
                         "spi": samples_per_iter,
                         "delta": 10,
                         "K": 1,
                         "cov": 1,
                         "std": 0.01,
                         "alpha": alpha}

    model_options_ls = {"max_samples": max_samples,
                        "output_weighting": "rank",  # "rank",
                        "whiten_input": True,
                        "normalize_features": True,
                        # "normalize_output": "mean_std_clipped",  # "mean_std",  # "rank", "mean_std", "min_max",
                        "normalize_output": "mean_std",  # "mean_std",  # "rank", "mean_std", "min_max",
                        "top_data_fraction": 0.5,
                        "min_clip_value": -3.,
                        "K": 1,
                        "unnormalize_output": False,  # "rank",
                        "ridge_factor": 1e-12,
                        "limit_model_opt": True,
                        "refit": False,
                        "buffer_fac": 1.5,
                        "seed": None}

    model_options_sub = {"normalize_features": True,
                         "normalize_output": None,  # "mean_std",  # "mean_std_clipped",  # "rank", "mean_std", "min_max",
    }

    if sur == "RLS":
        surrogate = QuadModelRLS(dim, model_options_rls, logger)
        config = {**config, **model_options_rls}
    elif sur == "LS":
        surrogate = QuadModelLS(dim, model_options_ls)
        config = {**config, **model_options_ls}
    elif sur == "BLR":
        surrogate = QuadModelSubBLR(dim, model_options_sub)
        config = {**config, **model_options_sub}

    if log_wandb:
        wandb.init(project="example_reaching",
                   group="more",
                   config=config,
                   job_type=f'{sur}_{task}reach_{name}',
                   name="rep_0")
    ################################################################################
    ################################################################################


    x_start = 0.5 * np.random.randn(dim)
    init_sigma = 1
    sample_db = SimpleSampleDatabase(max_samples)
    search_dist = GaussFullCov(x_start, init_sigma * np.eye(dim))
    more = MORE(dim, more_config, logger=logger)

    total_samples = 0


    if task == "via":    
        # env = DmpAsyncVectorEnv([make_viapointreacher_env(i, allow_self_collision=False) for i in range(n_cpus)],
                            # n_samples=n_samples)
        env = DmpAsyncVectorEnv(
            [make_viapointreacher_env(i, allow_self_collision=allow_self_collision, weights=weights_scale, penalty=penalty) for i in range(n_cpus)],
                            n_samples=n_samples)
    if task == "hole":
        env = DmpAsyncVectorEnv([make_holereacher_env(i) for i in range(n_cpus)],
                            n_samples=n_samples)
    if task == "cup":
        env = DmpAsyncVectorEnv([make_simple_env(i) for i in range(n_cpus)],
                                n_samples=n_samples)

    # loop for running MORE
    loss = []
    lin_terms = []
    quad_terms = []
    params = []
    kalman_gains = []
    diag_of_covs = []

    for i in range(max_iters):
        # warm start
        sample_amount = samples_per_iter
        if i == 0:
            sample_amount = warm_start
        new_samples = search_dist.sample(sample_amount)
        new_samples_counter = [[s, 0] for s in new_samples]
        total_samples += sample_amount

        new_rewards = env(new_samples)[0]
        if minimize:
            # negate, MORE maximizes, but we want to minimize
            new_rewards = -new_rewards

        sample_db.add_data_with_counter(new_samples_counter, new_rewards)
        samples, rewards, counter = sample_db.get_data_with_counter()

        if new_samples_multiple_times:
            # until we have max_samples, repeat new_samples
            # vertically stack samples
            # breakpoint()
            t = new_samples
            while t.shape[0] < max_samples:
                t = np.vstack((t, new_samples))
            samples = t
            # for i in range(samples.shape[0]):
                # print(samples[i])
            # print("samples shape: \n", samples.shape)
            # breakpoint()

        # print("samples \n", samples)
        if sur == "RLS":
            success = surrogate.fit(
                samples, rewards, search_dist, counter=counter)
        else:
            success = surrogate.fit(samples, rewards, search_dist)
        if not success:
            continue

        new_mean, new_cov, success = more.step(search_dist, surrogate)

        if success:
            try:
                search_dist.update_params(new_mean, new_cov)
            except Exception as e:
                print(e)

        # empty sample pool after warm start so that no old
        # samples are used for RLS
        if i == 0 and no_pool:
            sample_db = SimpleSampleDatabase(samples_per_iter)

        lam = env(search_dist.mean.T)
        logger.info("Iteration {}".format(i))
        logger.info(f'Loss at mean: {lam}')
        logger.info('--------------------------------------------------------------------------------')
        loss.append(lam[0])
        if log_wandb:
            quad, lin = surrogate.get_model_params()
            lin_terms.append(lin.flatten().tolist())
            quad_terms.append(quad.flatten().tolist())
            params.append(search_dist.mean.T.flatten().tolist())
            if sur == "RLS":
                kalman_gains.append(
                    surrogate.rls.kalman_gain.flatten().tolist())



            results_dict = {"loss_at_mean": -lam[0].item(),
                            "total_samples": total_samples,
                            f"mean of new samples ({samples_per_iter_})": surrogate._data_y_mean,
                            "KL": more._kl,

                            "Entropy": search_dist.entropy,
                            "params": search_dist.mean.T,
                            "iteration": i}
            # if sur == "RLS":
                # results_dict[f"exp_moving mean of output (alpha:{alpha})"] = surrogate.out_mean
                # results_dict[f"mean of pool with size of heuristic ({max_samples_})"] = surrogate.out_mean_pool
                # results_dict["exp_moving var"] = surrogate.out_var
            wandb.log(results_dict)

    def log_multiline(wandb, data, name):
        xs = [i for i in range(len(data))]
        ys = []
        keys = []
        for i in range(len(data[0])):
            ys.append([x[i] for x in data])
            keys.append(f'{name}_{i}')
        wandb.log({name: wandb.plot.line_series(
            xs=xs,
            ys=ys,
            keys=keys,
            title=name,
            xname="iteration")})            

    # Plot surrogate parameters with wandb
    # Plot linear term of surrogate model
    if plot_custom:
        # data from normalization
        # if sur == "RLS":
            # if model_options_rls["normalize_output"]:
                # log_multiline(
                    # wandb, surrogate.feat_means_moving, "means samples (moving)")
                # log_multiline(
                    # wandb, surrogate.out_means_moving, "means rewards (moving)")
                # log_multiline(
                    # wandb, surrogate.means_batch_norm_samples, "batch means of samples")
        # data about surrogate model
        log_multiline(wandb, lin_terms, "lin_term")
        log_multiline(wandb, quad_terms, "quad_term")
        log_multiline(wandb, params, "search_dist mean")
        if sur == "RLS":
            log_multiline(wandb, kalman_gains, "kalman gains")
            # log_multiline(wandb, surrogate.p_matrix, "p_matrix diagonal")
            x_values = range(len(surrogate.s_scalars))
            y_values = surrogate.s_scalars
            data = [[x, y] for (x, y) in zip(x_values, y_values)]
            table = wandb.Table(data=data, columns = ["x", "y"])
            wandb.log({"my_custom_plot_id" : wandb.plot.line(table, "x", "y",
                                        title="s scalars")})

    
    print(f'Final Parameters: \n {search_dist.mean.T}')
