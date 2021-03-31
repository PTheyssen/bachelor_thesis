import numpy as np
import matplotlib.pylab as plt
from more.gauss_full_cov import GaussFullCov
from more.quad_model import QuadModelLS, QuadModelSubBLR
from more.more_algo import MORE
from more.sample_db import SimpleSampleDatabase
from cma.bbobbenchmarks import nfreefunclasses
import logging
import util.plot_surrogate_models as plot_sur


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('MORE')
logger.setLevel("INFO")

if __name__ == "__main__":
    plot_contour = True
    
    dim = 2
    max_iters = 20
    kl_bound = 1
    gamma = 0.99
    entropy_loss_bound = 0.1
    minimize = True
    warm_start = 2

    buffer_fac = 1.5
    max_samples = int(np.ceil((buffer_fac * (1 + dim + int(dim * (dim + 1) / 2)))))
    samples_per_iter = int(4 + np.floor(3 * np.log(dim)))

    # specify samples config manually
    max_samples = 10
    samples_per_iter = 10
    np.random.seed(1221)
    whiten = True
    
    model_options_ls = {"max_samples": max_samples,
                        "output_weighting": "rank",  # "rank",
                        "whiten_input": whiten,
                        "normalize_features": False,
                        # "normalize_output": "mean_std_clipped",  # "mean_std",  # "rank", "mean_std", "min_max",
                        "normalize_output": "mean_std_clipped",  # "mean_std",  # "rank", "mean_std", "min_max",                        
                        "top_data_fraction": 0.5,
                        "min_clip_value": -3.,
                        "K": 1,
                        "unnormalize_output": False,  # "rank",
                        "ridge_factor": 1e-12,
                        "limit_model_opt": True,
                        "refit": False,
                        "buffer_fac": buffer_fac,
                        "seed": None}

    more_config = {"epsilon": kl_bound,
                   "gamma": gamma,
                   "beta_0": entropy_loss_bound,
                   "n_samples": samples_per_iter,
                   "max_samples": max_samples,
                   # "buffer_fac": buffer_fac,
                   "min_data_fraction": 0.5
                   }


    x_start = 0.3 * np.random.randn(dim)
    init_sigma = 1

    # borrowing Rosenbrock from the cma package
    # objective = nfreefunclasses[2](0, zerof=True, zerox=True) # rastigrin    
    objective = nfreefunclasses[7](0, zerof=True, zerox=True) # rosenbrock
    objective.initwithsize(curshape=(1, dim), dim=dim)

    sample_db = SimpleSampleDatabase(max_samples)

    search_dist = GaussFullCov(x_start, init_sigma * np.eye(dim))
    surrogate = QuadModelLS(dim, model_options_ls)

    more = MORE(dim, more_config, logger=logger)

    # for plotting
    whiten_pars = []
    model_pars = []
    targets_org = []
    targets_norm = []
    plot_samples = []
    for i in range(max_iters):
        logger.info("Iteration {}".format(i))

        # warm start
        if i == 0:
            new_samples = search_dist.sample(warm_start)
        else:
            new_samples = search_dist.sample(samples_per_iter)

        new_rewards = objective(new_samples)
            
        if minimize:
            # negate, MORE maximizes, but we want to minimize
            new_rewards = -new_rewards

        if i > 0:
            prev_samples, p_rewards = sample_db.get_data()
        sample_db.add_data(new_samples, new_rewards)

        samples, rewards = sample_db.get_data()

        success, model_par, whiten_par, t_org, t_norm = surrogate.fit(samples, rewards, search_dist, )
        if not success:
            continue

        # track parameters for plotting
        for j in range(surrogate.phi.shape[0]):
            plot_samples.append(surrogate.phi[j])
        whiten_pars.append(whiten_par)
        model_pars.append(model_par)
        targets_org.extend(t_org.flatten().tolist())
        targets_norm.extend(t_norm.flatten().tolist())        
        
        new_mean, new_cov, success = more.step(search_dist, surrogate)

        if success:
            try:
                search_dist.update_params(new_mean, new_cov)
            except Exception as e:
                print(e)


        if plot_contour and i in [3,4]:
        # if plot_contour and i == 1:
            plot_sur.plot_surrogate(prev_samples, rewards, search_dist,
                                    surrogate.model_params, 2, 2)


        lam = objective(search_dist.mean.T)
        logger.info("Loss at mean {}".format(lam))
        logger.info("Change KL {}, Entropy {}".format(more._kl, search_dist.entropy))
        logger.info("Dist to x_opt {}".format(np.linalg.norm(objective._xopt - search_dist.mean.flatten())))

        dist_to_opt = np.abs((objective._fopt - lam))
        logger.info("Dist to f_opt {}".format(dist_to_opt))
        logger.info("-------------------------------------------------------------------------------")


        if dist_to_opt < 1e-8:
            break

        


################################################################################
# plotting
################################################################################
def plot_parameters(parameters):
    model_dim = len(parameters[0])
    x = np.linspace(0, len(parameters), len(parameters))

    for i in range(0,model_dim):
        plt.plot(x, [y[i] for y in parameters], label=f'LS par_{i}')
        plt.legend()
    # plt.yscale('symlog')
    plt.title("Test")
    # plt.show()

def plot_log(targets):
    x = np.linspace(0, len(targets), len(targets))

    plt.plot(x, targets)
    plt.yscale('symlog')    
    plt.show()

# plot_parameters(whiten_pars)
# plot_parameters(model_pars)
# plot_parameters(plot_samples)
# plt.show()
# import tikzplotlib
# tikzplotlib.save(f"LS_unwhitened.tex")
# plot_parameters(model_pars)

# plot_log(targets_org)
# plot_log(targets_norm)
