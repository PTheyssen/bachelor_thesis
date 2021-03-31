import numpy as np
import matplotlib.pylab as plt
from more.gauss_full_cov import GaussFullCov
from more.quad_model import QuadModelRLS
from more.more_algo import MORE
from more.sample_db import SimpleSampleDatabase
from cma.bbobbenchmarks import nfreefunclasses
import util.plotting_more as plotting
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('MORE')
logger.setLevel("INFO")
# plot = True
plot = False

dim = 15
max_iters = 1000
max_samples = 100 # 500 for ls model
samples_per_iter = 7
kl_bound = 0.2
gamma = 0.99
entropy_loss_bound = 0.1
minimize = True
warm_start = 40

model_options_rls = {"whiten_input": True,
                     "norm_feat": False,
                     "unnorm_feat": False,
                     "norm_out": False,
                     "norm_type": "moving", # moving (average with weighting), running (mean of all samples), "plot"
                     "weighting": 0,
                     "unnormalize_output": False,
                     "output_weighting": False,
                     "spi": samples_per_iter,
                     "delta": 10,
                     "K": 1,
                     "cov": 1,
                     "std": 0.01,
                     "alpha": 0.99}

more_config = {"epsilon": kl_bound,
               "gamma": gamma,
               "beta_0": entropy_loss_bound}

# np.random.seed(0)
np.random.seed(12012)
# np.random.seed(2)
x_start = 1 * np.random.randn(dim)
# x_start = 1.5 * np.random.randn(dim)
# x_start = np.array([-0.8, 4])
init_sigma = 1

################################################################################
# Different objective Functions
################################################################################
# objective = nfreefunclasses[14](0, zerof=True, zerox=True) # rastigrin
# objective = nfreefunclasses[0](0, zerof=True, zerox=True) # sphere
objective = nfreefunclasses[7](0, zerof=True, zerox=True) # rosenbrock
objective.initwithsize(curshape=(1, dim), dim=dim)

sample_db = SimpleSampleDatabase(max_samples)
search_dist = GaussFullCov(x_start, init_sigma * np.eye(dim))
surrogate = QuadModelRLS(dim, model_options_rls, logger)
more = MORE(dim, more_config, logger=logger)

# Log metrics for plotting
whiten_pars = []
model_pars = []
targets_org = []
targets_norm = []
loss = []
kl = []
ent = []
means = []
unsuccessful_opt = 0
for i in range(max_iters):
    logger.info("Iteration {}".format(i))

    # warm start
    if i == 0:
        new_samples = search_dist.sample(warm_start)
        new_samples_counter = [[s, 0] for s in new_samples]
    else:
        new_samples = search_dist.sample(samples_per_iter)
        new_samples_counter = [[s, 0] for s in new_samples]

    new_rewards = objective(new_samples)
    if minimize:
        # negate, MORE maximizes, but we want to minimize
        new_rewards = -new_rewards

    sample_db.add_data_with_counter(new_samples_counter, new_rewards)
    samples, rewards, counter = sample_db.get_data_with_counter()

    success, model_par, whiten_par, t_org, t_norm = surrogate.fit(samples, rewards, search_dist, counter=counter)
    if not success:
        continue


    # track metrics for plotting
    whiten_pars.append(whiten_par)
    model_pars.append(model_par)
    means.append(search_dist.mean)
    targets_org.extend(t_org.flatten().tolist())
    targets_norm.extend(t_norm.flatten().tolist())
    
    new_mean, new_cov, success = more.step(search_dist, surrogate)

    if success:
        try:
            search_dist.update_params(new_mean, new_cov)
        except Exception as e:
            print(e)
    else:
        unsuccessful_opt += 1


    lam = objective(search_dist.mean.T)
    logger.info("Loss at mean {}".format(lam))
    logger.info("Change KL {}, Entropy {}".format(more._kl, search_dist.entropy))
    kl.append(more._kl)
    ent.append(search_dist.entropy)
    logger.info("Dist to x_opt {}".format(np.linalg.norm(objective._xopt - search_dist.mean.flatten())))
    dist_to_opt = np.abs((objective._fopt - lam))
    logger.info("Dist to f_opt {}".format(dist_to_opt))
    logger.info("-------------------------------------------------------------------------------")
    loss.append(dist_to_opt)
    if dist_to_opt < 1e-8:
        break

if plot:
    # plotting.plot(objective, samples, search_dist, model_par)
    plotting.animate_run(objective, means)

print(f"{unsuccessful_opt} times unsuccessful optimization")

if model_options_rls["norm_type"] == "plot":
    labels = ['mean of running mean_std', 'mean of moving average', 'mean of all data']
    data = [surrogate.out_means_running, surrogate.out_means_moving, surrogate.out_means_all]
    plotting.plot_data(data, labels, "Comparison of normalization methods")
    plotting.plot_data(
        [surrogate.out_norm_moving, surrogate.out_norm_running, surrogate.out_buffer],
        ['normalized with moving average', 'normalized with running', 'not normalized'],
        'Reward')

# plotting.plot_parameters(whiten_pars, "Zuerst skalar und linearer Term")
# plotting.plot_parameters(model_pars, "Zuerst quadratischer Term")
# plotting.plot_log(loss, "Loss")
# plotting.plot_(kl, "KL")
# plotting.plot_(ent, "entropy")
# plotting.plot_l(targets_org, "original reward")
# plotting.plot_l(targets_norm, "normalized reward")
