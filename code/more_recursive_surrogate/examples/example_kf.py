import numpy as np
import matplotlib.pylab as plt
from more.gauss_full_cov import GaussFullCov
from more.quad_model import QuadModelKF
from more.more_algo import MORE
from more.sample_db import SimpleSampleDatabase
from cma.bbobbenchmarks import nfreefunclasses
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('MORE')
logger.setLevel("INFO")

dim = 5
dim_tri = int(dim * (dim + 1) / 2)
model_dim = 1 + dim + dim_tri

max_iters = 1000
max_samples = 15 # 500 for ls model
sample_pool = False
samples_per_iter = 8
kl_bound = 0.1
gamma = 0.99
entropy_loss_bound = 0.1
minimize = True
warm_start = 30

m0 = np.random.randn(model_dim,1)
P0 = 1 * np.eye(model_dim)

Q = 10 * np.eye(model_dim) # covariance for parameters
R =  10 **2 # covariance for measurements

model_options_kf = {"whiten_input": True,
                    "Q": Q, # covariance for parameters
                    "R": R, # measurement noise
                    "m0": m0,
                    "P0": P0,
                    "normalize_output": False,
                    "unnormalize_output": False,
                    "output_weighting": False}

more_config = {"epsilon": kl_bound,
               "gamma": gamma,
               "beta_0": entropy_loss_bound}


np.random.seed(1)
x_start = 0.5 * np.random.randn(dim)
init_sigma = 1

objective = nfreefunclasses[7](0, zerof=True, zerox=True)
objective.initwithsize(curshape=(1, dim), dim=dim)

sample_db = SimpleSampleDatabase(max_samples)

search_dist = GaussFullCov(x_start, init_sigma * np.eye(dim))
surrogate = QuadModelKF(dim, model_options_kf)

more = MORE(dim, more_config, logger=logger)

whiten_pars = []
model_pars = []
loss = []
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

    sample_db.add_data(new_samples, new_rewards)

    samples, rewards = sample_db.get_data()
    # do not use sample pool
    if not sample_pool:
        samples, rewards = new_samples, new_rewards     
    success, model_par, whiten_par = surrogate.fit(samples, rewards, search_dist, )
    if not success:
        continue

    whiten_pars.append(whiten_par)
    model_pars.append(model_par)
    
    new_mean, new_cov, success = more.step(search_dist, surrogate)

    if success:
        try:
            search_dist.update_params(new_mean, new_cov)
        except Exception as e:
            print(e)

    lam = objective(search_dist.mean.T)
    logger.info("Loss at mean {}".format(lam))
    logger.info("Change KL {}, Entropy {}".format(more._kl, search_dist.entropy))
    logger.info("Dist to x_opt {}".format(np.linalg.norm(objective._xopt - search_dist.mean.flatten())))
    dist_to_opt = np.abs((objective._fopt - lam))
    logger.info("Dist to f_opt {}".format(dist_to_opt))

    # logger.info("a_0 {}".format(surrogate._a_0))
    # logger.info("a_lin {}".format(surrogate._a_lin))
    # logger.info("a_quad {}".format(surrogate._a_quad))
    logger.info("-------------------------------------------------------------------------------")
    loss.append(dist_to_opt)
    if dist_to_opt < 1e-8:
        break


# plot model parameters
def plot_parameters(parameters):
    model_dim = len(parameters[0])
    x = np.linspace(0, len(parameters), len(parameters))

    for i in range(0,model_dim):
        # plt.subplot(model_dim, 2, i+1)
        # plt.subplots_adjust(hspace=0.5, wspace=0.4)
        # plt.plot(x, [y[i] for y in parameters], label=f'LS par_{i}')
        plt.plot(x, [y[i] for y in parameters], label=f'RLS par_{i}')
        plt.legend()
    # plt.plot(x, [y[0] for y in parameters], label=f'RLS par_{0}')
    plt.yscale('symlog')
        
    plt.show()

def plot_loss(loss):
    x = np.linspace(0, len(loss), len(loss))

    plt.semilogy(x, loss)
    plt.show()

plot_loss(loss)    
# plot_parameters(whiten_pars)
# plot_parameters(model_pars)
# print(whiten_pars)
# print(model_pars)
