import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.cm import inferno as colormap
from matplotlib.cm import ocean as model_cm
from matplotlib.colors import LogNorm
import matplotlib.animation as animation

def plot_parameters(parameters, title):
    model_dim = len(parameters[0])
    x = np.linspace(0, len(parameters), len(parameters))

    for i in range(0,model_dim):
        plt.plot(x, [y[i] for y in parameters], label=f'RLS par_{i}')
    plt.yscale('symlog')
    plt.title(title)
    plt.legend()
    plt.show()

def plot_data(data, labels, title):
    x = np.linspace(0, len(data[0]), len(data[0]))
    for i in range(len(data)):
        plt.plot(x, data[i], label=f'{labels[i]}')
    plt.yscale('symlog')
    plt.title(title)
    plt.legend()
    plt.show()

def plot_log(metric, title=""):
    x = np.linspace(0, len(metric), len(metric))
    plt.semilogy(x, metric)
    plt.title(title)
    plt.legend()
    plt.show()

def plot_l(metric, title=""):
    x = np.linspace(0, len(metric), len(metric))
    plt.plot(x, metric)
    plt.yscale('symlog')
    plt.title(title)
    plt.legend()
    plt.show()

def plot_(metric, title=""):
    x = np.linspace(0, len(metric), len(metric))
    plt.plot(x, metric)
    plt.title(title)
    plt.legend()
    plt.show()


def calc_sur(x, model_par):
    """Calculates the scalar values for a surrogate model.
    """
    A = np.array([[model_par[0,0], model_par[1,0]],
                  [model_par[1,0], model_par[2,0]]])
    b = np.array([[model_par[3,0]],
                  [model_par[4,0]]])
    c = model_par[5,0]
    return x.T @ A @ x + x.T @ b + c

def plot(objective, samples, search_dist, model_par) -> None:
    """Plotting objective function and search distribution of MORE algorithm,
       at one iteration step.

       Args:
         objective: function that is optimized
         samples: samples from search distribution of the current iteration
         search_dist: new search distribution after this iteration
    """
    fig = plt.figure()
    fig.clf()
    ax = Axes3D(fig, azim=-128.0, elev=43.0)

    s = 0.05
    x = np.arange(-3, 3 + s, s)
    y = np.arange(-3, 3 + s, s)
    X, Y = np.meshgrid(x, y)
    Z = (1.0 - X)**2 + 100.0 * (Y - X*X)**2
    
    obj = np.zeros((len(x), len(x)))
    for i in range(len(x)):
        for j in range(len(x)):
            obj[i,j] = objective(np.array([X[i,j], Y[i,j]]))

    sur = np.zeros((len(x), len(x)))
    for i in range(len(x)):
        for j in range(len(x)):
            sur[i,j] = calc_sur(np.array([X[i,j], Y[i,j]]), model_par)

    # objective function
    ax.plot_surface(X, Y, obj, rstride=1, cstride=1, norm=LogNorm(),
                cmap=model_cm, linewidth=0, edgecolor='none')

    # surrogate model
    ax.plot_surface(X, Y, sur, rstride=1, cstride=1, cmap=colormap, linewidth=0, edgecolor='none')
    
    # samples
    # ax.scatter(samples[:,0], samples[:,1], objective(samples), c='black', marker='x',  s=20)
    
    # search distribution mean
    # mean = search_dist.mean
    # ax.scatter(mean[0], mean[1], objective(mean), c='green', s=75)
    
    # optimum
    # ax.scatter(0.0,0.0,0.0, c='red', s=300)


    # ax.set_zlim([-100, 2500])
    
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])    
    # ax.set_axis_off()
    plt.show()


def manual_log(data):
    if data < 10: # Linear scaling up to 1
        return data/10
    else: # Log scale above 1
        return math.log10(data)

    
def animate_run(objective, means) -> None:
    """Animate the run of the MORE algorithm.
    """
    fig = plt.figure()
    fig.clf()
    ax = Axes3D(fig, azim=-128.0, elev=43.0)
    # ax = fig.add_subplot(projection="3d")    

    s = 0.1
    x = np.arange(-5.0, 5 + s, s)
    y = np.arange(-5.0, 5 + s, s)
    X, Y = np.meshgrid(x, y)
    
    obj = np.zeros((len(x), len(x)))
    for i in range(len(x)):
        for j in range(len(x)):
            obj[i,j] = objective(np.array([X[i,j], Y[i,j]]))

    # objective function
    ax.plot_surface(X, Y, obj, rstride=1, cstride=1, norm=LogNorm(),
                    cmap=model_cm, linewidth=0, edgecolor='none', alpha=0.7)

    mean_0 = [np.asscalar(x[0]) for x in means]
    mean_1 = [np.asscalar(x[1]) for x in means]
    out = objective(means)
    ax.scatter(mean_0[1:], mean_1[1:], out[1:], c='green', s=25)    

    # plot as line with starting mean
    ax.plot(mean_0, mean_1, objective(means).flatten().tolist())
    mean = means[0]
    ax.scatter(mean[0], mean[1], out[0], c='red', s=50)

    # line_ani = animation.ArtistAnimation(fig, data, interval=50)

    # ax.set_zlim([-500, 1000])        
    plt.show()
