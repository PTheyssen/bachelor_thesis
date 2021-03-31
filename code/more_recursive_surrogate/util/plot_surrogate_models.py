import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from more.gauss_full_cov import GaussFullCov
from cma.bbobbenchmarks import nfreefunclasses

"""Plot example of surrogate models for MORE algorithm
   to illustrate correlation and idea for using recursive 
   estimation techniques.

   1. Plot old surrogate model
   2. Draw samples from search distribution
   3. Plot new estimated surrogate model

- https://matplotlib.org/stable/gallery/images_contours_and_fields/contour_demo.html#sphx-glr-gallery-images-contours-and-fields-contour-demo-py
"""

def calc_sur(x, model_par):
    """Calculates the scalar values for a surrogate model.
    """
    A = np.array([[model_par[0,0], model_par[1,0]],
                  [model_par[1,0], model_par[2,0]]])
    b = np.array([[model_par[3,0]],
                  [model_par[4,0]]])
    c = model_par[5,0]
    return x.T @ A @ x + x.T @ b + c

def plot_surrogate(samples, rewards, search_dist, model, range_, limit):
    delta = 0.025
    x = np.arange(-range_, range_, delta)
    y = np.arange(-range_, range_, delta)
    X, Y = np.meshgrid(x, y)
    Z = (1.0 - X)**2 + 100.0 * (Y - X*X)**2


    sur = np.zeros((len(x), len(x)))
    for i in range(len(x)):
        for j in range(len(x)):
            sur[i,j] = calc_sur(np.array([X[i,j], Y[i,j]]), model)
    
    fig, ax = plt.subplots()
    # levels = np.arange(-10000,10000,200)

    
    # l = np.concatenate([np.arange(0, 100, 15), np.arange(100, 2000, 150)])

    l = np.concatenate([np.arange(0, 5, 1.5), np.arange(5, 100, 20),
                        np.arange(100, 750, 100),
                        np.arange(750, 10000, 500)]
    )
    # CS = ax.contour(X, Y, Z, l)
    CS = ax.contour(X, Y, sur, 15)    
    ax.scatter(samples[:,0], samples[:,1], c='black', marker='x', label='samples')
    mean = search_dist.mean
    ax.scatter(mean[0], mean[1], s=100, label='search distribtion mean')

    # ax.clabel(CS, inline=True, fontsize=10)
    # ax.set_title('Simplest default with labels')    
    plt.xlim([-limit,limit])
    plt.ylim([-limit,limit])
    plt.legend()
    plt.show()
    # import tikzplotlib
    # tikzplotlib.save("figure_2.tex")


