# My Bachelor Thesis at KIT
I wrote my bachelor thesis at the
[ALR group](https://alr.anthropomatik.kit.edu/).
The main goal of the thesis was to improve
the sample efficiency of the MORE [1] algorithm.
The MORE algorithm is a stochastic search algorithm that
can be used for optimization problems. A key step of the 
MORE algorithm is approximating the often complex original object
function with a quadratic surrogate model.
In the field of 
robotics we can use it for model-free policy search, a subfield of 
reinfocement learning. 
The original code for the MORE algorithm is based on 
[https://github.com/maxhuettenrauch/MORE](https://github.com/maxhuettenrauch/MORE).

The main idea for the thesis was to
use recursive estimation techniques like the Kalman filter and
Recursive Least Squares for estimation of the surrogate model which I
implemented from scratch. The MORE version with recursive surrogate-modeling
is benchmarked on the rosenbrock function, a simple planar reaching task
and on a simulation of ball-in-a-cup task using MuJoCo with the
Barret WAM robot arm.


In the reaching task the end effector has to pass through
two via-points at a specified time.
![](name-of-giphy.gif)

A solution for the ball-in-the-cup-task:

[1] (Model-Based Relative Entropy Stochastic Search, Abdolmaleki et al. 2015)
