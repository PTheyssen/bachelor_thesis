# MORE with recursive surrogate-modeling

Implementation of the Model Based Relative Entropy Search [1] algorithm
with recursive surrogate-modeling using the
Recrusive least squares algorithm with a drift model.

## Project Structure
The more folder contains the main algorithm implementations.
The examples folder contains quick prototyping code, where
I tried new ideas. The cw2_experiments folder
contains the implementation of MORE as a cw2 experiment
and various experiment yml files.

## Installation

In your virtual environment do

`pip install -e .`

Then install cw2 do

`pip install -e your_path_to_cw2`

The experiments for ball-in-a-cup require MuJoCo.

## Usage


[1] Model-Based Relative Entropy Stochastic Search, Abdolmaleki et al. 2015

## Using [cw2](https://github.com/ALRhub/cw2) for running experiments

The folder more_cw2 contains an Implementation of a
cw2 experiment for the MORE algorithm.

Currently the configuration file is set for hyperparameter optimization
for MORE with Recursive Least Squares for estimating the
parameters of the surrogate model.

To run the experiment, inside the more_cw2 folder:

`python more_main.py more_config.yml`

