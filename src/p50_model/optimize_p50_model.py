# Optimize the parameters of the model: t parameters and k parameters
# Initial parameters from parameter scan

from p50_model import *
from parameter_scan_p50_model import calc_state_prob, plot_state_probabilities, plot_predictions, plot_parameters, plot_parameter_distributions
import matplotlib.pyplot as plt
import pandas as pd
import os
import scipy.optimize as opt
import time
from multiprocessing import Pool
import argparse
import seaborn as sns
# import scipy.stats.qmc as qmc

def minimize_objective(pars, N, I, P, beta, model_type, bounds, constraints):
    return opt.minimize(p50_objective, pars, args=(N, I, P, beta, model_type), method='SLSQP', bounds=bounds, constraints=constraints)

def optimize(N, I, P, beta, model_type, initial_pars, num_threads=40):
    '''
    Optimize the parameters of the model
    N: list or array of NFkB concentrations
    I: list or array of IRF concentrations
    P: list or array of p50 concentrations
    beta: list or array of IFNb concentrations
    model_type: string indicating which parameters will be optimized (k, c, kc, kc, or t)
    initial_pars: array where each row is a set of initial parameters
    num_threads: number of threads to use for parallel processing
    '''

    start = time.time()
    print("Starting optimization at %s for %d initial parameters" % (time.ctime(), len(initial_pars)), flush=True)
    
    # Define constraints
    cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - x[4]},
        {'type': 'ineq', 'fun': lambda x: (x[0] - x[4])-(x[6] - x[7])},
        {'type': 'ineq', 'fun': lambda x:  (x[3] - x[2])},
        {'type': 'ineq', 'fun': lambda x: (x[3] - x[2])-(x[1] - x[0])})
    bnds = ((0.05, None), (0.025, None), (0.05, None), (0.025, None))

    # Optimize
    with Pool(num_threads) as p:
        results = p.starmap(minimize_objective, [(pars, N, I, P, beta, model_type, bnds, cons) for pars in initial_pars])
    
    final_pars = np.array([result.x for result in results]) # each row is a set of optimized parameters
    rmsd = np.array([result.fun for result in results])

    end = time.time()

    print("Finished optimization at %s" % time.ctime(), flush=True)
    t = end - start
    if t < 60:
        print("Time elapsed: %.2f seconds" % t)
    elif t < 3600:
        print("Time elapsed: %.2f minutes" % (t/60))
    else:
        print("Time elapsed: %.2f hours" % (t/3600))
    return final_pars, rmsd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o","--optimize", action="store_true")
    args = parser.parse_args()

    start_start = time.time()

    print("###############################################\n")
    print("Starting at %s\n" % time.ctime(), flush=True)

    training_data = pd.read_csv("../data/p50_training_data.csv")
    print("Using the following training data:\n", training_data)
    model = "p50_diff_binding"
    figures_dir = "optimization_k/figures/"
    results_dir = "optimization_k/results/"

    N = training_data["NFkB"]
    I = training_data["IRF"]
    P = training_data["p50"]
    beta = training_data["IFNb"]
    conditions = training_data["Stimulus"] + "_" + training_data["Genotype"]
    num_pts = len(N)
    len_training = len(N)
    num_t_pars = 6
    num_k_pars = 4

    par_names = ["t%d" % (i+1) for i in range(num_t_pars)] + ["k%d" % (i+1) for i in range(num_k_pars)]
    par_names.str.replace("k3", "kn")
    par_names.str.replace("k4", "kp")
    print("Optimizing the following parameters:\n", par_names)

    num_pars = num_t_pars + num_k_pars
    num_threads = 40
    num_par_sets = 10**6

    num_best_pars = []