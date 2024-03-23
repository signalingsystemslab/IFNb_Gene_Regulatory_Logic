# Optimize the parameters of the model: t parameters and k parameters
# Initial parameters from parameter scan

from p50_model import *
from parameter_scan_p50_model import calc_state_prob, plot_state_probabilities, plot_predictions, plot_parameters, plot_parameter_distributions, get_N_I_P, calculate_ifnb
import matplotlib.pyplot as plt
import pandas as pd
import os
import scipy.optimize as opt
import time
from multiprocessing import Pool
import argparse
import seaborn as sns
# import scipy.stats.qmc as qmc

def minimize_objective(pars, N, I, P, beta, par_type, bounds, constraints):
    return opt.minimize(p50_objective, pars, args=(N, I, P, beta, par_type), method='COBYLA', bounds=bounds, constraints=constraints)

def constraint1(x):
    return x[0] - x[4] - 0.15

def constraint2(x):
    return (x[0] - x[4]) - (x[6] - x[7]) - 0.1

def constraint3(x):
    return x[3] - x[2] - 0.05

def constraint4(x):
    return (x[3] - x[2]) - (x[1] - x[0] ) - 0.1

# def bound1(x):
#     return min(x)

# def bound2(x):
#     return 1-max(x)

def optimize_model(N, I, P, beta, par_type, initial_pars, num_threads=40):
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
    
    min_k_order = -3
    max_k_order = 4
    min_c_order = -2
    max_c_order = 2

    # Define constraints
    bnds = [(0,1) for i in range(6)]
    if par_type == "k":
            bnds += [(10**min_k_order, 10**max_k_order) for i in range(4)]
    elif par_type == "c":
        bnds += [(10**min_c_order, 10**max_c_order)]
    elif par_type == "kc":
        bnds += [(10**min_k_order, 10**max_k_order) for i in range(4)] + [(10**min_c_order, 10**max_c_order)]

    bnds = tuple(bnds)

    cons = ({'type': 'ineq', 'fun': constraint1},
            {'type': 'ineq', 'fun': constraint2},
            {'type': 'ineq', 'fun': constraint3},
            {'type': 'ineq', 'fun': constraint4})

    # Optimize
    with Pool(num_threads) as p:
        results = p.starmap(minimize_objective, [(pars, N, I, P, beta, par_type, bnds, cons) for pars in initial_pars])
    
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
    parser.add_argument("-s","--state_probabilities", action="store_true")
    args = parser.parse_args()

    start_start = time.time()

    print("###############################################\n")
    print("Starting at %s\n" % time.ctime(), flush=True)

    # Settings    
    num_threads = 40
    model = "p50_diff_binding"
    par_type = "k"

    # Model details
    num_t_pars = 6
    if model == "p50_diff_binding":
        num_k_pars = 4
        num_c_pars = 0
    elif model == "p50_binding_coop":
        num_k_pars = 4
        num_c_pars = 1
    par_names = ["t%d" % (i+1) for i in range(num_t_pars)] + ["k%d" % (i+1) for i in range(num_k_pars)] + ["c" for i in range(num_c_pars)]
    par_names = [par.replace("k3", "kn") for par in par_names]
    par_names = [par.replace("k4", "kp") for par in par_names]

    print("Optimizing the following parameters:\n", par_names)
    num_pars = num_t_pars + num_k_pars

    # Directories
    figures_dir = "optimization_k/figures/"
    results_dir = "optimization_k/results/"
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    if model == "p50_diff_binding":
        pars_dir = "param_scan_k/results/"
    elif model == "p50_binding_coop":
        pars_dir = "param_scan_kc/results/"
    else:
        raise ValueError("Invalid model: %s" % model)
    
    # Load training data
    training_data = pd.read_csv("../data/p50_training_data.csv")
    print("Using the following training data:\n", training_data)
    N = training_data["NFkB"]
    I = training_data["IRF"]
    P = training_data["p50"]
    beta = training_data["IFNb"]
    conditions = training_data["Stimulus"] + "_" + training_data["Genotype"]
    num_pts = len(N)
    len_training = len(N)
    # stimuli = training_data["Stimulus"].unique()
    # genotypes = training_data["Genotype"].unique()
    
    # Load initial parameters
    initial_pars = np.loadtxt("%s/%s_all_initial_parameters.csv" % (pars_dir, model), delimiter=",")

    # Calculate rmsd for initial parameters
    beta_stacked = np.stack([beta for _ in range(len(initial_pars))], axis=0)
    with Pool(num_threads) as p:
        results = p.starmap(calculate_ifnb, [(pars, training_data) for pars in initial_pars])

    ifnb_predicted = np.array(results)
    residuals = ifnb_predicted - beta_stacked
    rmsd_initial = np.sqrt(np.mean(residuals**2, axis=1))

    # Optimize
    if args.optimize:
        print("Optimizing %d initial parameters..." % len(initial_pars), flush=True)
        final_pars, rmsd = optimize_model(N, I, P, beta, par_type, initial_pars, num_threads=num_threads)
        np.savetxt("%s/%s_optimized_parameters.csv" % (results_dir, model), final_pars, delimiter=",")
        np.savetxt("%s/%s_rmsd.csv" % (results_dir, model), rmsd, delimiter=",")
    else:
        print("Loading optimized parameters...", flush=True)
        final_pars = np.loadtxt("%s/%s_optimized_parameters.csv" % (results_dir, model), delimiter=",")
        rmsd = np.loadtxt("%s/%s_rmsd.csv" % (results_dir, model), delimiter=",")

    # Plot results
    print("Plotting results...", flush=True)
    plot_parameters(final_pars, subset="All", name="optimized_parameters", figures_dir=figures_dir, param_names=par_names)

    # Plot parameter distributions
    plot_parameter_distributions(final_pars, subset="All", name="optimized_parameter_dist", figures_dir=figures_dir, param_names=par_names)

    # Plot state probabilities
    if args.state_probabilities:
        print("Calculating and plotting state probabilities...", flush=True)
        all_k_pars = final_pars[:, num_t_pars:]

        extra_training_data = pd.DataFrame({"Stimulus":"basal", "Genotype":"WT", "IRF":0.01, "NFkB":0.01, "p50":1}, index=[0])
        training_data_extended = pd.concat([training_data, extra_training_data], ignore_index=True)
        stimuli = training_data_extended["Stimulus"]
        genotypes = training_data_extended["Genotype"]

        probabilities = dict()
        for stimulus, genotype in zip(stimuli, genotypes):
                    nfkb, irf, p50 = get_N_I_P(training_data_extended, stimulus, genotype)
                    print("Calculating state probabilities for %s %s" % (stimulus, genotype), flush=True)
                    print("N=%.2f, I=%.2f, P=%.2f" % (nfkb, irf, p50), flush=True)
                
                    with Pool(num_threads) as p:
                        results = p.starmap(calc_state_prob, [(tuple(all_k_pars[i]), nfkb, irf, p50) for i in range(len(all_k_pars))])
                
                    state_names = results[0][1]
                    state_probabilities = np.array([x[0] for x in results])
                    probabilities[(stimulus, genotype)] = state_probabilities
                    plot_state_probabilities(state_probabilities, state_names, "optimized_state_probabilities_%s_%s" % (stimulus, genotype), figures_dir)
                    np.savetxt("%s/%s_state_probabilities_optimized_%s_%s.csv" % (results_dir, model, stimulus, genotype), state_probabilities, delimiter=",")
    else:
        print("Skipping state probabilities.", flush=True)

    # Plot predictions
    print("Plotting predictions...", flush=True)

    with Pool(num_threads) as p:
        results = p.starmap(calculate_ifnb, [(pars, training_data) for pars in final_pars])
    
    ifnb_predicted = np.array(results)
    plot_predictions(ifnb_predicted, beta, conditions, name="optimized_predictions", figures_dir=figures_dir)
    np.savetxt("%s/%s_ifnb_predicted_optimized.csv" % (results_dir, model), ifnb_predicted, delimiter=",")

    # Calculate rmsd
    residuals = ifnb_predicted - beta_stacked
    rmsd_final = np.sqrt(np.mean(residuals**2, axis=1))

    rmsd_df = pd.DataFrame({"rmsd_initial": rmsd_initial, "rmsd_final": rmsd_final})
    rmsd_df.to_csv("%s/%s_rmsd.csv" % (results_dir, model), index=False)
    rmsd_df["par_set"] = rmsd_df.index
    rmsd_df = rmsd_df.melt(var_name="rmsd_type", value_name="rmsd", id_vars="par_set")

    # Plot rmsd
    fig = plt.figure()
    sns.relplot(data=rmsd_df, x="par_set", y="rmsd", hue="rmsd_type", kind="scatter")
    sns.despine()
    plt.savefig("%s/rmsd_comparison.png" % figures_dir)
    plt.close()

    plt.figure()
    sns.kdeplot(data=rmsd_df, x="rmsd", hue="rmsd_type", common_norm=False)
    sns.despine()
    plt.savefig("%s/rmsd_density_plot.png" % figures_dir)


    end_end = time.time()
    t = end_end - start_start
    if t < 60:
        print("Total time elapsed: %.2f seconds" % t)
    elif t < 3600:
        print("Total time elapsed: %.2f minutes" % (t/60))
    else:
        print("Total time elapsed: %.2f hours" % (t/3600))

if __name__ == "__main__":
    main()