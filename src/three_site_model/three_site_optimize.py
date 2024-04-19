# Optimize t and k parameters based on the starting values from the three_site_explore_binding_pars.py script
from three_site_model import *
from three_site_param_scan_hill import plot_parameters, plot_parameter_distributions, plot_predictions, get_N_I_P, calc_state_prob, \
    plot_state_probabilities, calculate_ifnb
import matplotlib.pyplot as plt
import pandas as pd
import os
import scipy.optimize as opt
import time
from multiprocessing import Pool
import argparse
import seaborn as sns

def minimize_objective(pars, N, I, beta, h_pars, bounds, c_par=None):
    return opt.minimize(three_site_objective, pars, args=(N, I, beta, c_par, h_pars), method='Nelder-Mead', bounds=bounds)

def optimize_model(N, I, P, beta, initial_pars, h, c=None, num_threads=40, num_t_pars=5, num_k_pars=3):
    '''
    Optimize the parameters of the model
    N: list or array of NFkB concentrations
    I: list or array of IRF concentrations
    P: list or array of p50 concentrations
    beta: list or array of IFNb concentrations
    initial_pars: array where each row is a set of initial parameters to be optimized
    h: Hill coefficients (list of 2 or 3 values)
    c: cooperativity parameter (single value)
    num_threads: number of threads to use for parallel processing
    '''

    start = time.time()
    # print("Starting optimization at %s for %d initial parameters" % (time.ctime(), len(initial_pars)), flush=True)
    
    min_k_order = -3
    max_k_order = 4

    # Define constraints
    bnds = [(0,1) for i in range(num_t_pars)]
    bnds += [(10**min_k_order, 10**max_k_order) for i in range(num_k_pars)]
    bnds = tuple(bnds)

    if len(bnds) != len(initial_pars[0]):
        raise ValueError("Number of bounds (%d) does not match number of parameters (%d)" % (len(bnds), len(initial_pars[0])))

    # Optimize
    with Pool(num_threads) as p:
        results = p.starmap(minimize_objective, [(pars, N, I, beta, h, bnds, c) for pars in initial_pars])
    
    final_pars = np.array([result.x for result in results]) # each row is a set of optimized parameters
    rmsd_scaled = np.array([result.fun for result in results])
    rmsd = rmsd_scaled / 1000

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
    model = "three_site_only_hill"

    # Model details
    num_t_pars = 5
    num_k_pars = 3
    num_h_pars = 3

    # par_names = ["t%d" % (i+1) for i in range(num_t_pars)] + ["k%d" % (i+1) for i in range(num_k_pars)] + ["h%d" % (i+1) for i in range(num_h_pars)]
    # par_names = [par.replace("k3", "kn") for par in par_names]
    # par_names = [par.replace("h3", "hn") for par in par_names]

    num_pars = num_t_pars + num_k_pars

    # Directories
    figs_dir = "optimization/figures/"
    res_dir = "optimization/results/"
    os.makedirs(figs_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)
    params_dir = "three_site_param_scan_hill/results"
    
    # Load training data
    training_data = pd.read_csv("../data/training_data.csv")
    print("Using the following training data:\n", training_data)
    N = training_data["NFkB"]
    I = training_data["IRF"]
    P = training_data["p50"]
    beta = training_data["IFNb"]
    conditions = training_data["Stimulus"] + "_" + training_data["Genotype"]
    # num_pts = len(N)
    # len_training = len(N)
    # stimuli = training_data["Stimulus"].unique()
    # genotypes = training_data["Genotype"].unique()
    
    for seed in range(3):
    # for seed in range(1):
        pars_dir = "%s/seed_%d" % (params_dir, seed)
        figures_dir = "%s/seed_%d" % (figs_dir, seed)
        results_dir = "%s/seed_%d" % (res_dir, seed)
        os.makedirs(figures_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)

        h_vals = np.loadtxt("%s/%s_h_values.csv" % (pars_dir, model), delimiter=",")
        for row in h_vals:
            h = row
            h_vals_str = "_".join([str(int(val)) for val in h])
            print("\n######\nBeginning optimization for h = %s" % h_vals_str, flush=True)

            # Load initial parameters
            # initial_pars = np.loadtxt("%s/%s_all_best_20_pars_h_%s.csv" % (pars_dir, model, h_vals_str), delimiter=",")
            initial_pars = pd.read_csv("%s/%s_best_20_pars_h_%s.csv" % (pars_dir, model, h_vals_str))
            rmsd_initial = initial_pars["rmsd"]
            initial_pars = initial_pars.drop(columns=["rmsd"]+["h%d" % (i+1) for i in range(num_h_pars)])
            par_names = initial_pars.columns
            initial_pars = initial_pars.to_numpy()

            # Optimize
            if args.optimize:
                t = time.time()
                print("Optimizing %d initial parameters..." % len(initial_pars), flush=True)
                final_pars, rmsd = optimize_model(N, I, P, beta, initial_pars, h, num_threads=num_threads)
                np.savetxt("%s/%s_optimized_parameters_h_%s.csv" % (results_dir, model, h_vals_str), final_pars, delimiter=",")
                np.savetxt("%s/%s_rmsd_h_%s.csv" % (results_dir, model, h_vals_str), rmsd, delimiter=",")
                print("Optimization took %.2f minutes" % ((time.time() - t)/60), flush=True)
            else:
                print("Loading optimized parameters...", flush=True)
                final_pars = np.loadtxt("%s/%s_optimized_parameters_h_%s.csv" % (results_dir, model, h_vals_str), delimiter=",")
                rmsd = np.loadtxt("%s/%s_rmsd_h_%s.csv" % (results_dir, model, h_vals_str), delimiter=",")

            # Plot results
            print("Plotting results...", flush=True)
            plot_parameters(final_pars, subset="All", name="optimized_parameters_h_%s" % h_vals_str, figures_dir=figures_dir, param_names=par_names)
    
            # Plot parameter distributions
            plot_parameter_distributions(final_pars, subset="All", name="optimized_parameter_distribution_h_%s" % h_vals_str, figures_dir=figures_dir, param_names=par_names)

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
                            plot_state_probabilities(state_probabilities, state_names, "optimized_state_probabilities_%s_%s_h_%s" % (stimulus, genotype, h_vals_str), figures_dir=figures_dir)
                            np.savetxt("%s/%s_state_probabilities_optimized_%s_%s_h_%s.csv" % (results_dir, model, stimulus, genotype, h_vals_str), state_probabilities, delimiter=",")
            else:
                print("Skipping state probabilities.", flush=True)

            # Plot predictions
            print("Plotting predictions...", flush=True)

            # Add h parameters to final_pars
            final_pars = np.concatenate((final_pars, np.stack([h for _ in range(len(final_pars))], axis=0)), axis=1)

            with Pool(num_threads) as p:
                results = p.starmap(calculate_ifnb, [(pars, training_data) for pars in final_pars])
            
            ifnb_predicted = np.array(results)
            plot_predictions(ifnb_predicted, beta, conditions, name="optimized_predictions_h_%s" % h_vals_str, figures_dir=figures_dir)
            np.savetxt("%s/%s_ifnb_predicted_optimized_h_%s.csv" % (results_dir, model, h_vals_str), ifnb_predicted, delimiter=",")

            # Calculate rmsd
            beta_stacked = np.stack([beta for _ in range(len(initial_pars))], axis=0)
            residuals = ifnb_predicted - beta_stacked
            rmsd_final = np.sqrt(np.mean(residuals**2, axis=1))

            rmsd_df = pd.DataFrame({"rmsd_initial": rmsd_initial, "rmsd_final": rmsd_final})
            rmsd_df.to_csv("%s/%s_rmsd_h_%s.csv" % (results_dir, model, h_vals_str), index=False)
            rmsd_df["par_set"] = rmsd_df.index
            rmsd_df = rmsd_df.melt(var_name="rmsd_type", value_name="rmsd", id_vars="par_set")

            # Plot rmsd
            fig = plt.figure()
            sns.relplot(data=rmsd_df, x="par_set", y="rmsd", hue="rmsd_type", kind="scatter")
            sns.despine()
            plt.savefig("%s/rmsd_comparison_h_%s.png" % (figures_dir, h_vals_str))
            plt.close()

            plt.figure()
            sns.kdeplot(data=rmsd_df, x="rmsd", hue="rmsd_type", common_norm=False)
            sns.despine()
            plt.savefig("%s/rmsd_density_plot_h_%s.png" % (figures_dir, h_vals_str))


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