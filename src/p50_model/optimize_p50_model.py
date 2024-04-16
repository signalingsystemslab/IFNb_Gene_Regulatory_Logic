# Optimize the parameters of the model: t parameters and k parameters
# Initial parameters from parameter scan

from p50_model import *
from parameter_scan_p50_model import calc_state_prob, plot_state_probabilities, plot_parameter_distributions, get_N_I_P
from p50_model_figures import plot_parameters, plot_predictions
import matplotlib.pyplot as plt
import pandas as pd
import os
import scipy.optimize as opt
import time
from multiprocessing import Pool
import argparse
import seaborn as sns
# import scipy.stats.qmc as qmc

def calculate_ifnb(pars, data, h_pars, c=None, num_t_pars=5, num_k_pars=4):
    t_pars, k_pars = pars[:num_t_pars], pars[num_t_pars:]
    N, I, P = data["NFkB"], data["IRF"], data["p50"]
    ifnb = [get_f(t_pars, k_pars, n, i, p, c_par=c, h_pars=h_pars) for n, i, p in zip(N, I, P)]
    ifnb = np.array(ifnb)
    return ifnb

def p50_objective(pars, *args):
    """Minimization objective function for optimizing a k parameter.
    Args:
    pars: array of parameters
    args: tuple of (N, I, IFNb_data)
    kwargs: additional parameters (c, h)
    """

    N, I, P, beta, h_pars, which_pars, other_pars = args

    if which_pars == "All":
        if len(pars) != len(other_pars):
            raise ValueError("Number of parameters (%d) does not match number of t and k parameters (%d)" % (len(pars), other_pars.shape[1]))
    elif len(which_pars) != len(pars):
        raise ValueError("Number of parameters (%d) does not match par names (%d)" % (len(pars), len(which_pars)))

    t_pars = other_pars[0:5]
    k_pars = other_pars[5:9]
    if which_pars == "All":
        t_pars = pars[0:5]
        k_pars = pars[5:9]
    else:
        for i in range(len(which_pars)):
            par = which_pars[i]
            if par == "kn":
                k_pars[2] = pars[i]
            elif par == "kp":
                k_pars[3] = pars[i]
            elif par == "k1":
                k_pars[0] = pars[i]
            elif par == "k2":
                k_pars[1] = pars[i]
            elif par == "h1":
                h_pars[0] = pars[i]
            elif par == "h2":
                h_pars[1] = pars[i]
            else:
                raise ValueError("Parameter %s not recognized" % par)
        
    num_pts = len(N)
    
    f_list = [get_f(t_pars, k_pars, N[i], I[i], P[i], h_pars=h_pars) for i in range(num_pts)] 
    residuals = np.array(f_list) - beta
    
    rmsd = np.sqrt(np.mean(residuals**2))
    return rmsd

def minimize_objective(pars, N, I, P, beta, h_pars, which_pars, other_pars, bounds):
    return opt.minimize(p50_objective, pars, args=(N, I, P, beta, h_pars, which_pars, other_pars), method="Nelder-Mead", bounds=bounds)

def optimize_model(N, I, P, beta, initial_pars, h, opt_pars, num_threads=40):
    '''
    Optimize the parameters of the model
    N: list or array of NFkB concentrations
    I: list or array of IRF concentrations
    P: list or array of p50 concentrations
    beta: list or array of IFNb concentrations
    initial_pars: array where each row is a set of initial parameters
    h: Hill coefficients (list of 2 or 3 values)
    opt_pars: list of parameter names to optimize (e.g. ["k1", "k2"])
    num_threads: number of threads to use for parallel processing
    '''

    start = time.time()
    print("Starting optimization at %s for parameters: %s" % (time.ctime(), opt_pars), flush=True)
    
    min_k_order = -3
    max_k_order = 4
    min_h = 1
    max_h = 5

    pars_locs = {"t1":0, "t2":1, "t3":2, "t4":3, "t5":4, "k1":5, "k2":6, "kn":7, "kp":8}
    h_locs = {"h1":0, "h2":1}

    # Define bounds
    bnds=[]
    if opt_pars == "All":
        bnds = [(0,1) for i in range(5)]
        bnds += [(10**min_k_order, 10**max_k_order) for i in range(4)]
        pars = initial_pars

    else:
        pars=np.zeros((len(initial_pars), len(opt_pars)))
        for i in range(len(opt_pars)):
            par = opt_pars[i]
            if par[0] == "k":
                bnds.append((10**min_k_order, 10**max_k_order))
                pars[:,i] = initial_pars[:,pars_locs[par]]
            elif par[0] == "h":
                bnds.append((min_h, max_h))
                pars[:,i] = h[:,h_locs[par]]
            else:
                raise ValueError("Parameter %s not recognized" % par)
        # print(pars)
    # Optimize
    with Pool(num_threads) as p:
        results = p.starmap(minimize_objective, [(pars[i], N, I, P, beta, h, opt_pars, initial_pars[i], bnds) for i in range(len(pars))])
    
    final_pars = np.array([result.x for result in results]) # each row is a set of optimized parameters
    rmsd = np.array([result.fun for result in results])

    if opt_pars != "All":
        final_pars_temp = initial_pars.copy()
        # replace the optimized parameters with the initial parameters for the parameters that were not optimized
        for i in range(len(opt_pars)):
            par = opt_pars[i]
            if par[0] == "k":
                final_pars_temp[:,pars_locs[par]] = final_pars[:,i]
            elif par[0] == "h":
                final_pars_temp[:,h_locs[par]] = final_pars[:,i]
            else:
                raise ValueError("Parameter %s not recognized" % par)
        final_pars = final_pars_temp

    # add each h parameter to the final parameters
    h_repeat = np.repeat(h, len(final_pars)).reshape(-1, len(final_pars)).T
    final_pars = np.concatenate((final_pars, h_repeat), axis=1)

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
    parser.add_argument("-p","--optimize_kp", action="store_true")
    parser.add_argument("-s","--state_probabilities", action="store_true")
    args = parser.parse_args()

    start_start = time.time()

    print("###############################################\n")
    print("Starting at %s\n" % time.ctime(), flush=True)

    # Settings    
    num_threads = 40
    model = "p50_diff_binding"
    par_type = "k"
    h_val = "3_3_1"
    h1, h2, h3 = [int(h) for h in h_val.split("_")]

    # Model details
    num_t_pars = 5
    num_k_pars = 4
    num_c_pars = 0
   
    num_pars = num_t_pars + num_k_pars

    # Directories
    figures_dir = "optimization/figures/"
    results_dir = "optimization/results/"
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    pars_dir = "../three_site_model/optimization/results/seed_0/"
    
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
    initial_pars = np.loadtxt("%s/three_site_only_hill_optimized_parameters_h_%s.csv" % (pars_dir, h_val), delimiter=",")
    initial_pars = np.concatenate((initial_pars, np.ones((len(initial_pars),1))), axis=1) # Add kp parameter
    initial_pars_names = ["t%d" % (i+1) for i in range(num_t_pars)] + ["k%d" % (i+1) for i in range(num_k_pars)]
    result_par_names = initial_pars_names + ["h1", "h2"]

    # Calculate rmsd for initial parameters
    beta_stacked = np.stack([beta for _ in range(len(initial_pars))], axis=0)
    with Pool(num_threads) as p:
        results = p.starmap(calculate_ifnb, [(pars, training_data, [h1, h2]) for pars in initial_pars])

    ifnb_predicted = np.array(results)
    residuals = ifnb_predicted - beta_stacked
    rmsd_initial = np.sqrt(np.mean(residuals**2, axis=1))

    # Optimize
    if args.optimize:
        # Optimize all k and t parameters
        print("Optimizing all k and t parameters...", flush=True)
        par_names = "All"
        final_pars, rmsd = optimize_model(N, I, P, beta, initial_pars, [h1, h2], par_names, num_threads=num_threads)
        np.savetxt("%s/%s_optimized_parameters.csv" % (results_dir, model), final_pars, delimiter=",")
        np.savetxt("%s/%s_rmsd.csv" % (results_dir, model), rmsd, delimiter=",")
    elif not args.optimize_kp:
        print("Loading optimized parameters...", flush=True)
        final_pars = np.loadtxt("%s/%s_optimized_parameters.csv" % (results_dir, model), delimiter=",")
        rmsd = np.loadtxt("%s/%s_rmsd.csv" % (results_dir, model), delimiter=",")

    if not args.optimize_kp:
                # Plot results
        print("Plotting results...", flush=True)
        final_pars_df = pd.DataFrame(final_pars, columns=result_par_names)
        plot_parameters(final_pars_df, name="optimized_parameters", figures_dir=figures_dir)

        # # Plot parameter distributions
        # plot_parameter_distributions(final_pars, subset="All", name="optimized_parameter_dist", figures_dir=figures_dir, param_names=par_names)

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

        final_pars_tk = final_pars[:, :num_t_pars+num_k_pars]
        with Pool(num_threads) as p:
            results = p.starmap(calculate_ifnb, [(pars, training_data, [h1, h2]) for pars in final_pars_tk])
        
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

    if args.optimize_kp:
        # Optimize kp
        print("Optimizing parameter kp...", flush=True)
        par_names = ["kp"]
        final_pars, rmsd = optimize_model(N, I, P, beta, initial_pars, [h1, h2], par_names, num_threads=num_threads)
        np.savetxt("%s/%s_optimized_parameters_kp.csv" % (results_dir, model), final_pars, delimiter=",")
        np.savetxt("%s/%s_rmsd_kp.csv" % (results_dir, model), rmsd, delimiter=",")

        # Plot results
        print("Plotting results...", flush=True)
        final_pars_df = pd.DataFrame(final_pars, columns=result_par_names)
        plot_parameters(final_pars_df, name="optimized_parameters_kp", figures_dir=figures_dir)

        # # Plot parameter distributions
        # plot_parameter_distributions(final_pars, subset="kp", name="optimized_parameter_dist_kp", figures_dir=figures_dir, param_names=par_names)

        # Plot predictions
        print("Plotting predictions...", flush=True)

        final_pars_tk = final_pars[:, :num_t_pars+num_k_pars]
        with Pool(num_threads) as p:
            results = p.starmap(calculate_ifnb, [(pars, training_data, [h1, h2]) for pars in final_pars_tk])

        ifnb_predicted = np.array(results)
        plot_predictions(ifnb_predicted, beta, conditions, name="optimized_predictions_kp", figures_dir=figures_dir)
        np.savetxt("%s/%s_ifnb_predicted_optimized_kp.csv" % (results_dir, model), ifnb_predicted, delimiter=",")



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