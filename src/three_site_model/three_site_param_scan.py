# Optimize the parameters of the model: t parameters and k parameters
# Initial parameters from parameter scan

from three_site_model import *
from state_probability_functions import calc_state_prob, plot_state_probabilities, get_N_I_P
from three_site_model_figures import plot_parameters, plot_predictions, plot_parameter_distributions
import matplotlib.pyplot as plt
import pandas as pd
import os
import scipy.optimize as opt
import time
from multiprocessing import Pool
import argparse
import seaborn as sns
import scipy.stats.qmc as qmc

def calculate_rmsd(ifnb_predicted, beta):
    residuals = ifnb_predicted - beta
    rmsd = np.sqrt(np.mean(residuals**2))
    return rmsd

def calculate_grid(training_data, h1=3, h2=1, t_bounds=(0,1), k_bounds=(10**-3,10**3), seed=0, num_samples=10**6, num_threads=60, num_t_pars=5, num_k_pars=4, num_h_pars=2, c_par=False):
    min_k_order = np.log10(k_bounds[0])
    max_k_order = np.log10(k_bounds[1])
    min_c_order = np.log10(10**-3)
    max_c_order = np.log10(10**3)
    min_t = t_bounds[0]
    max_t = t_bounds[1]

    seed += 10

    l_bounds = np.concatenate([np.zeros(num_t_pars)+min_t, np.ones(num_k_pars)*min_k_order])
    u_bounds = np.concatenate([np.zeros(num_t_pars)+max_t, np.ones(num_k_pars)*max_k_order])

    print("Calculating grid with %d samples using Latin Hypercube sampling" % num_samples, flush=True)
    if not c_par:
        sampler=qmc.LatinHypercube(d=num_t_pars+num_k_pars, seed=seed)
        grid_tk = sampler.random(n=num_samples)
        grid_tk = qmc.scale(grid_tk, l_bounds, u_bounds) # rows are parameter sets
        # convert k parameters to log space
        kgrid = grid_tk[:,num_t_pars:]
        kgrid = 10**kgrid
        grid_tk[:,num_t_pars:] = kgrid
    else:
        l_bounds = np.append(l_bounds, min_c_order)
        u_bounds = np.append(u_bounds, max_c_order)
        sampler=qmc.LatinHypercube(d=num_t_pars+num_k_pars+1, seed=seed)
        grid_tk = sampler.random(n=num_samples)
        grid_tk = qmc.scale(grid_tk, l_bounds, u_bounds)
        # convert k parameters to log space
        kgrid = grid_tk[:,num_t_pars:-1]
        kgrid = 10**kgrid
        grid_tk[:,num_t_pars:-1] = kgrid
        # Convert c parameter to log space
        cgrid = grid_tk[:,-1]
        cgrid = 10**cgrid
        grid_tk[:,-1] = cgrid

    grid=grid_tk

    # Load data points
    N, I, P = training_data["NFkB"], training_data["IRF"], training_data["p50"]
    beta = training_data["IFNb"]

    # Calculate IFNb value at each point in grid
    print("Calculating effects at %d points in grid" % len(grid), flush=True)
    start = time.time()
    if c_par:
        with Pool(num_threads) as p:
            results = p.starmap(get_f, [(grid[i,0:num_t_pars], grid[i,num_t_pars:-1], N[j], I[j], grid[i,-1], [h1, h2], False) for i in range(len(grid)) for j in range(len(N))])
    else:
        with Pool(num_threads) as p:
            results = p.starmap(get_f, [(grid[i,0:num_t_pars], grid[i,num_t_pars:], N[j], I[j], None, [h1, h2], False) for i in range(len(grid)) for j in range(len(N))])

    num_param_sets = len(grid)
    num_data_points = len(N)
    ifnb_predicted = np.array(results).reshape(num_param_sets, num_data_points)

    # Calculate residuals
    with Pool(num_threads) as p:
        rmsd = p.starmap(calculate_rmsd, [(ifnb_predicted[i], beta) for i in range(num_param_sets)])

    rmsd = np.array(rmsd)
    best_fits = np.argsort(rmsd)[:100]
    # print("Best fits: ", best_fits, flush=True)
    best_fits_rmsd = rmsd[best_fits]
    best_fits_pars = grid[best_fits]
    best_fits_results = ifnb_predicted[best_fits]

    end = time.time()
    t = end - start
    if t < 60*60:
        print("Time elapsed: %.2f minutes" % (t/60), flush=True)
    else:
        print("Time elapsed: %.2f hours" % (t/3600), flush=True)

    return best_fits_pars, best_fits_results, best_fits_rmsd

def objective_function(pars, *args):
    N, I, beta, h_pars, c_par = args
    t_pars = pars[0:5]
    k_pars = pars[5:8]
    if c_par:
        c = pars[8]
    else:
        c = None

    num_pts = len(N)
    
    ifnb_predicted = [get_f(t_pars, k_pars, N[i], I[i], h_pars=h_pars, c_par=c) for i in range(num_pts)]   
    rmsd = calculate_rmsd(ifnb_predicted, beta)

    return rmsd

def minimize_objective(pars, N, I, beta, h_pars, c_par, bounds):
    return opt.minimize(objective_function, pars, args=(N, I, beta, h_pars, c_par), method="Nelder-Mead", bounds=bounds)

def optimize_model(N, I, beta, initial_pars, h, c=False, num_threads=40, num_t_pars=5, num_k_pars=3):
    start = time.time()    
    min_k_order = -3
    max_k_order = 4
    min_c_order = -3
    max_c_order = 4

    # Define bounds
    bnds = [(0, 1) for i in range(num_t_pars)] + [(10**min_k_order, 10**max_k_order) for i in range(num_k_pars)]
    if c:
        bnds.append((10**min_c_order, 10**max_c_order))
    bnds = tuple(bnds)

    # Optimize
    with Pool(num_threads) as p:
        results = p.starmap(minimize_objective, [(pars, N, I, beta, h, c, bnds) for pars in initial_pars])

    final_pars = np.array([result.x for result in results]) # each row is a set of optimized parameters
    rmsd = np.array([result.fun for result in results])

    if c:
        with Pool(num_threads) as p:
            ifnb_predicted = p.starmap(get_f, [(final_pars[i,0:num_t_pars], final_pars[i,num_t_pars:-1], N[j], I[j], final_pars[i,-1], h, False) for i in range(len(final_pars)) for j in range(len(N))])
    else:
        with Pool(num_threads) as p:
            ifnb_predicted = p.starmap(get_f, [(final_pars[i,0:num_t_pars], final_pars[i,num_t_pars:], N[j], I[j], None, h, False) for i in range(len(final_pars)) for j in range(len(N))])

    ifnb_predicted = np.array(ifnb_predicted).reshape(len(final_pars), len(N))

    end = time.time()

    print("Finished optimization at %s" % time.ctime(), flush=True)
    t = end - start
    if t < 60:
        print("Time elapsed: %.2f seconds" % t)
    elif t < 3600:
        print("Time elapsed: %.2f minutes" % (t/60))
    else:
        print("Time elapsed: %.2f hours" % (t/3600))
    return final_pars, ifnb_predicted, rmsd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o","--optimize", action="store_true")
    # parser.add_argument("-k","--optimize_kp", action="store_true")
    parser.add_argument("-p","--param_scan", action="store_true")
    parser.add_argument("-s","--state_probabilities", action="store_true")
    parser.add_argument("-c","--c_parameter", action="store_true")
    parser.add_argument("-1","--h1", type=int, default=3)
    parser.add_argument("-2","--h2", type=int, default=1)
    args = parser.parse_args()

    start_start = time.time()

    print("###############################################\n")
    print("Starting at %s\n" % time.ctime(), flush=True)

    # Settings    
    num_threads = 40
    model = "three_site"
    par_type = "k"
    h1, h2 = args.h1, args.h2
    h3 = 1
    h_val = "%d_%d_%d" % (h1, h2, h3)
    print("h_I1: %d, h_I2: %d, h3_N: %d" % (h1, h2, h3), flush=True)
    c_par= args.c_parameter

    # Model details
    num_t_pars = 5
    num_k_pars = 3
    # num_c_pars = 1
    # num_h_pars = 2

    # Directories
    figures_dir = "parameter_scan/figures/"
    results_dir = "parameter_scan/results/"
    if h_val != "3_1_1":
        figures_dir = figures_dir[:-1] + "_h_%s/" % h_val
        results_dir = results_dir[:-1] + "_h_%s/" % h_val
    if c_par:
        figures_dir = figures_dir[:-1] + "_c_scan/"
        results_dir = results_dir[:-1] + "_c_scan/"
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    pars_dir = "../three_site_model/optimization/results/seed_0/"
    print("Saving results to %s" % results_dir, flush=True)
    
    # Load training data
    training_data = pd.read_csv("../data/training_data.csv")
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
    
    if c_par:
        # result_par_names = [r"t_I",r"$t_Ig$",r"$t_N$",r"$t_{I\cdotIg}$", r"$t_{I\cdotN}$", "k1", "k2", "kn", "c"]
        result_par_names = ["t1","t3","t4","t5", "k1", "k2", "kn", "c"]
    else:
        # result_par_names = [r"$t_I$",r"$t_Ig$",r"$t_N$",r"$t_{I\cdotIg}$", r"$t_{I\cdotN}$", "k1", "k2", "kn"]
        result_par_names = ["t1","t3","t4","t5", "k1", "k2", "kn"]

    # Optimize
    if args.param_scan:
        # Perform parameter scan
        print("Performing parameter scan...", flush=True)
        final_pars, ifnb_predicted, rmsd = calculate_grid(training_data, seed=0, num_threads=num_threads, h1=h1, h2=h2, c_par=c_par, num_t_pars=num_t_pars, num_k_pars=num_k_pars)
        print("Finished parameter scan.", flush=True)

        
        print("Plotting results...", flush=True)
        final_pars_df = pd.DataFrame(final_pars, columns=result_par_names)
        plot_parameters(final_pars_df, name="scanned_parameters", figures_dir=figures_dir)
        final_pars_df.to_csv("%s/%s_scanned_parameters.csv" % (results_dir, model), index=False)


        # Plot predictions
        print("Plotting predictions...", flush=True)

        plot_predictions(ifnb_predicted, beta, conditions, name="scanned_predictions", figures_dir=figures_dir)
        np.savetxt("%s/%s_ifnb_predicted_scanned.csv" % (results_dir, model), ifnb_predicted, delimiter=",")

        # Calculate rmsd
        rmsd_df = pd.DataFrame({"rmsd": rmsd, "par_set": np.arange(len(rmsd))})
        rmsd_df.to_csv("%s/%s_rmsd.csv" % (results_dir, model), index=False)

        # Plot rmsd
        fig = plt.figure()
        sns.relplot(data=rmsd_df, x="par_set", y="rmsd", kind="scatter")
        sns.despine()
        plt.savefig("%s/rmsd_comparison.png" % figures_dir)
        plt.close()

    if args.optimize:
        # Optimize the model
        print("Optimizing model...", flush=True)
        initial_pars = pd.read_csv("%s/%s_scanned_parameters.csv" % (results_dir, model)).values

        final_pars, ifnb_predicted, rmsd = optimize_model(N, I, beta, initial_pars, [h1, h2], c=c_par, num_threads=num_threads, num_t_pars=num_t_pars, num_k_pars=num_k_pars)
        print("Finished optimization.", flush=True)

        print("Plotting results...", flush=True)
        final_pars_df = pd.DataFrame(final_pars, columns=result_par_names)
        plot_parameters(final_pars_df, name="optimized_parameters", figures_dir=figures_dir)
        final_pars_df.to_csv("%s/%s_optimized_parameters.csv" % (results_dir, model), index=False)

        # Plot state probabilities
        if args.state_probabilities:
            print("Calculating and plotting state probabilities...", flush=True)
            all_k_pars = final_pars[:, num_t_pars:num_t_pars+num_k_pars]

            extra_training_data = pd.DataFrame({"Stimulus":"basal", "Genotype":"WT", "IRF":0.01, "NFkB":0.01, "p50":1}, index=[0])
            training_data_extended = pd.concat([training_data, extra_training_data], ignore_index=True)
            stimuli = training_data_extended["Stimulus"]
            genotypes = training_data_extended["Genotype"]

            probabilities = dict()
            for stimulus, genotype in zip(stimuli, genotypes):
                        nfkb, irf, p50 = get_N_I_P(training_data_extended, stimulus, genotype)
                        print("Calculating state probabilities for %s %s" % (stimulus, genotype), flush=True)
                        print("N=%.2f, I=%.2f, P=%.2f" % (nfkb, irf, p50), flush=True)


        print("Plotting predictions...", flush=True)
        plot_predictions(ifnb_predicted, beta, conditions, name="optimized_predictions", figures_dir=figures_dir)
        np.savetxt("%s/%s_ifnb_predicted_optimized.csv" % (results_dir, model), ifnb_predicted, delimiter=",")

        # Calculate rmsd
        initial_rmsd = pd.read_csv("%s/%s_rmsd.csv" % (results_dir, model))
        rmsd_df = pd.DataFrame({"rmsd_final": rmsd, "par_set": np.arange(len(rmsd)), "rmsd_initial": initial_rmsd["rmsd"]})
        rmsd_df = rmsd_df.melt(id_vars="par_set", value_vars=["rmsd_final", "rmsd_initial"], var_name="rmsd_type", value_name="RMSD")
        rmsd_df.to_csv("%s/%s_rmsd_optimized.csv" % (results_dir, model), index=False)

        # Plot rmsd
        fig = plt.figure()
        sns.relplot(data=rmsd_df, x="par_set", y="RMSD", kind="scatter", hue="rmsd_type", palette="plasma")
        sns.despine()
        plt.savefig("%s/rmsd_comparison_optimized.png" % figures_dir)
        plt.close()


        # Best 20 optimized parameters
        best_fits = np.argsort(rmsd)[:20]
        best_fits_rmsd = rmsd[best_fits]
        best_fits_pars = final_pars[best_fits]
        best_fits_results = ifnb_predicted[best_fits]
        best_fits_df = pd.DataFrame(best_fits_pars, columns=result_par_names)
        best_fits_df["rmsd"] = best_fits_rmsd
        best_fits_df.to_csv("%s/%s_best_fits_pars.csv" % (results_dir, model), index=False)
        np.savetxt("%s/%s_best_fits_ifnb_predicted.csv" % (results_dir, model), best_fits_results, delimiter=",")

        # Plot best fits
        plot_predictions(best_fits_results, beta, conditions, name="best_fits_predictions", figures_dir=figures_dir)
        plot_parameters(best_fits_df, name="best_fits_parameters", figures_dir=figures_dir)

    # Plot state probabilities
    if args.state_probabilities:
        final_pars = pd.read_csv("%s/%s_best_fits_pars.csv" % (results_dir, model)).values
        print("Calculating and plotting state probabilities...", flush=True)
        all_k_pars = final_pars[:, num_t_pars:num_t_pars+num_k_pars]
        h_pars = [h1, h2]

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
                        results = p.starmap(calc_state_prob, [(tuple(all_k_pars[i]), nfkb, irf, p50, num_t_pars, h_pars) for i in range(len(all_k_pars))])
                
                    state_names = results[0][1]
                    state_probabilities = np.array([x[0] for x in results])
                    probabilities[(stimulus, genotype)] = state_probabilities
                    plot_state_probabilities(state_probabilities, state_names, "optimized_state_probabilities_%s_%s" % (stimulus, genotype), figures_dir)
                    np.savetxt("%s/%s_state_probabilities_optimized_%s_%s.csv" % (results_dir, model, stimulus, genotype), state_probabilities, delimiter=",")
    else:
        print("Skipping state probabilities.", flush=True)

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
