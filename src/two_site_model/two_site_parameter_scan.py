# Optimize the parameters of the model: t parameters and k parameters
# Initial parameters from parameter scan

from two_site_model import *
# from three_site_model_figures import plot_parameters, plot_predictions, plot_parameter_distributions
import matplotlib.pyplot as plt
import pandas as pd
import os
import scipy.optimize as opt
import time
from multiprocessing import Pool
import argparse
import seaborn as sns
import scipy.stats.qmc as qmc

def get_N_I_P(data, stimulus, genotype):
    row = data.loc[(data["Stimulus"] == stimulus) & (data["Genotype"] == genotype)]
    N = row["NFkB"].values[0]
    I = row["IRF"].values[0]
    P = row["p50"].values[0]
    return N, I, P

def get_state_prob(t_pars, k_pars, N, I, c_par=None, h_pars=None):
    model = two_site(t_pars, k_pars, c_par=c_par, h_pars=h_pars)
    model.calculateState(N, I)
    model.calculateProb()
    probabilities = model.prob
    
    # 1, I, Ig, N,  I*Ig, I*N, Ig*N, I*Ig*N
    state_names = ["none", r"$IRF$", r"$IRF_G$", r"$NF\kappa B$", r"$IRF\cdot IRF_G$", 
                   r"$IRF\cdot NF\kappa B$", r"$IRF_G\cdot NF\kappa B$", r"$IRF\cdot IRF_G\cdot NF\kappa B$"]

    return probabilities, state_names

def calc_state_prob(k_pars, N, I, P, num_t=6, h_pars=None):
    # print(N, I, P, flush=True)
    t_pars = [1 for _ in range(num_t)]
    probabilities, state_names = get_state_prob(t_pars, k_pars, N, I, P, h_pars=h_pars)
    return probabilities, state_names

def plot_state_probabilities(state_probabilities, state_names, name, figures_dir):
        stimuli = ["basal", "CpG", "LPS", "polyIC"]
        stimulus = [s for s in stimuli if s in name]
        if len(stimulus) == 0:
            stimulus = "No Stim"
        elif len(stimulus) > 1:
            raise ValueError("More than one stimulus in name")
        else:
            stimulus = stimulus[0]

        condition = name.split("_")[-2:]
        condition = " ".join(condition)
        df_state_probabilities = pd.DataFrame(state_probabilities, columns=state_names)
        df_state_probabilities["par_set"] = np.arange(len(df_state_probabilities))
        df_state_probabilities = df_state_probabilities.melt(var_name="State", value_name="Probability", id_vars="par_set")

        with sns.plotting_context("talk", rc={"lines.markersize": 7}):
            fig, ax = plt.subplots(figsize=(6,5))
            p = sns.lineplot(data=df_state_probabilities, x = "State", y="Probability", color="black", alpha=0.2,
                                estimator=None, units="par_set", legend=False).set_title(condition)
            sns.scatterplot(data=df_state_probabilities, x = "State", y="Probability", color="black", alpha=0.2, ax=ax, legend=False, zorder=10)
            sns.despine()
            plt.xticks(rotation=90)
            # Save plot
            plt.savefig("%s/%s.png" % (figures_dir, name), bbox_inches="tight")
def calculate_rmsd(ifnb_predicted, beta):
    residuals = ifnb_predicted - beta
    rmsd = np.sqrt(np.mean(residuals**2))
    return rmsd


def calculate_grid(training_data, hi=1, hn=1, t_bounds=(0,1), k_bounds=(10**-3,10**3), seed=0, num_pts=11, num_samples=10**6, num_threads=60, c_par=False):
    min_k_order = np.log10(k_bounds[0])
    max_k_order = np.log10(k_bounds[1])
    min_c_order = np.log10(10**-3)
    max_c_order = np.log10(10**3)
    min_t = t_bounds[0]
    max_t = t_bounds[1]
    num_t_pars = 2
    num_k_pars = 2

    seed += 10

    print("Calculating grid with %d evenly spaced points for each parameter" % num_pts, flush=True)
    if not c_par:
        # make all combinations of t, k, and h parameters
        t_vals = np.linspace(t_bounds[0], t_bounds[1], num_pts)
        k_vals = np.logspace(min_k_order, max_k_order, num_pts)
        grid = np.array(np.meshgrid(t_vals, t_vals, k_vals, k_vals)).T.reshape(-1,num_t_pars+num_k_pars)
        print("Total number of samples: %d" % len(grid), flush=True)

        # sampler=qmc.LatinHypercube(d=num_t_pars+num_k_pars, seed=seed)
        # grid_tk = sampler.random(n=num_samples)
        # grid_tk = qmc.scale(grid_tk, l_bounds, u_bounds) # rows are parameter sets
        # # convert k parameters to log space
        # kgrid = grid_tk[:,num_t_pars:]
        # kgrid = 10**kgrid
        # grid_tk[:,num_t_pars:] = kgrid
    else:
        t_vals = np.linspace(t_bounds[0], t_bounds[1], num_pts)
        k_vals = np.logspace(min_k_order, max_k_order, num_pts)
        c_vals = np.logspace(min_c_order, max_c_order, num_pts)
        grid = np.array(np.meshgrid(t_vals, t_vals, k_vals, k_vals, c_vals)).T.reshape(-1,num_t_pars+num_k_pars+1)
        print("Total number of samples: %d" % len(grid), flush=True)

    # Load data points
    N, I, P = training_data["NFkB"], training_data["IRF"], training_data["p50"]
    beta = training_data["IFNb"]

    # Calculate IFNb value at each point in grid
    print("Calculating effects at %d points in grid" % len(grid), flush=True)
    start = time.time()
    if c_par:
        with Pool(num_threads) as p:
            results = p.starmap(get_f, [(N[j], I[j], None, grid[i,0:num_t_pars], grid[i,num_t_pars:-1], [hi, hn], grid[i,-1]) for i in range(len(grid)) for j in range(len(N))])
    else:
        with Pool(num_threads) as p:
            # def get_f(N, I, model=None, t=None, k=None, h=0, C=1):
            results = p.starmap(get_f, [(N[j], I[j], None, grid[i,0:num_t_pars], grid[i,num_t_pars:], [hi, hn]) for i in range(len(grid)) for j in range(len(N))])

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
    t_pars = pars[0:2]
    k_pars = pars[2:4]
    if c_par:
        c = pars[4]
    else:
        c = 1

    num_pts = len(N)
    
    ifnb_predicted = [get_f(N[j], I[j], None, t_pars, k_pars, h_pars, c) for j in range(num_pts)]
    rmsd = calculate_rmsd(ifnb_predicted, beta)

    return rmsd

def minimize_objective(pars, N, I, beta, h_pars, c_par, bounds):
    return opt.minimize(objective_function, pars, args=(N, I, beta, h_pars, c_par), method="Nelder-Mead", bounds=bounds)

def optimize_model(N, I, beta, initial_pars, h, c=False, num_threads=40, num_t_pars=2, num_k_pars=2):
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
            # ifnb_predicted = p.starmap(get_f, [(final_pars[i,0:num_t_pars], final_pars[i,num_t_pars:-1], N[j], I[j], final_pars[i,-1], h, False) for i in range(len(final_pars)) for j in range(len(N))])
            # def get_f(N, I, model=None, t=None, k=None, h=0, C=1):
            ifnb_predicted = p.starmap(get_f, [(N[j], I[j], None, final_pars[i,0:num_t_pars], final_pars[i,num_t_pars:-1], h, final_pars[i,-1]) for i in range(len(final_pars)) for j in range(len(N))])
    else:
        with Pool(num_threads) as p:
            # ifnb_predicted = p.starmap(get_f, [(final_pars[i,0:num_t_pars], final_pars[i,num_t_pars:], N[j], I[j], None, h, False) for i in range(len(final_pars)) for j in range(len(N))])
            ifnb_predicted = p.starmap(get_f, [(N[j], I[j], None, final_pars[i,0:num_t_pars], final_pars[i,num_t_pars:], h) for i in range(len(final_pars)) for j in range(len(N))])

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
    parser.add_argument("-i","--hi", type=int, default=1)
    parser.add_argument("-n","--hn", type=int, default=1)
    args = parser.parse_args()

    start_start = time.time()

    print("###############################################\n")
    print("Starting at %s\n" % time.ctime(), flush=True)

    # Settings    
    num_threads = 40
    model = "two_site"
    par_type = "k"
    hi = args.hi
    hn = args.hn
    h_val = "%d_%d" % (hi, hn)
    print("h_I: %d, h_N: %d" % (hi, hn), flush=True)
    c_par= args.c_parameter

    # Model details
    num_t_pars = 2
    num_k_pars = 2
    # num_c_pars = 1
    # num_h_pars = 2

    # Directories
    figures_dir = "parameter_scan/figures/"
    results_dir = "parameter_scan/results/"
    figures_dir = figures_dir[:-1] + "_h_%s/" % h_val
    results_dir = results_dir[:-1] + "_h_%s/" % h_val
    if c_par:
        figures_dir = figures_dir[:-1] + "_c_scan/"
        results_dir = results_dir[:-1] + "_c_scan/"
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
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
        result_par_names = ["t1","t2", "k1", "k2","c"]
    else:
        result_par_names = ["t1","t2", "k1", "k2"]

    # Optimize
    if args.param_scan:
        # Perform parameter scan
        print("Performing parameter scan...", flush=True)
        final_pars, ifnb_predicted, rmsd = calculate_grid(training_data, hi=hi, hn=hn, t_bounds=(0,1), k_bounds=(10**-3,10**3), seed=0, 
                                                          num_pts=11, num_samples=10**6, num_threads=num_threads, c_par=c_par)
        print("Finished parameter scan.", flush=True)

        
        print("Plotting results...", flush=True)
        final_pars_df = pd.DataFrame(final_pars, columns=result_par_names)
        # plot_parameters(final_pars_df, name="scanned_parameters", figures_dir=figures_dir)
        final_pars_df.to_csv("%s/%s_scanned_parameters.csv" % (results_dir, model), index=False)


        # Plot predictions
        print("Plotting predictions...", flush=True)

        # plot_predictions(ifnb_predicted, beta, conditions, name="scanned_predictions", figures_dir=figures_dir)
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

        final_pars, ifnb_predicted, rmsd = optimize_model(N, I, beta, initial_pars, [hi, hn], c=c_par, num_threads=num_threads, num_t_pars=num_t_pars, num_k_pars=num_k_pars)
        print("Finished optimization.", flush=True)

        print("Plotting results...", flush=True)
        final_pars_df = pd.DataFrame(final_pars, columns=result_par_names)
        # plot_parameters(final_pars_df, name="optimized_parameters", figures_dir=figures_dir)
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
        # plot_predictions(ifnb_predicted, beta, conditions, name="optimized_predictions", figures_dir=figures_dir)
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
        # plot_predictions(best_fits_results, beta, conditions, name="best_fits_predictions", figures_dir=figures_dir)
        # plot_parameters(best_fits_df, name="best_fits_parameters", figures_dir=figures_dir)

    # Plot state probabilities
    if args.state_probabilities:
        raise NotImplementedError("State probabilities not implemented yet.")
        # final_pars = pd.read_csv("%s/%s_best_fits_pars.csv" % (results_dir, model)).values
        # print("Calculating and plotting state probabilities...", flush=True)
        # all_k_pars = final_pars[:, num_t_pars:num_t_pars+num_k_pars]
        # h_pars = [h1, h2, hn]

        # extra_training_data = pd.DataFrame({"Stimulus":"basal", "Genotype":"WT", "IRF":0.01, "NFkB":0.01, "p50":1}, index=[0])
        # training_data_extended = pd.concat([training_data, extra_training_data], ignore_index=True)
        # stimuli = training_data_extended["Stimulus"]
        # genotypes = training_data_extended["Genotype"]

        # probabilities = dict()
        # for stimulus, genotype in zip(stimuli, genotypes):
        #             nfkb, irf, p50 = get_N_I_P(training_data_extended, stimulus, genotype)
        #             print("Calculating state probabilities for %s %s" % (stimulus, genotype), flush=True)
        #             print("N=%.2f, I=%.2f, P=%.2f" % (nfkb, irf, p50), flush=True)
                
        #             with Pool(num_threads) as p:
        #                 results = p.starmap(calc_state_prob, [(tuple(all_k_pars[i]), nfkb, irf, p50, num_t_pars, h_pars) for i in range(len(all_k_pars))])
                
        #             state_names = results[0][1]
        #             state_probabilities = np.array([x[0] for x in results])
        #             probabilities[(stimulus, genotype)] = state_probabilities
        #             plot_state_probabilities(state_probabilities, state_names, "optimized_state_probabilities_%s_%s" % (stimulus, genotype), figures_dir)
        #             np.savetxt("%s/%s_state_probabilities_optimized_%s_%s.csv" % (results_dir, model, stimulus, genotype), state_probabilities, delimiter=",")
    else:
        print("Skipping state probabilities.", flush=True)

    end_end = time.time()
    t = end_end - start_start
    if t < 60:
        print("Total time elapsed for h_I=%d, h_N=%d, c=%s: %.2f seconds" % (hi, hn, c_par, t))
    elif t < 3600:
        print("Total time elapsed for h_I=%d, h_N=%d, c=%s: %.2f minutes" % (hi, hn, c_par, t/60))
    else:
        print("Total time elapsed for h_I=%d, h_N=%d, c=%s: %.2f hours" % (hi, hn, c_par, t/3600))

if __name__ == "__main__":
    main()
