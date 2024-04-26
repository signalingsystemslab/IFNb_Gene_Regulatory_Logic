# Perform grid search over parameter space to get all possible values of IFNb
# t parameters and k parameters are varied
from two_site_model import *
import matplotlib.pyplot as plt
import pandas as pd
import os
# import scipy.optimize as opt
import time
from multiprocessing import Pool
import argparse
import seaborn as sns
import scipy.stats.qmc as qmc
import matplotlib.colors as mcolors

figures_dir = "param_scan_2site/figures/"
results_dir = "param_scan_2site/results/"
os.makedirs(results_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)
num_t_pars = 2
num_k_pars = 2
num_h_pars = 2

def calculate_ifnb(pars, data):
    if len(pars) != num_t_pars + num_k_pars + num_h_pars:
        raise ValueError("Number of parameters (%d) does not match number of t, k, and h parameters (%d)" % (len(pars), num_t_pars + num_k_pars + num_h_pars))
    t_pars, k_pars, h_pars = pars[:num_t_pars], pars[num_t_pars:num_t_pars+num_k_pars], pars[num_t_pars+num_k_pars:]
    N, I = data["NFkB"], data["IRF"]
    ifnb = [get_f(n,i, model=None, t=t_pars, k=k_pars, h=h_pars, C=1) for n,i in zip(N,I)]
    ifnb = np.array(ifnb)
    return ifnb

def calculate_grid(training_data, t_bounds=(0,1), k_bounds=None, h_bounds=None, num_pts=11, seed=0, num_samples=10**6, num_threads=60):
    min_k_order = np.log10(k_bounds[0])
    max_k_order = np.log10(k_bounds[1])
    # seed += 10

    # l_bounds = np.concatenate([np.zeros(num_t_pars)*t_bounds[0], np.ones(num_k_pars)*min_k_order])
    # u_bounds = np.concatenate([np.ones(num_t_pars)*t_bounds[1], np.ones(num_k_pars)*max_k_order])

    # print("Calculating grid with %d samples using Latin Hypercube sampling" % num_samples, flush=True)
    # sampler=qmc.LatinHypercube(d=num_t_pars+num_k_pars, seed=seed)
    # grid_tk = sampler.random(n=num_samples)
    # grid_tk = qmc.scale(grid_tk, l_bounds, u_bounds) # rows are parameter sets
    # # convert k parameters to log space
    # kgrid = grid_tk[:,num_t_pars:]
    # kgrid = 10**kgrid
    # grid_tk[:,num_t_pars:] = kgrid

    # # Add h values to grid.
    # h_vals = np.arange(h_bounds[0], h_bounds[1]+1, 1)
    # for h in h_vals:
    #     hgrid = np.zeros((num_samples, num_h_pars)) + h
    #     grid_partial = np.array(np.concatenate([grid_tk, hgrid], axis=1))
    #     if h == h_vals[0]:
    #         grid = grid_partial
    #     else:
    #         grid = np.concatenate([grid, grid_partial], axis=0)
    # kgrid = grid[:,num_t_pars:]
    # print("Total number of samples after adding h parameters: %d" % len(grid), flush=True)

    # make all combinations of t, k, and h parameters
    t_vals = np.linspace(t_bounds[0], t_bounds[1], num_pts)
    k_vals = np.logspace(min_k_order, max_k_order, num_pts)
    h_vals = np.arange(h_bounds[0], h_bounds[1]+1, 1)
    grid = np.array(np.meshgrid(t_vals, t_vals, k_vals, k_vals, h_vals, h_vals)).T.reshape(-1,num_t_pars+num_k_pars+num_h_pars)
    kgrid = grid[:,num_t_pars:]
    print("Total number of samples: %d" % len(grid), flush=True)
    
    # Calculate IFNb value at each point in grid
    print("Calculating IFNb at %d points in grid" % len(grid), flush=True)
    start = time.time()
    with Pool(num_threads) as p:
        results = p.starmap(calculate_ifnb, [(pars, training_data) for pars in grid])

    end = time.time()
    t = end - start
    if t < 60*60:
        print("Time elapsed: %.2f minutes" % (t/60), flush=True)
    else:
        print("Time elapsed: %.2f hours" % (t/3600), flush=True)

    ifnb_predicted = np.array(results)

    if ifnb_predicted.shape[0] != len(grid):
        raise ValueError("Number of results does not match number of points in grid")
    if ifnb_predicted.shape[1] != len(training_data):
        raise ValueError("Number of results does not match number of training data points")

    print("Size of results: %s, grid: %s" % (ifnb_predicted.shape, grid.shape), flush=True)
    return ifnb_predicted, grid, kgrid

def make_predictions_data_frame(ifnb_predicted, beta, conditions):
    df_ifnb_predicted = pd.DataFrame(ifnb_predicted, columns=conditions)
    df_ifnb_predicted["par_set"] = np.arange(len(df_ifnb_predicted))
    df_ifnb_predicted = df_ifnb_predicted.melt(var_name="Data point", value_name=r"IFN$\beta$", id_vars="par_set")

    df_ifnb_predicted_data = pd.DataFrame({"Data point":conditions, r"IFN$\beta$":beta, "par_set":"Data"})
    df_ifnb_predicted = pd.concat([df_ifnb_predicted, df_ifnb_predicted_data], ignore_index=True)
    df_ifnb_predicted["Stimulus"] = df_ifnb_predicted["Data point"].str.split("_", expand=True)[0]
    df_ifnb_predicted["Genotype"] = df_ifnb_predicted["Data point"].str.split("_", expand=True)[1]

    # stimuli_levels = ["basal", "CpG", "LPS", "polyIC"]
    # genotypes_levels = ["WT", "irf3irf7KO", "irf3irf5irf7KO", "relacrelKO", "p50KO"]
    # df_ifnb_predicted["Stimulus"] = pd.Categorical(df_ifnb_predicted["Stimulus"], categories=stimuli_levels, ordered=True)
    # df_ifnb_predicted["Genotype"] = pd.Categorical(df_ifnb_predicted["Genotype"], categories=genotypes_levels, ordered=True)
                                                   
    df_ifnb_predicted["Stimulus"] = df_ifnb_predicted["Stimulus"].replace("polyIC", "PolyIC")
    df_ifnb_predicted["Genotype"] = df_ifnb_predicted["Genotype"].replace("relacrelKO", r"$rela^{-/-}crel^{-/-}$")
    df_ifnb_predicted["Genotype"] = df_ifnb_predicted["Genotype"].replace("irf3irf7KO", r"$irf3^{-/-}irf7^{-/-}$")
    df_ifnb_predicted["Genotype"] = df_ifnb_predicted["Genotype"].replace("irf3irf5irf7KO", r"$irf3^{-/-}irf5^{-/-}irf7^{-/-}$")
    df_ifnb_predicted["Genotype"] = df_ifnb_predicted["Genotype"].replace("p50KO", r"$nfkb1^{-/-}$")

    df_ifnb_predicted["Data point"] = df_ifnb_predicted["Stimulus"] + " " + df_ifnb_predicted["Genotype"]   

    stimuli_levels = ["Basal", "CpG", "LPS", "PolyIC"]
    genotypes_levels = [r"WT", r"$irf3^{-/-}irf7^{-/-}$", r"$irf3^{-/-}irf5^{-/-}irf7^{-/-}$", r"$rela^{-/-}crel^{-/-}$", r"$nfkb1^{-/-}$"]
    df_ifnb_predicted["Stimulus"] = pd.Categorical(df_ifnb_predicted["Stimulus"], categories=stimuli_levels, ordered=True)
    df_ifnb_predicted["Genotype"] = pd.Categorical(df_ifnb_predicted["Genotype"], categories=genotypes_levels, ordered=True)
    df_ifnb_predicted = df_ifnb_predicted.sort_values(["Stimulus", "Genotype"])
    return df_ifnb_predicted

def plot_predictions(ifnb_predicted, beta, conditions, subset="All",name="ifnb_predictions", figures_dir=figures_dir):
    if type(subset) == str:
        if subset == "All":
            subset = np.arange(len(ifnb_predicted))
        else:
            print(subset)
            raise ValueError("Subset must be a list of indices or 'All'")

    # print(subset, flush=True)
    ifnb_predicted = ifnb_predicted[subset,:]
    # print(ifnb_predicted.shape, flush=True)

    with sns.plotting_context("talk", rc={"lines.markersize": 7}):

        df_ifnb_predicted = make_predictions_data_frame(ifnb_predicted, beta, conditions)
        col = sns.color_palette("rocket", n_colors=7)[4]
        col = mcolors.rgb2hex(col) 
        fig, ax = plt.subplots(figsize=(6.5, 6))
        
        sns.scatterplot(data=df_ifnb_predicted, x="Data point", y=r"IFN$\beta$", 
                        color="black", alpha=0.5, ax=ax, zorder = 1, label="Predicted")
        sns.lineplot(data=df_ifnb_predicted.loc[df_ifnb_predicted["par_set"] != "Data"], x="Data point", y=r"IFN$\beta$", 
                        units="par_set", color="black", estimator=None, ax=ax, legend=False, zorder = 2, alpha=0.2)
        sns.lineplot(data=df_ifnb_predicted.loc[df_ifnb_predicted["par_set"] == "Data"], x="Data point", y=r"IFN$\beta$",
                        color=col, estimator=None, ax=ax, legend=False, zorder = 3)
        sns.scatterplot(data=df_ifnb_predicted.loc[df_ifnb_predicted["par_set"] == "Data"], x="Data point", y=r"IFN$\beta$", 
                        color=col, marker="o", ax=ax, zorder = 4, label="Observed")
        xticks = ax.get_xticks()
        labels = [item.get_text().replace(" ", "\n") for item in ax.get_xticklabels()]
        ax.set_xticks(xticks)
        ax.set_xticklabels(labels)
        sns.despine()
        plt.xticks(rotation=90)
        plt.tight_layout()
        sns.move_legend(ax, bbox_to_anchor=(1, 0.5), title=None, frameon=False, loc="center left")
        plt.savefig("%s/%s.png" % (figures_dir, name), bbox_inches="tight")
        plt.close()

def plot_parameters(pars, subset="All", name="parameters", figures_dir=figures_dir, param_names=None):
    if type(subset) == str:
        if subset == "All":
            subset = np.arange(len(pars))

    if param_names is None:
        par_names = ["t%d" % (i+1) for i in range(num_t_pars)] + ["k%d" % (i+1) for i in range(num_k_pars)] + ["h%d" % (i+1) for i in range(num_h_pars)]
    else:
        par_names = param_names

    df_pars = pd.DataFrame(pars[subset,:], columns=par_names)
    df_pars["par_set"] = np.arange(len(df_pars))
    # df_pars["kp_ratio"] = df_pars["k4"] / df_pars["k2"]
    df_pars = df_pars.melt(var_name="Parameter", value_name="Value", id_vars="par_set")
    df_pars["Parameter"] = df_pars["Parameter"].str.replace("k3", "kn")

    df_t_pars = df_pars[df_pars["Parameter"].str.startswith("t")]
    df_k_pars = df_pars[df_pars["Parameter"].str.startswith("k")]
    # df_kp_par = df_pars[df_pars["Parameter"].str.startswith("kp_")]
    # # remove kp and kp_ratio from k parameters
    # df_k_pars = df_k_pars[~df_k_pars["Parameter"].str.startswith("kp")]
    if "c" in df_pars["Parameter"].values:
        df_c_pars = df_pars[df_pars["Parameter"].str.startswith("c")]
        fig, ax = plt.subplots(1,3, figsize=(11,5), gridspec_kw={"width_ratios":[num_t_pars, num_k_pars, 1]})
        sns.lineplot(data=df_t_pars, x="Parameter", y="Value", units="par_set", color="black", alpha=0.5, estimator=None, ax=ax[0])
        sns.lineplot(data=df_k_pars, x="Parameter", y="Value", units="par_set", color="black", alpha=0.5, estimator=None, ax=ax[1])
        sns.lineplot(data=df_c_pars, x="Parameter", y="Value", units="par_set", color="black", alpha=0.5, estimator=None, ax=ax[2])
        ax[1].set_yscale("log")
        ax[2].set_yscale("log")
        sns.despine()
        plt.tight_layout()
    elif "h" in df_pars["Parameter"].values or "h1" in df_pars["Parameter"].values:
        df_h_pars = df_pars[df_pars["Parameter"].str.startswith("h")]
        fig, ax = plt.subplots(1,3, figsize=(12,5), gridspec_kw={"width_ratios":[num_t_pars, num_k_pars, num_h_pars]})
        sns.lineplot(data=df_t_pars, x="Parameter", y="Value", units="par_set", color="black", alpha=0.5, estimator=None, ax=ax[0])
        sns.lineplot(data=df_k_pars, x="Parameter", y="Value", units="par_set", color="black", alpha=0.5, estimator=None, ax=ax[1])
        sns.lineplot(data=df_h_pars, x="Parameter", y="Value", units="par_set", color="black", alpha=0.5, estimator=None, ax=ax[2])
        sns.scatterplot(data=df_h_pars, x="Parameter", y="Value", color="black", alpha=0.5, ax=ax[2])
        ax[1].set_yscale("log")
        sns.despine()
        plt.tight_layout()
    else:
        fig, ax = plt.subplots(1,2, figsize=(10,5), gridspec_kw={"width_ratios":[num_t_pars, num_k_pars]})
        sns.lineplot(data=df_t_pars, x="Parameter", y="Value", units="par_set", color="black", alpha=0.5, estimator=None, ax=ax[0])
        sns.lineplot(data=df_k_pars, x="Parameter", y="Value", units="par_set", color="black", alpha=0.5, estimator=None, ax=ax[1])
        ax[1].set_yscale("log")
        # sns.lineplot(data=df_kp_par, x="Parameter", y="Value", units="par_set", color="black", alpha=0.5, estimator=None, ax=ax[2])
        sns.despine()
        plt.tight_layout()

    plt.savefig("%s/%s.png" % (figures_dir, name))
    plt.close()

def calc_state_prob(khpars, N, I):
    # print(N, I, flush=True)
    k_pars = khpars[:num_k_pars]
    h_pars = khpars[num_k_pars:]
    t_pars = [1 for _ in range(num_t_pars)]
    probabilities, state_names = get_state_prob(N, I, t=t_pars, k=k_pars, h=h_pars)
    return probabilities, state_names

def plot_state_probabilities(state_probabilities, state_names, name, figures_dir=figures_dir):
    stimuli = ["basal", "CpG", "LPS", "polyIC"]
    stimulus = [s for s in stimuli if s in name]


    if len(stimulus) == 0:
        stimulus = "No Stim"
        condition = "No Stim"
    elif len(stimulus) > 1:
        raise ValueError("More than one stimulus in name")
    else:
        stimulus = stimulus[0]
        # Condition is text after stimulus_
        name_parts = name.split("_")
        stim_loc = name_parts.index(stimulus)
        cond_loc = stim_loc + 1
        genotype = name_parts[cond_loc]
        condition = "%s %s" % (stimulus, genotype)
    df_state_probabilities = pd.DataFrame(state_probabilities, columns=state_names)
    df_state_probabilities["par_set"] = np.arange(len(df_state_probabilities))
    df_state_probabilities = df_state_probabilities.melt(var_name="State", value_name="Probability", id_vars="par_set")

    fig, ax = plt.subplots()
    p = sns.lineplot(data=df_state_probabilities, x = "State", y="Probability", color="black", alpha=0.5,
                        estimator=None, units="par_set", legend=False).set_title(condition)
    sns.despine()
    plt.xticks(rotation=90)
    # Save plot
    plt.savefig("%s/%s.png" % (figures_dir, name), bbox_inches="tight")
    plt.close()

def plot_parameter_pairwise(pars_df, subset="All", name="parameter_distributions", figures_dir=figures_dir):
    if type(subset) == str:
        if subset == "All":
            subset = np.arange(len(pars_df))

    df_pars = pars_df.iloc[subset,:]
    df_pars = df_pars.drop(columns=["h%d" % (i+1) for i in range(num_h_pars)])
    # jitter points
    for par in df_pars.columns:
        if "t" in par:
            df_pars[par] = df_pars[par] + np.random.normal(0, 0.01, df_pars.shape[0])
        if "k" in par:
            df_pars[par] = df_pars[par] + np.random.normal(0, 1, df_pars.shape[0])
    p=sns.pairplot(df_pars, diag_kind="kde", plot_kws={"alpha":0.8}, diag_kws={"alpha":0.4, "hue":None, "color":"black","palette":None}, hue="rmsd", palette="viridis")
    for ax in p.axes.flatten():
        if "k" in ax.get_xlabel():
            ax.set_xscale("log")
            ax.set_xlim(10**-3*0.01, 10**3*10)
        if "k" in ax.get_ylabel():
            ax.set_yscale("log")
            ax.set_ylim(10**-3*0.01, 10**3*10)
        if "t" in ax.get_xlabel():
            ax.set_xlim(0-0.1,1+0.1)
        if "t" in ax.get_ylabel():
            ax.set_ylim(0-0.1,1+0.1)
    plt.savefig("%s/%s.png" % (figures_dir, name))
    plt.close()
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g","--calc_grid", action="store_true")
    parser.add_argument("-s","--calc_states", action="store_true")
    parser.add_argument("-n","--no_plot", action="store_true") 
    args = parser.parse_args()

    start_start = time.time()

    print("###############################################\n")
    print("Starting at %s\n" % time.ctime(), flush=True)

    training_data = pd.read_csv("../data/training_data.csv")
    print("Using the following training data:\n", training_data)
    model = "two_site_hill"
    results_directory = results_dir
    figures_directory = figures_dir
    t_bounds = (0,1)
    k_bounds = (10**-3, 10**3)
    h_bounds = (1, 5)
    state_names = ["none", "IRF", r"$NF\kappa B$", r"IRF and $NF\kappa B$"]

    beta = training_data["IFNb"]
    conditions = training_data["Stimulus"] + "_" + training_data["Genotype"]

    par_names = ["t%d" % (i+1) for i in range(num_t_pars)] + ["k%d" % (i+1) for i in range(num_k_pars)] + ["h%d" % (i+1) for i in range(num_h_pars)]

    num_threads = 50
    num_par_sets = 10**6

    # num_seeds = 1
    num_seeds = 3
    for seed in range(num_seeds):
        print("Seed: %d" % seed, flush=True)
        results_directory = results_directory.strip("/")
        if results_directory.split("/")[-1] == "results":
            results_directory = results_directory + "/seed_%d/" % seed
            figures_directory = figures_directory + "/seed_%d/" % seed
            # print("Results directory: %s" % os.path.normpath(results_directory), flush=True)
            print("Results directory: %s" % results_directory, flush=True)
        elif "seed" in results_directory.split("/")[-1]:
            results_directory = results_directory + "/../seed_%d/" % seed
            figures_directory = figures_directory + "/../seed_%d/" % seed
            print("Results directory: %s" % results_directory, flush=True)
            # print("Results directory: %s" % os.path.normpath(results_directory), flush=True)
        else:
            raise ValueError("Results directory %s not recognized" % results_directory)

        os.makedirs(results_directory, exist_ok=True)
        os.makedirs(figures_directory, exist_ok=True)

        if args.calc_grid:
            print("Calculating grid ON", flush=True)

            # calculate_grid(training_data, t_bounds=(0,1), k_bounds=None, h_bounds=None, num_pts=10, seed=0, num_samples=10**6, num_threads=60):
            # ifnb_predicted, pars, kgrid = calculate_grid(training_data, num_samples=num_par_sets, seed=seed, num_threads=num_threads)
            ifnb_predicted, pars, kgrid = calculate_grid(training_data, t_bounds=t_bounds, k_bounds=k_bounds, h_bounds=h_bounds, num_pts=11, seed=seed, num_samples=num_par_sets, num_threads=num_threads)

            np.savetxt("%s/%s_grid_pars.csv" % (results_directory, model), pars, delimiter=",")
            np.savetxt("%s/%s_grid_ifnb.csv" % (results_directory, model), ifnb_predicted, delimiter=",")
            np.savetxt("%s/%s_grid_kpars.csv" % (results_directory, model), kgrid, delimiter=",")

        if not args.calc_grid:
            pars = np.loadtxt("%s/%s_grid_pars.csv" % (results_directory, model), delimiter=",")
            ifnb_predicted = np.loadtxt("%s/%s_grid_ifnb.csv" % (results_directory, model), delimiter=",")
            kgrid = np.loadtxt("%s/%s_grid_kpars.csv" % (results_directory, model), delimiter=",")

        # Get all states
        extra_training_data = pd.DataFrame({"Stimulus":"basal", "Genotype":"WT", "IRF":0.001, "NFkB":0.001, "p50":1}, index=[0])
        training_data_extended = pd.concat([training_data, extra_training_data], ignore_index=True)
        
        stimuli = training_data_extended["Stimulus"]
        genotypes = training_data_extended["Genotype"]

        # Calculate residuals
        residuals = ifnb_predicted - np.stack([beta for _ in range(len(pars))])
        rmsd = np.sqrt(np.mean(residuals**2, axis=1))

        # Sort rmsd
        sorted_indices = np.argsort(rmsd)
        best_20 = sorted_indices[:20]
        best_20_params = pars[best_20]
        best_20_kpars = best_20_params[:,num_t_pars:]
        best_20_pars_df = pd.DataFrame(best_20_params, columns=par_names)
        best_20_pars_df["rmsd"] = rmsd[best_20]
        best_20_pars_df.to_csv("%s/%s_best_20_pars.csv" % (results_directory, model), index=False)
        best_20_ifnb = pd.DataFrame(ifnb_predicted[best_20], columns=conditions)
        best_20_ifnb.to_csv("%s/%s_best_20_ifnb.csv" % (results_directory, model), index=False)

        print("Finished calculating best 20 parameters overall", flush=True)

        # Get best 20 from each h value
        h_vals = np.unique(pars[:,-2:], axis=0)
        h_best_20_inds = {}
        for row in h_vals:
            h1, h2 = row
            h_indices = np.where((pars[:,-2] == h1) & (pars[:,-1] == h2))[0]
            h_rmsd = rmsd[h_indices]
            h_sorted_indices = h_indices[np.argsort(h_rmsd)]
            h_best_20_inds[(h1,h2)] = h_sorted_indices[:20]
            h_best_20 = h_sorted_indices[:20]
            h_best_20_df = pd.DataFrame(pars[h_best_20], columns=par_names)
            h_best_20_df["rmsd"] = rmsd[h_best_20]
            h_best_20_df.to_csv("%s/%s_best_20_pars_hi_%d_hn_%d.csv" % (results_directory, model, h1, h2), index=False)

        if args.calc_states:
            print("Calculating state probabilities ON", flush=True)
            for stimulus, genotype in zip(stimuli, genotypes):
                row = training_data_extended.loc[(training_data_extended["Stimulus"] == stimulus) & (training_data_extended["Genotype"] == genotype)]
                nfkb = row["NFkB"].values[0]
                irf = row["IRF"].values[0]

                print("Calculating state probabilities for %s %s" % (stimulus, genotype), flush=True)
                print("N=%.2f, I=%.2f" % (nfkb, irf), flush=True)
            
                with Pool(num_threads) as p:
                    results = p.starmap(calc_state_prob, [(tuple(pars[i,num_t_pars:]), nfkb, irf) for i in range(len(pars))])
            
                state_names = results[0][1]
                state_probabilities = np.array([x[0] for x in results])

                np.savetxt("%s/%s_%s_%s_state_probabilities.csv" % (results_directory, model, stimulus, genotype), state_probabilities, delimiter=",")

            np.savetxt("%s/%s_state_names.csv" % (results_directory, model), state_names, delimiter=",", fmt="%s")

        probabilities = {}
        # state_names = np.loadtxt("%s/%s_state_names.csv" % (results_directory, model), delimiter=",", dtype=str)
        for stimulus, genotype in zip(stimuli, genotypes):
            state_probabilities = np.loadtxt("%s/%s_%s_%s_state_probabilities.csv" % (results_directory, model, stimulus, genotype), delimiter=",")
            condition = "%s_%s" % (stimulus, genotype)
            probabilities[condition] = state_probabilities
            # print("Plotting state probabilities for %s %s" % (stimulus, genotype), flush=True)
            # plot_state_probabilities(state_probabilities, state_names, condition, figures_dir=figures_directory)

        if args.no_plot == False:
            plot_time_start = time.time()
            print("Plotting ON", flush=True)

            for h1, h2 in h_best_20_inds.keys():
                h_best_20 = h_best_20_inds[(h1,h2)]
                plot_parameters(pars, subset=h_best_20, name="parameters_best_20_hi_%d_hn_%d" % (h1, h2), figures_dir=figures_directory)
                plot_predictions(ifnb_predicted, beta, conditions, subset=h_best_20, name="ifnb_predictions_best_20_hi_%d_hn_%d" % (h1, h2), figures_dir=figures_directory)
                for cond in probabilities.keys():
                    plot_state_probabilities(probabilities[cond][h_best_20], state_names, "%s_%s_best_20_state_probabilities_hi_%d_hn_%d" % (model, cond, h1, h2), figures_dir=figures_directory)

            # print("Plotting state probabilities", flush=True)
            # for cond in probabilities.keys():
            #     plot_state_probabilities(probabilities[cond][best_20], state_names,
            #                             "%s_%s_best_20_state_probabilities" % (model, cond), figures_dir=figures_directory)
            # print("Done plotting best 20 state probabilities", flush=True)

            # # Plot best 20 ifnb predictions
            # print("Plotting best 20 IFNb predictions", flush=True)
            # plot_predictions(ifnb_predicted, beta, conditions, subset=best_20, name="ifnb_predictions_best_20", figures_dir=figures_directory)
            # print("Done plotting best 20 IFNb predictions", flush=True)

            # # Plot best 20 parameters
            # print("Plotting best 20 parameters", flush=True)
            # plot_parameters(pars, subset=best_20, name="parameters_best_20", figures_dir=figures_directory)

            # # Plot best 20 predictions and parameters for each h value
            # print("Plotting best 20 predictions and parameters for each h value", flush=True)
            # for h in h_vals:
            #     h_best_20 = h_best_20_inds[h]
            #     plot_predictions(ifnb_predicted, beta, conditions, subset=h_best_20, name="ifnb_predictions_best_20_h_%d" % h, figures_dir=figures_directory)
            #     plot_parameters(pars, subset=h_best_20, name="parameters_best_20_h_%d" % h, figures_dir=figures_directory)


            # # Plot distributions of all parameters
            # print("Plotting all parameter distributions", flush=True)
            # t = time.time()
            # plot_parameter_distributions(pars, subset="All", name="all_parameter_distributions", figures_dir=figures_directory)
            # t = time.time() - t
            # print("Time elapsed: %.2f minutes" % (t/60), flush=True)

            plot_time = time.time() - plot_time_start
            print("Total time elapsed for plotting: %.2f minutes" % (plot_time/60), flush=True)


    # Combine best 20 parameters for each seed
    print("Combining best 20 parameters for each seed", flush=True)
    all_best_pars = pd.DataFrame()
    for seed in range(num_seeds):
        results_directory = "%s/seed_%d" % (results_dir, seed)
        best_20_pars_df = pd.read_csv("%s/%s_best_20_pars.csv" % (results_directory, model))
        all_best_pars = pd.concat([all_best_pars, best_20_pars_df], ignore_index=True)

    all_best_pars.to_csv("%s/%s_all_best_20_pars.csv" % (results_dir, model), index=False)

    # Combine best 20 parameters for each h value for each seed
    print("Combining best 20 parameters for each h value for each seed", flush=True)
    for h1, h2 in h_best_20_inds.keys():
        t = time.time()
        all_best_h_pars = pd.DataFrame()
        for seed in range(num_seeds):
            results_directory = "%s/seed_%d" % (results_dir, seed)
            best_20_pars_df = pd.read_csv("%s/%s_best_20_pars_hi_%d_hn_%d.csv" % (results_directory, model, h1, h2))
            all_best_h_pars = pd.concat([all_best_h_pars, best_20_pars_df], ignore_index=True)

        all_best_h_pars.to_csv("%s/%s_all_best_20_pars_hi_%d_hn_%d.csv" % (results_dir, model, h1, h2), index=False)
        
        plot_parameter_pairwise(all_best_h_pars, subset="All", name="parameter_pairplots_hi_%d_hn_%d" % (h1, h2), figures_dir=figures_dir)
        print("Time elapsed for h_i=%d, h_n=%d: %.2f minutes" % (h1, h2, (time.time()-t)/60), flush=True)

    end_end = time.time()
    t = end_end - start_start
    print("\n###############################################")
    if t < 60*60:
        print("Total time elapsed: %.2f minutes" % (t/60), flush=True)
    else:
        print("Total time elapsed: %.2f hours" % (t/3600), flush=True)

if __name__ == "__main__":
    main()
