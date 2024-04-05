# Perform grid search over parameter space to get all possible values of IFNb
# t parameters and k parameters are varied
from three_site_model import *
import matplotlib.pyplot as plt
import pandas as pd
import os
# import scipy.optimize as opt
import time
from multiprocessing import Pool
import argparse
import seaborn as sns
import scipy.stats.qmc as qmc

figures_dir = "three_site_param_scan_hill/figures"
results_dir = "three_site_param_scan_hill/results"
os.makedirs(results_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)
num_t_pars = 5
num_k_pars = 3
num_h_pars = 3

# # Set seaborn context
# sns.set_context("talk")

def calculate_ifnb(pars, data):
    if len(pars) != num_t_pars + num_k_pars + num_h_pars:
        raise ValueError("Number of parameters (%d) does not match number of t, k, and h parameters (%d)" % (len(pars), num_t_pars + num_k_pars + num_h_pars))
    t_pars, k_pars, h_pars = pars[:num_t_pars], pars[num_t_pars:num_t_pars+num_k_pars], pars[num_t_pars+num_k_pars:]
    N, I = data["NFkB"], data["IRF"]
    ifnb = [get_f(t_pars, k_pars, n, i, h_pars=h_pars) for n, i, in zip(N, I)]
    ifnb = np.array(ifnb)
    return ifnb

def calculate_grid(training_data, t_bounds=(0,1), k_bounds=(10**-3,10**3), h_bounds=(3,3), seed=0, num_samples=10**6, num_threads=60):
    min_k_order = np.log10(k_bounds[0])
    max_k_order = np.log10(k_bounds[1])
    min_t = t_bounds[0]
    max_t = t_bounds[1]

    seed += 10

    l_bounds = np.concatenate([np.zeros(num_t_pars)+min_t, np.ones(num_k_pars)*min_k_order])
    u_bounds = np.concatenate([np.zeros(num_t_pars)+max_t, np.ones(num_k_pars)*max_k_order])

    print("Calculating grid with %d samples using Latin Hypercube sampling" % num_samples, flush=True)
    sampler=qmc.LatinHypercube(d=num_t_pars+num_k_pars, seed=seed)
    grid_tk = sampler.random(n=num_samples)
    grid_tk = qmc.scale(grid_tk, l_bounds, u_bounds) # rows are parameter sets
    # convert k parameters to log space
    kgrid = grid_tk[:,num_t_pars:]
    kgrid = 10**kgrid
    grid_tk[:,num_t_pars:] = kgrid

    # Add h values to grid.
    h_vals = np.arange(h_bounds[0], h_bounds[1]+2, 2)
    # make all possible combinations of h values
    h_combs = np.array(np.meshgrid(*[h_vals for _ in range(num_h_pars)])).T.reshape(-1,num_h_pars)
    for h in h_combs:
        hgrid = np.zeros((num_samples, num_h_pars)) + h
        grid_partial = np.array(np.concatenate([grid_tk, hgrid], axis=1))
        if "grid" not in locals():
            grid = grid_partial
        else:
            grid = np.concatenate([grid, grid_partial], axis=0)
    kgrid = grid[:,num_t_pars:]
    print("Total number of samples after adding h parameters: %d" % len(grid), flush=True)

    grid = grid.astype(np.float32)

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
    del results

    if ifnb_predicted.shape[0] != len(grid):
        raise ValueError("Number of results does not match number of points in grid")
    if ifnb_predicted.shape[1] != len(training_data):
        raise ValueError("Number of results does not match number of training data points")

    print("Size of results: %s, grid: %s" % (ifnb_predicted.shape, grid.shape), flush=True)
    return ifnb_predicted, grid, kgrid

def jitter_dots(dots, jitter=0.3, y_jitter=False):
    offsets = dots.get_offsets()
    jittered_offsets = offsets
    # only jitter in the x-direction
    jittered_offsets[:, 0] += np.random.uniform(-jitter, jitter, size=offsets.shape[0])
    if y_jitter:
        jittered_offsets[:, 1] += np.random.uniform(-jitter, jitter, size=offsets.shape[0])
    dots.set_offsets(jittered_offsets)
    return dots

def get_N_I_P(data, stimulus, genotype):
    row = data.loc[(data["Stimulus"] == stimulus) & (data["Genotype"] == genotype)]
    N = row["NFkB"].values[0]
    I = row["IRF"].values[0]
    return N, I

def calc_state_prob(khpars, N, I):
    # print(N, I, flush=True)
    k_pars = khpars[:num_k_pars]
    h_pars = khpars[num_k_pars:]
    t_pars = [1 for _ in range(num_t_pars)]
    probabilities, state_names = get_state_prob(t_pars, k_pars, N, I, h_pars=h_pars)
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

def plot_predictions(ifnb_predicted, beta, conditions, subset="All",name="ifnb_predictions", figures_dir=figures_dir):
    if type(subset) == str:
        if subset == "All":
            subset = np.arange(len(ifnb_predicted))
        else:
            print(subset)
            raise ValueError("Subset must be a list of indices or 'All'")

    df_ifnb_predicted = pd.DataFrame(ifnb_predicted, columns=conditions)
    df_ifnb_predicted["par_set"] = np.arange(len(df_ifnb_predicted))
    df_ifnb_predicted = df_ifnb_predicted.melt(var_name="Data point", value_name=r"IFN$\beta$", id_vars="par_set")

    df_ifnb_predicted_data = pd.DataFrame({"Data point":conditions, r"IFN$\beta$":beta, "par_set":"Data"})
    df_ifnb_predicted = pd.concat([df_ifnb_predicted, df_ifnb_predicted_data], ignore_index=True)
    df_ifnb_predicted["Stimulus"] = df_ifnb_predicted["Data point"].str.split("_", expand=True)[0]
    df_ifnb_predicted["Genotype"] = df_ifnb_predicted["Data point"].str.split("_", expand=True)[1]
    stimuli_levels = ["basal", "CpG", "LPS", "polyIC"]
    genotypes_levels = ["WT", "irf3irf7KO", "irf3irf5irf7KO", "relacrelKO"]
    df_ifnb_predicted["Stimulus"] = pd.Categorical(df_ifnb_predicted["Stimulus"], categories=stimuli_levels, ordered=True)
    df_ifnb_predicted["Genotype"] = pd.Categorical(df_ifnb_predicted["Genotype"], categories=genotypes_levels, ordered=True)
    df_ifnb_predicted = df_ifnb_predicted.sort_values(["Stimulus", "Genotype"])

    fig, ax = plt.subplots()
    sns.lineplot(data=df_ifnb_predicted.loc[df_ifnb_predicted["par_set"].isin(subset)], x="Data point", y=r"IFN$\beta$", 
                    units="par_set", color="black", alpha=0.5, estimator=None, ax=ax)
    sns.scatterplot(data=df_ifnb_predicted.loc[df_ifnb_predicted["par_set"] == "Data"], x="Data point", y=r"IFN$\beta$", 
                    color="red", marker="o", ax=ax, legend=False, zorder = 10)
    sns.despine()
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig("%s/%s.png" % (figures_dir, name))
    plt.close()

    # # Plot predictions on log scale
    # fig, ax = plt.subplots()
    # sns.lineplot(data=df_ifnb_predicted.loc[df_ifnb_predicted["par_set"].isin(subset)], x="Data point", y=r"IFN$\beta$",
    #                 units="par_set", color="black", alpha=0.5, estimator=None, ax=ax)
    # sns.scatterplot(data=df_ifnb_predicted.loc[df_ifnb_predicted["par_set"] == "Data"], x="Data point", y=r"IFN$\beta$",
    #                 color="red", marker="o", ax=ax, legend=False, zorder = 10)
    # sns.despine()
    # plt.xticks(rotation=90)
    # plt.yscale("log")
    # ax.set_ylim(bottom=0.01)
    # plt.tight_layout()
    # plt.savefig("%s/%s_log.png" % (figures_dir, name))
    # plt.close()

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
    elif "h1" in df_pars["Parameter"].values:
        df_h_pars = df_pars[df_pars["Parameter"].str.startswith("h")]
        fig, ax = plt.subplots(1,3, figsize=(12,5), gridspec_kw={"width_ratios":[num_t_pars, num_k_pars, num_h_pars]})
        sns.lineplot(data=df_t_pars, x="Parameter", y="Value", units="par_set", color="black", alpha=0.5, estimator=None, ax=ax[0])
        sns.lineplot(data=df_k_pars, x="Parameter", y="Value", units="par_set", color="black", alpha=0.5, estimator=None, ax=ax[1])
        sns.lineplot(data=df_h_pars, x="Parameter", y="Value", units="par_set", color="black", alpha=0.5, estimator=None, ax=ax[2])
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

def plot_parameters_wrapper(results_directory, model, h_vals_str, figures_directory, subset="All", name="parameters", figures_dir=figures_dir, param_names=None):
    # Load data and plot parameters
    # h_best_20_df.to_csv("%s/%s_best_20_pars_h_%s.csv" % (results_directory, model, h_vals_str), index=False)
    pars = pd.read_csv("%s/%s_all_best_20_pars_h_%s.csv" % (results_directory, model, h_vals_str))
    pars = pars.values
    plot_parameters(pars, subset=subset, name=name, figures_dir=figures_dir, param_names=param_names)


def plot_parameter_distributions(pars, subset ="All", name="parameter_distributions", figures_dir=figures_dir, param_names=None):
    if param_names is None:
        par_names = ["t%d" % (i+1) for i in range(num_t_pars)] + ["k%d" % (i+1) for i in range(num_k_pars)] + ["h%d" % (i+1) for i in range(num_h_pars)]
    else:
        par_names = param_names
    df_pars = pd.DataFrame(pars, columns=par_names)
    df_pars["par_set"] = np.arange(len(df_pars))
    
    t_pars = [par for par in par_names if par.startswith("t")]
    k_pars = [par for par in par_names if par.startswith("k")]
    c_pars = [par for par in par_names if par.startswith("c")]
    h_pars = [par for par in par_names if par.startswith("h")]

    df_t_pars = df_pars.loc[:,t_pars + ["par_set"]]
    df_k_pars = df_pars.loc[:,k_pars + ["par_set"]]

    colors = ["#bdbdbd", "#fbb4ae"]

    if type(subset) == str:
        if subset == "All":
            df_t_pars = df_t_pars.melt(var_name="Parameter", value_name="Value", id_vars="par_set")

            sns.displot(data=df_t_pars, x="Value", col="Parameter", fill=True, alpha=0.5, color=colors[0], kind="kde", col_wrap=3)
            sns.despine()
            plt.tight_layout()
            plt.savefig("%s/%s_t_pars.png" % (figures_dir, name))
            plt.close()

            df_k_pars = df_k_pars.melt(var_name="Parameter", value_name="Value", id_vars="par_set")
            sns.displot(data=df_k_pars, x="Value", col="Parameter", fill=True, alpha=0.5, color=colors[0], kind="kde", log_scale=(True, False))
            sns.despine()
            plt.tight_layout()
            plt.savefig("%s/%s_k_pars.png" % (figures_dir, name))
            plt.close()

            if len(c_pars) > 0:
                df_c_pars = df_pars.loc[:,c_pars + ["par_set"]]
                df_c_pars = df_c_pars.melt(var_name="Parameter", value_name="Value", id_vars="par_set")
                sns.displot(data=df_c_pars, x="Value", col="Parameter", fill=True, alpha=0.5, color=colors[0], kind="kde", log_scale=(True, False))
                sns.despine()
                plt.tight_layout()
                plt.savefig("%s/%s_c_pars.png" % (figures_dir, name))
                plt.close()

            if len(h_pars) > 0:
                df_h_pars = df_pars.loc[:,h_pars + ["par_set"]]
                df_h_pars = df_h_pars.melt(var_name="Parameter", value_name="Value", id_vars="par_set")
                sns.displot(data=df_h_pars, x="Value", col="Parameter", fill=True, alpha=0.5, color=colors[0], kind="hist")
                sns.despine()
                plt.tight_layout()
                plt.savefig("%s/%s_h_pars.png" % (figures_dir, name))
                plt.close()
        else:
            raise ValueError("Subset must be a list of indices or 'All'")
    else:
        
        df_t_pars["subset"] = [True if i in subset else False for i in range(len(df_pars))]
        print("There are %d points in the subset" % len(df_t_pars.loc[:, "subset"] == True))
        df_t_pars = df_t_pars.melt(var_name="Parameter", value_name="Value", id_vars=["par_set", "subset"])

        sns.displot(data=df_t_pars, x="Value", hue="subset", col="Parameter", fill=True, alpha=0.5, palette=colors, kind="kde", col_wrap=3, common_norm=False)
        sns.despine()
        plt.tight_layout()
        plt.savefig("%s/%s_t_pars.png" % (figures_dir, name))
        plt.close()

        df_k_pars["subset"] = [True if i in subset else False for i in range(len(df_pars))]
        df_k_pars = df_k_pars.melt(var_name="Parameter", value_name="Value", id_vars=["par_set", "subset"])
        sns.displot(data=df_k_pars, x="Value", hue="subset", col="Parameter", fill=True, alpha=0.5, palette=colors, kind="kde", log_scale=(True, False), common_norm=False)
        sns.despine()
        plt.tight_layout()
        plt.savefig("%s/%s_k_pars.png" % (figures_dir, name))
        plt.close()

def plot_parameter_pairwise(pars_df, subset="All", name="parameter_distributions", figures_dir=figures_dir):
    if type(subset) == str:
        if subset == "All":
            subset = np.arange(len(pars_df))

    df_pars = pars_df.iloc[subset,:]
    h_cols = [col for col in df_pars.columns if "h" in col]
    df_pars = df_pars.drop(columns=h_cols)
    # # jitter points
    # for par in df_pars.columns:
    #     if "t" in par:
    #         df_pars[par] = df_pars[par] + np.random.normal(0, 0.01, df_pars.shape[0])
    #     if "k" in par:
    #         df_pars[par] = df_pars[par] + np.random.normal(0, 1, df_pars.shape[0])
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

def combine_pars(row, num_seeds, model, results_dir, figures_directory):
    t = time.time()
    h_vals_str = "_".join([str(int(x)) for x in row])
    all_best_h_pars = pd.DataFrame()
    for seed in range(num_seeds):
        results_directory = "%s/seed_%d" % (results_dir, seed)
        best_20_pars_df = pd.read_csv("%s/%s_best_20_pars_h_%s.csv" % (results_directory, model, h_vals_str))
        all_best_h_pars = pd.concat([all_best_h_pars, best_20_pars_df], ignore_index=True)
    all_best_h_pars.to_csv("%s/%s_all_best_20_pars_h_%s.csv" % (results_dir, model, h_vals_str), index=False)
    plot_parameter_pairwise(all_best_h_pars, subset="All", name="combined_parameter_pairplots_h_%s" % h_vals_str, figures_dir=figures_directory)
    print("Plotted %s after %.2f minutes, saved to %s" % (h_vals_str, (time.time() - t)/60, figures_directory), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g","--calc_grid", action="store_true")
    parser.add_argument("-s","--calc_states", action="store_true")
    parser.add_argument("-n","--no_plot", action="store_true")
    parser.add_argument("-d","--seed", type=int, default=-1)
    args = parser.parse_args()

    start_start = time.time()

    print("###############################################\n")
    print("Starting at %s\n" % time.ctime(), flush=True)

    training_data = pd.read_csv("../data/training_data.csv")
    print("Using the following training data:\n", training_data)
    model = "three_site_only_hill"
    results_directory = results_dir
    figures_directory = figures_dir

    # N = training_data["NFkB"]
    # I = training_data["IRF"]
    beta = training_data["IFNb"]
    conditions = training_data["Stimulus"] + "_" + training_data["Genotype"]
    # num_pts = len(beta)
    # len_training = len(beta)
    par_names = ["t%d" % (i+1) for i in range(num_t_pars)] + ["k%d" % (i+1) for i in range(num_k_pars)] + ["h%d" % (i+1) for i in range(num_h_pars)]

    # num_pars = num_t_pars + num_k_pars
    num_threads = 60
    num_plot_threads = 20
    num_par_sets = 10**6
    hmin = 1
    hmax = 5

    num_seeds = 3
    if args.seed >= 0:
        num_seeds = args.seed
    for seed in range(num_seeds):
        if args.seed >= 0:
            if seed != args.seed:
                print("Skipping seed %d" % seed, flush=True)
                break
        print("Seed: %d" % seed, flush=True)
        if results_directory.split("/")[-1] == "results":
            results_directory = results_directory + "/seed_%d" % seed
            figures_directory = figures_directory + "/seed_%d" % seed
            # print("Results directory: %s" % os.path.normpath(results_directory), flush=True)
            print("Results directory: %s" % results_directory, flush=True)
        elif "seed" in results_directory.split("/")[-1]:
            results_directory = results_directory + "/../seed_%d" % seed
            figures_directory = figures_directory + "/../seed_%d" % seed
            print("Results directory: %s" % results_directory, flush=True)
            # print("Results directory: %s" % os.path.normpath(results_directory), flush=True)

        else:
            raise ValueError("Results directory %s not recognized" % results_directory)

        os.makedirs(results_directory, exist_ok=True)
        os.makedirs(figures_directory, exist_ok=True)

        if args.calc_grid:
            print("Calculating grid...", flush=True)

            ifnb_predicted, pars, kgrid = calculate_grid(training_data, num_samples=num_par_sets, h_bounds=(hmin,hmax), seed=seed, num_threads=num_threads)

            np.savetxt("%s/%s_grid_pars.csv" % (results_directory, model), pars, delimiter=",")
            np.savetxt("%s/%s_grid_ifnb.csv" % (results_directory, model), ifnb_predicted, delimiter=",")
            np.savetxt("%s/%s_grid_kpars.csv" % (results_directory, model), kgrid, delimiter=",")
            del kgrid

        if not args.calc_grid and not args.no_plot:
            print("Loading grid...", flush=True)
            pars = np.loadtxt("%s/%s_grid_pars.csv" % (results_directory, model), delimiter=",")
            ifnb_predicted = np.loadtxt("%s/%s_grid_ifnb.csv" % (results_directory, model), delimiter=",")
            # kgrid = np.loadtxt("%s/%s_grid_kpars.csv" % (results_directory, model), delimiter=",")

        extra_training_data = pd.DataFrame({"Stimulus":"basal", "Genotype":"WT", "IRF":0.001, "NFkB":0.001, "p50":1}, index=[0])
        training_data_extended = pd.concat([training_data, extra_training_data], ignore_index=True)
        
        stimuli = training_data_extended["Stimulus"]
        genotypes = training_data_extended["Genotype"]

        if args.no_plot == False:
            # Calculate residuals
            print("Calculating residuals...", flush=True)
            residuals = ifnb_predicted - np.stack([beta for _ in range(len(pars))])
            rmsd = np.sqrt(np.mean(residuals**2, axis=1))
            del residuals

            # Sort rmsd
            sorted_indices = np.argsort(rmsd)
            best_20 = sorted_indices[:20]
            best_20_params = pars[best_20]
            best_20_pars_df = pd.DataFrame(best_20_params, columns=par_names)
            best_20_pars_df["rmsd"] = rmsd[best_20]
            best_20_pars_df.to_csv("%s/%s_best_20_pars.csv" % (results_directory, model), index=False)
            print("Finished calculating best 20 parameters", flush=True)
            del best_20_params

            # Get best 20 from each h value
            h_vals = np.unique(pars[:,-num_h_pars:], axis=0)
            h_best_20_inds = {}
            for row in h_vals:
                h_indices = np.where((pars[:,-num_h_pars:] == row).all(axis=1))[0]
                h_rmsd = rmsd[h_indices]
                h_sorted_indices = h_indices[np.argsort(h_rmsd)]
                h_best_20_inds[tuple(row)] = h_sorted_indices[:20]
                h_best_20 = h_sorted_indices[:20]
                h_best_20_df = pd.DataFrame(pars[h_best_20], columns=par_names)
                h_best_20_df["rmsd"] = rmsd[h_best_20]
                h_vals_str = "_".join([str(int(x)) for x in row])
                h_best_20_df.to_csv("%s/%s_best_20_pars_h_%s.csv" % (results_directory, model, h_vals_str), index=False)
            
            if args.calc_states == False:
                del pars, rmsd


        if args.calc_states:
            print("Calculating state probabilities...", flush=True)
            t = time.time()
            for row in h_vals:
                grid_partial = pars[h_best_20_inds[tuple(row)]]
                kgrid = grid_partial[:,num_t_pars:]
                for stimulus, genotype in zip(stimuli, genotypes):
                    nfkb, irf = get_N_I_P(training_data_extended, stimulus, genotype)
                    print("Calculating state probabilities for %s %s, h=%s" % (stimulus, genotype, "_".join([str(int(x)) for x in row])), flush=True)
                    print("N=%.2f, I=%.2f" % (nfkb, irf), flush=True)
                
                    with Pool(num_threads) as p:
                        results = p.starmap(calc_state_prob, [(tuple(kgrid[i]), nfkb, irf) for i in range(len(kgrid))])
                
                    state_names = results[0][1]
                    state_probabilities = np.array([x[0] for x in results])

                    np.savetxt("%s/%s_%s_%s_best_20_state_probabilities_h_%s.csv" % (results_directory, model, stimulus, genotype, "_".join([str(int(x)) for x in row])), state_probabilities, delimiter=",")
                    del state_probabilities, results

                np.savetxt("%s/%s_state_names.csv" % (results_directory, model), state_names, delimiter=",", fmt="%s")
            print("Time elapsed: %.2f minutes" % ((time.time() - t)/60), flush=True)

            del pars, rmsd


        if args.no_plot == False:
            plot_time_start = time.time()
            print("Plotting ON", flush=True)

            print("Plotting state probabilities", flush=True)
            probabilities = {}
            state_names = np.loadtxt("%s/%s_state_names.csv" % (results_directory, model), delimiter=",", dtype=str)
            for row in h_vals:
                h_vals_str = "_".join([str(int(x)) for x in row])
                probabilities[h_vals_str] = {}
                for stimulus, genotype in zip(stimuli, genotypes):
                    state_probabilities = np.loadtxt("%s/%s_%s_%s_best_20_state_probabilities_h_%s.csv" % (results_directory, model, stimulus, genotype, h_vals_str), delimiter=",")
                    condition = "%s_%s" % (stimulus, genotype)
                    probabilities[h_vals_str][condition] = state_probabilities
                    del state_probabilities

            h_strs = ["_".join([str(int(x)) for x in row]) for row in h_vals]
            with Pool(num_plot_threads) as p:
                # p.starmap(plot_state_probabilities, [(probabilities["_".join([str(int(x)) for x in row])][cond], state_names, "%s_%s_best_20_state_probabilities_h_%s" % 
                #                                             (model, cond, "_".join([str(int(x)) for x in row]), figures_directory)) for row in h_vals for cond in probabilities["_".join([str(int(x)) for x in row])].keys()])
                p.starmap(plot_state_probabilities, [(probabilities[h_str][cond], state_names, "%s_%s_best_20_state_probabilities_h_%s" %
                                                            (model, cond, h_str), figures_directory) for h_str in h_strs for cond in probabilities[h_str].keys()])
            del probabilities
            print("Done plotting state probabilities", flush=True)
            
            np.savetxt("%s/%s_h_values.csv" % (results_dir, model), h_vals, delimiter=",", fmt="%d")
            #### Best 20 parameters for each h value ####
            # for row in h_vals:
            #     print("Plotting best 20 parameters, predictions, state probabilities, and parameter distributions for h values:", flush=True)
            #     print(", ".join([str(int(x)) for x in row]), flush=True)

            #     h_vals_str = "_".join([str(int(x)) for x in row])
            #     h_best_20 = h_best_20_inds[tuple(row)]
            #     plot_parameters(pars, subset=h_best_20, name="parameters_best_20_h_%s" % h_vals_str, figures_dir=figures_directory)
            #     plot_predictions(ifnb_predicted, beta, conditions, subset=h_best_20, name="ifnb_predictions_best_20_h_%s" % h_vals_str, figures_dir=figures_directory)
            #     for cond in probabilities[h_vals_str].keys():
            #         plot_state_probabilities(probabilities[h_vals_str][cond], state_names, "%s_%s_best_20_state_probabilities_h_%s" % (model, cond, h_vals_str), figures_dir=figures_directory)
            #     plot_parameter_pairwise(pars_df=h_best_20_df, subset="All", name="parameter_pairplots_h_%s" % h_vals_str, figures_dir=figures_directory)

            t = time.time()
            print("Plotting best 20 parameters, predictions, and parameter distributions for h values:", flush=True)
            with Pool(num_plot_threads) as p:
                # p.starmap(plot_parameters, [(pars, h_best_20_inds[tuple(row)], "parameters_best_20_h_%s" % "_".join([str(int(x)) for x in row]), figures_directory) for row in h_vals])
                p.starmap(plot_parameters_wrapper, [(results_directory, model, "_".join([str(int(x)) for x in row]), figures_directory, h_best_20_inds[tuple(row)], "parameters_best_20_h_%s" % "_".join([str(int(x)) for x in row]), figures_directory) for row in h_vals])
            with Pool(num_plot_threads) as p:
                p.starmap(plot_predictions, [(ifnb_predicted, beta, conditions, h_best_20_inds[tuple(row)], "ifnb_predictions_best_20_h_%s" % "_".join([str(int(x)) for x in row]), figures_directory) for row in h_vals])
            with Pool(num_plot_threads) as p:
                p.starmap(plot_parameter_pairwise, [(h_best_20_df, "All", "parameter_pairplots_h_%s" % "_".join([str(int(x)) for x in row]), figures_directory) for row in h_vals])
            print("Time elapsed to make all plots: %.2f minutes" % ((time.time() - t)/60), flush=True)

            # Done with h values
            del h_best_20_df, h_best_20_inds

            #### All h values together ####
            tog_fig_dir = "%s/h_values_together/" % figures_directory
            os.makedirs(tog_fig_dir, exist_ok=True)
            # # Plot best 20 state probabilities
            # print("Plotting best 20 state probabilities", flush=True)
            # for cond in probabilities.keys():
            #     plot_state_probabilities(probabilities[cond][best_20_k_indices], state_names,
            #                             "%s_%s_best_20_state_probabilities" % (model, cond), figures_dir=tog_fig_dir)
            # print("Done plotting best 20 state probabilities", flush=True)

            # Plot best 20 ifnb predictions
            print("Plotting best 20 IFNb predictions", flush=True)
            plot_predictions(ifnb_predicted, beta, conditions, subset=best_20, name="ifnb_predictions_best_20", figures_dir=tog_fig_dir)
            print("Done plotting best 20 IFNb predictions", flush=True)

            # Plot best 20 parameters
            print("Plotting best 20 parameters", flush=True)
            plot_parameters(pars, subset=best_20, name="parameters_best_20", figures_dir=tog_fig_dir)

            # Plot distributions of all parameters
            print("Plotting all parameter distributions", flush=True)
            t = time.time()
            plot_parameter_distributions(pars, subset="All", name="all_parameter_distributions", figures_dir=tog_fig_dir)
            t = time.time() - t
            print("Time elapsed: %.2f minutes" % (t/60), flush=True)

            plot_time = time.time() - plot_time_start
            print("Total time elapsed for plotting: %.2f minutes" % (plot_time/60), flush=True)

            # Done with all values
            del pars, ifnb_predicted

    # Combine best 20 parameters for each h value for each seed
    print("Combining best 20 parameters for each h value for each seed", flush=True)
    t = time.time()
    h_vals = np.loadtxt("%s/%s_h_values.csv" % (results_dir, model), delimiter=",", dtype=int)

    with Pool(num_threads) as p:
        tmp = p.starmap(combine_pars, [(row, num_seeds, model, results_dir, figures_dir) for row in h_vals])

    # for row in h_vals:
    #     h_vals_str = "_".join([str(int(x)) for x in row])
    #     all_best_h_pars = pd.DataFrame()
    #     for seed in range(num_seeds):
    #         results_directory = "%s/seed_%d" % (results_dir, seed)
    #         best_20_pars_df = pd.read_csv("%s/%s_best_20_pars_h_%s.csv" % (results_directory, model, h_vals_str))
    #         all_best_h_pars = pd.concat([all_best_h_pars, best_20_pars_df], ignore_index=True)
    #     all_best_h_pars.to_csv("%s/%s_all_best_20_pars_h_%s.csv" % (results_dir, model, h_vals_str), index=False)
    #     plot_parameter_pairwise(all_best_h_pars, subset="All", name="parameter_pairplots_h_%s" % h_vals_str, figures_dir=figures_dir)
    #     print("Plotted %s after %.2f minutes, saved to %s" % (h_vals_str, (time.time() - t)/60, figures_dir), flush=True)
    print("Time elapsed for all h-values: %.2f minutes" % ((time.time() - t)/60), flush=True)
    
    # Plot rmsd, facet by h1, h2, color by h3
    if num_h_pars == 3:
        print("Plotting rmsd by h values", flush=True)
        all_rmsd = pd.DataFrame()
        for row in h_vals:
            h_vals_str = "_".join([str(int(x)) for x in row])
            all_best_h_pars = pd.read_csv("%s/%s_all_best_20_pars_h_%s.csv" % (results_dir, model, h_vals_str))
            all_rmsd = pd.concat([all_rmsd, all_best_h_pars], ignore_index=True)
        sns.displot(data=all_rmsd, x="rmsd", col="h2", row="h1", hue="h3", kind="kde", fill=True, alpha=0.5, palette="viridis")
        sns.despine()
        plt.tight_layout()
        plt.savefig("%s/%s_rmsd_by_h_values.png" % (figures_dir, model))
        plt.close()

        sns.displot(data=all_rmsd, x="rmsd", col="h3", row="h1", hue="h2", kind="kde", fill=True, alpha=0.5, palette="viridis")
        sns.despine()
        plt.tight_layout()
        plt.savefig("%s/%s_rmsd_by_h_values_2.png" % (figures_dir, model))
        plt.close()
        del all_rmsd


    # Combine best 20 parameters for each seed
    print("Combining best 20 parameters for each seed", flush=True)
    t = time.time()
    all_best_pars = pd.DataFrame()
    for seed in range(num_seeds):
        results_directory = "%s/seed_%d" % (results_dir, seed)
        best_20_pars_df = pd.read_csv("%s/%s_best_20_pars.csv" % (results_directory, model))
        all_best_pars = pd.concat([all_best_pars, best_20_pars_df], ignore_index=True)

    all_best_pars.to_csv("%s/%s_all_best_20_pars.csv" % (results_dir, model), index=False)
    
    plot_parameter_pairwise(all_best_pars, subset="All", name="parameter_pairplots_together", figures_dir=figures_dir)
    print("Time elapsed: %.2f minutes" % ((time.time() - t)/60), flush=True)


    end_end = time.time()
    t = end_end - start_start
    print("\n###############################################")
    if t < 60*60:
        print("Total time elapsed: %.2f minutes" % (t/60), flush=True)
    else:
        print("Total time elapsed: %.2f hours" % (t/3600), flush=True)

if __name__ == "__main__":
    main()