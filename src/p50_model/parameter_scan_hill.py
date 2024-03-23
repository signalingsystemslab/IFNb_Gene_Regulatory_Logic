# Perform grid search over parameter space to get all possible values of IFNb
# t parameters and k parameters are varied
# TODO: Add Hill coefficient as a parameter
from p50_model import *
import matplotlib.pyplot as plt
import pandas as pd
import os
# import scipy.optimize as opt
import time
from multiprocessing import Pool
import argparse
import seaborn as sns
import scipy.stats.qmc as qmc


figures_dir = "param_scan_hill/figures/"
results_dir = "param_scan_hill/results/"
os.makedirs(results_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)
num_t_pars = 6
num_k_pars = 4
num_h_pars = 2

# # Set seaborn context
# sns.set_context("talk")

def calculate_ifnb(pars, data):
    if len(pars) != num_t_pars + num_k_pars + num_h_pars:
        raise ValueError("Number of parameters does not match number of t, k, and h parameters")
    t_pars, k_pars, h_pars = pars[:num_t_pars], pars[num_t_pars:num_t_pars+num_k_pars], pars[num_t_pars+num_k_pars:]
    N, I, P = data["NFkB"], data["IRF"], data["p50"]
    ifnb = [get_f(t_pars, k_pars, n, i, p, h_pars=h_pars) for n, i, p in zip(N, I, P)]
    ifnb = np.array(ifnb)
    return ifnb

def calculate_grid(training_data, num_samples= 10**6, seed=0, num_threads=60):
    min_k_order = -2
    max_k_order = 3

    seed += 10

    l_bounds = np.concatenate([np.zeros(num_t_pars), np.ones(num_k_pars)*min_k_order])
    u_bounds = np.concatenate([np.ones(num_t_pars), np.ones(num_k_pars)*max_k_order])

    print("Calculating grid with %d samples using Latin Hypercube sampling" % num_samples, flush=True)
    sampler=qmc.LatinHypercube(d=num_t_pars+num_k_pars, seed=seed)
    grid = sampler.random(n=num_samples)
    grid = qmc.scale(grid, l_bounds, u_bounds) # rows are parameter sets
    # convert k parameters to log space
    kgrid = grid[:,num_t_pars:]
    kgrid = 10**kgrid
    grid[:,num_t_pars:] = kgrid

    # Add h values to grid. 
    hgrid = np.ones((num_samples, num_h_pars)) + 1
    grid = np.array(np.concatenate([grid, hgrid], axis=1))
    kgrid = grid[:,num_t_pars:]
    print("Total number of samples after adding h parameters: %d" % len(grid), flush=True)

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
    P = row["p50"].values[0]
    return N, I, P

def calc_state_prob(khpars, N, I, P):
    # print(N, I, P, flush=True)
    k_pars = khpars[:num_k_pars]
    h_pars = khpars[num_k_pars:]
    t_pars = [1 for _ in range(num_t_pars)]
    probabilities, state_names = get_state_prob(t_pars, k_pars, N, I, P, h_pars=h_pars)
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
        # df_ifnb_predicted["Stimulus"] = df_ifnb_predicted["Data point"].str.split("_", expand=True)[0]
        # # sort by stimulus
        # stimuli = conditions.str.split("_", expand=True)[0].unique()
        # df_ifnb_predicted["Stimulus"] = pd.Categorical(df_ifnb_predicted["Stimulus"], stimuli)
        # df_ifnb_predicted = df_ifnb_predicted.sort_values("Stimulus")

        fig, ax = plt.subplots()
        sns.lineplot(data=df_ifnb_predicted.loc[df_ifnb_predicted["par_set"].isin(subset)], x="Data point", y=r"IFN$\beta$", 
                     units="par_set", color="black", alpha=0.5, estimator=None, ax=ax)
        sns.scatterplot(data=df_ifnb_predicted.loc[df_ifnb_predicted["par_set"] == "Data"], x="Data point", y=r"IFN$\beta$", 
                        color="red", marker="o", ax=ax, legend=False)
        sns.despine()
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig("%s/%s.png" % (figures_dir, name))
        plt.close()

        # Plot predictions on log scale
        fig, ax = plt.subplots()
        sns.lineplot(data=df_ifnb_predicted.loc[df_ifnb_predicted["par_set"].isin(subset)], x="Data point", y=r"IFN$\beta$",
                        units="par_set", color="black", alpha=0.5, estimator=None, ax=ax)
        sns.scatterplot(data=df_ifnb_predicted.loc[df_ifnb_predicted["par_set"] == "Data"], x="Data point", y=r"IFN$\beta$",
                        color="red", marker="o", ax=ax, legend=False)
        sns.despine()
        plt.xticks(rotation=90)
        plt.yscale("log")
        ax.set_ylim(bottom=0.01)
        plt.tight_layout()
        plt.savefig("%s/%s_log.png" % (figures_dir, name))
        plt.close()

def plot_parameters(pars, subset="All", name="parameters", figures_dir=figures_dir, param_names=None):
    if type(subset) == str:
        if subset == "All":
            subset = np.arange(len(pars))

    if param_names is None:
        par_names = ["t%d" % (i+1) for i in range(num_t_pars)] + ["k%d" % (i+1) for i in range(num_k_pars)] + ["h1", "h2"]
    else:
        par_names = param_names

    df_pars = pd.DataFrame(pars[subset,:], columns=par_names)
    df_pars["par_set"] = np.arange(len(df_pars))
    # df_pars["kp_ratio"] = df_pars["k4"] / df_pars["k2"]
    df_pars = df_pars.melt(var_name="Parameter", value_name="Value", id_vars="par_set")
    df_pars["Parameter"] = df_pars["Parameter"].str.replace("k3", "kn")
    df_pars["Parameter"] = df_pars["Parameter"].str.replace("k4", "kp")

    df_t_pars = df_pars[df_pars["Parameter"].str.startswith("t")]
    df_k_pars = df_pars[df_pars["Parameter"].str.startswith("k")]
    # df_kp_par = df_pars[df_pars["Parameter"].str.startswith("kp_")]
    # # remove kp and kp_ratio from k parameters
    # df_k_pars = df_k_pars[~df_k_pars["Parameter"].str.startswith("kp")]
    if "c" in df_pars["Parameter"].values:
        df_c_pars = df_pars[df_pars["Parameter"].str.startswith("c")]
        fig, ax = plt.subplots(1,3, figsize=(11,5), gridspec_kw={"width_ratios":[3,2,1]})
        sns.lineplot(data=df_t_pars, x="Parameter", y="Value", units="par_set", color="black", alpha=0.5, estimator=None, ax=ax[0])
        sns.lineplot(data=df_k_pars, x="Parameter", y="Value", units="par_set", color="black", alpha=0.5, estimator=None, ax=ax[1])
        sns.lineplot(data=df_c_pars, x="Parameter", y="Value", units="par_set", color="black", alpha=0.5, estimator=None, ax=ax[2])
        ax[1].set_yscale("log")
        ax[2].set_yscale("log")
        sns.despine()
        plt.tight_layout()
    elif "h1" in df_pars["Parameter"].values:
        df_h_pars = df_pars[df_pars["Parameter"].str.startswith("h")]
        fig, ax = plt.subplots(1,3, figsize=(12,5), gridspec_kw={"width_ratios":[3,2,2]})
        sns.lineplot(data=df_t_pars, x="Parameter", y="Value", units="par_set", color="black", alpha=0.5, estimator=None, ax=ax[0])
        sns.lineplot(data=df_k_pars, x="Parameter", y="Value", units="par_set", color="black", alpha=0.5, estimator=None, ax=ax[1])
        sns.lineplot(data=df_h_pars, x="Parameter", y="Value", units="par_set", color="black", alpha=0.5, estimator=None, ax=ax[2])
        ax[1].set_yscale("log")
        sns.despine()
        plt.tight_layout()
    else:
        fig, ax = plt.subplots(1,2, figsize=(10,5), gridspec_kw={"width_ratios":[3,2]})
        sns.lineplot(data=df_t_pars, x="Parameter", y="Value", units="par_set", color="black", alpha=0.5, estimator=None, ax=ax[0])
        sns.lineplot(data=df_k_pars, x="Parameter", y="Value", units="par_set", color="black", alpha=0.5, estimator=None, ax=ax[1])
        ax[1].set_yscale("log")
        # sns.lineplot(data=df_kp_par, x="Parameter", y="Value", units="par_set", color="black", alpha=0.5, estimator=None, ax=ax[2])
        sns.despine()
        plt.tight_layout()

    plt.savefig("%s/%s.png" % (figures_dir, name))
    plt.close()

def plot_parameter_distributions(pars, subset ="All", name="parameter_distributions", figures_dir=figures_dir, param_names=None):
    if param_names is None:
        par_names = ["t%d" % (i+1) for i in range(num_t_pars)] + ["k%d" % (i+1) for i in range(num_k_pars)] + ["h1", "h2"]
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

# def loss_function(ifnb_predicted, ifnb, params, p=10, function="rmsd"):
#     residuals = ifnb_predicted - ifnb
#     rmsd = np.sqrt(np.mean(residuals**2))
#     if function == "rmsd" or function == "l2":
#         return rmsd
#     elif function == "fun1":
#         if np.any(params < 0) or np.any(params[:6] > 1):
#             return 1e6
#         hard_cons1 = np.min(np.append(params, 0))**2
#         hard_cons2 = np.min(1 - np.append(params, 1))**2

#         cons1 = ifnb_predicted[0] - ifnb_predicted[4] - 0.05
#         cons2 = ifnb_predicted[0] - ifnb_predicted[4] - (ifnb_predicted[6] - ifnb_predicted[7]) - 0.05
#         cons3 = ifnb_predicted[3] - ifnb_predicted[2] - 0.05
#         cons4 = ifnb_predicted[3] - ifnb_predicted[2] - (ifnb_predicted[1] - ifnb_predicted[0]) - 0.05
#         # penalty for not meeting constraints
#         penalty = -cons1 - cons2 - cons3 - cons4
#         penalty = np.minimum(penalty, 0)**2
#         return rmsd + penalty*p + 10**6*hard_cons1 + 10**6*hard_cons2, penalty
#     elif function == "fun2":
#         if np.any(params < 0) or np.any(params[:6] > 1):
#            return 1e6
#         hard_cons1 = np.min(np.append(params, 0))**2
#         hard_cons2 = np.min(1 - np.append(params, 1))**2

#         maximize1 = ifnb_predicted[0] - ifnb_predicted[4]
#         maximize2 = ifnb_predicted[0] - ifnb_predicted[4] - (ifnb_predicted[6] - ifnb_predicted[7])
#         maximize3 = ifnb_predicted[3] - ifnb_predicted[2]
#         maximize4 = ifnb_predicted[3] - ifnb_predicted[2] - (ifnb_predicted[1] - ifnb_predicted[0])
#         # penalty for not meeting constraints
#         rmsd_adjusted = rmsd + 10**6*hard_cons1 + 10**6*hard_cons2 - maximize1 - maximize2 - maximize3 - maximize4
#         return rmsd_adjusted, (maximize1, maximize2, maximize3, maximize4)
#     elif function == "fun3":
#         comparisons = [(0,1),(0,2),(3,2),(0,4),(0,5),(0,6),(6,7),(6,8),(6,9)]
#         # calculate value of pt 1 - pt 2 for each comparison for ifnb_predicted and ifnb
#         predicted_comps = [ifnb_predicted[i] - ifnb_predicted[j] for i,j in comparisons]
#         actual_comps = [ifnb[i] - ifnb[j] for i,j in comparisons]
#         # calculate residuals
#         residuals = np.array(predicted_comps) - np.array(actual_comps)
#         # calculate mean squared error
#         mse = np.mean(residuals**2)
#         return mse, residuals
    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g","--calc_grid", action="store_true")
    parser.add_argument("-s","--calc_states", action="store_true")
    parser.add_argument("-n","--no_plot", action="store_false") 
    args = parser.parse_args()

    start_start = time.time()

    print("###############################################\n")
    print("Starting at %s\n" % time.ctime(), flush=True)

    training_data = pd.read_csv("../data/p50_training_data.csv")
    print("Using the following training data:\n", training_data)
    model = "p50_hill"

    N = training_data["NFkB"]
    I = training_data["IRF"]
    P = training_data["p50"]
    beta = training_data["IFNb"]
    conditions = training_data["Stimulus"] + "_" + training_data["Genotype"]
    num_pts = len(N)
    len_training = len(N)
    par_names = ["t%d" % (i+1) for i in range(num_t_pars)] + ["k%d" % (i+1) for i in range(num_k_pars)] + ["h1", "h2"]

    num_pars = num_t_pars + num_k_pars
    num_threads = 40
    num_par_sets = 10**6

    num_best_pars = []

    num_seeds = 3
    for seed in range(num_seeds):
        print("Seed: %d" % seed, flush=True)
        results_dir = "param_scan_hill/results/seed_%d" % seed
        figures_dir = "param_scan_hill/figures/seed_%d" % seed
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(figures_dir, exist_ok=True)

        if args.calc_grid:
            print("Calculating grid ON", flush=True)

            ifnb_predicted, pars, kgrid = calculate_grid(training_data, num_samples=num_par_sets, seed=seed, num_threads=num_threads)

            np.savetxt("%s/%s_grid_pars.csv" % (results_dir, model), pars, delimiter=",")
            np.savetxt("%s/%s_grid_ifnb.csv" % (results_dir, model), ifnb_predicted, delimiter=",")
            np.savetxt("%s/%s_grid_kpars.csv" % (results_dir, model), kgrid, delimiter=",")

        if not args.calc_grid:
            pars = np.loadtxt("%s/%s_grid_pars.csv" % (results_dir, model), delimiter=",")
            ifnb_predicted = np.loadtxt("%s/%s_grid_ifnb.csv" % (results_dir, model), delimiter=",")
            kgrid = np.loadtxt("%s/%s_grid_kpars.csv" % (results_dir, model), delimiter=",")

        # Calculate all states
        extra_training_data = pd.DataFrame({"Stimulus":"basal", "Genotype":"WT", "IRF":0.001, "NFkB":0.001, "p50":1}, index=[0])
        training_data_extended = pd.concat([training_data, extra_training_data], ignore_index=True)
        
        stimuli = training_data_extended["Stimulus"]
        genotypes = training_data_extended["Genotype"]

        if args.calc_states:
            print("Calculating state probabilities ON", flush=True)
            for stimulus, genotype in zip(stimuli, genotypes):
                nfkb, irf, p50 = get_N_I_P(training_data_extended, stimulus, genotype)
                print("Calculating state probabilities for %s %s" % (stimulus, genotype), flush=True)
                print("N=%.2f, I=%.2f, P=%.2f" % (nfkb, irf, p50), flush=True)
            
                with Pool(num_threads) as p:
                    results = p.starmap(calc_state_prob, [(tuple(kgrid[i]), nfkb, irf, p50) for i in range(len(kgrid))])
            
                state_names = results[0][1]
                state_probabilities = np.array([x[0] for x in results])

                np.savetxt("%s/%s_%s_%s_state_probabilities.csv" % (results_dir, model, stimulus, genotype), state_probabilities, delimiter=",")

            np.savetxt("%s/%s_state_names.csv" % (results_dir, model), state_names, delimiter=",", fmt="%s")

        if args.no_plot:
            print("Plotting ON", flush=True)

            # print("Plotting state probabilities", flush=True)
            probabilities = {}
            state_names = np.loadtxt("%s/%s_state_names.csv" % (results_dir, model), delimiter=",", dtype=str)
            for stimulus, genotype in zip(stimuli, genotypes):
                state_probabilities = np.loadtxt("%s/%s_%s_%s_state_probabilities.csv" % (results_dir, model, stimulus, genotype), delimiter=",")
                condition = "%s_%s" % (stimulus, genotype)
                probabilities[condition] = state_probabilities

                # plot_state_probabilities(state_probabilities, state_names, "%s_%s_%s_all_state_probabilities" % (model, stimulus, genotype), figures_dir=figures_dir)


            # # Plot all ifnb predictions - slow
            # print("Plotting all IFNb predictions", flush=True)
            # plot_predictions(ifnb_predicted, beta, conditions, subset="All", name="ifnb_predictions", figures_dir=figures_dir)


            # Calculate residuals
            residuals = ifnb_predicted - np.stack([beta for _ in range(len(pars))])
            rmsd = np.sqrt(np.mean(residuals**2, axis=1))

            # Sort rmsd
            sorted_indices = np.argsort(rmsd)
            best_20 = sorted_indices[:20]
            best_20_params = pars[best_20]
            best_20_kpars = best_20_params[:,num_t_pars:]
            best_20_k_indices = np.array([np.where(np.all(kgrid == kp, axis=1))[0][0] for kp in best_20_kpars])
            best_20_pars_df = pd.DataFrame(best_20_params, columns=par_names)
            best_20_pars_df.to_csv("%s/%s_best_20_pars.csv" % (results_dir, model), index=False)

            # Plot best 20 state probabilities
            print("Plotting best 20 state probabilities", flush=True)
            for cond in probabilities.keys():
                plot_state_probabilities(probabilities[cond][best_20_k_indices], state_names,
                                        "%s_%s_best_20_state_probabilities" % (model, cond), figures_dir=figures_dir)
            print("Done plotting best 20 state probabilities", flush=True)

            # Plot best 20 ifnb predictions
            print("Plotting best 20 IFNb predictions", flush=True)
            plot_predictions(ifnb_predicted, beta, conditions, subset=best_20, name="ifnb_predictions_best_20", figures_dir=figures_dir)
            print("Done plotting best 20 IFNb predictions", flush=True)

            # Plot best 20 parameters
            print("Plotting best 20 parameters", flush=True)
            plot_parameters(pars, subset=best_20, name="parameters_best_20", figures_dir=figures_dir)

            # Plot distributions of all parameters
            print("Plotting all parameter distributions", flush=True)
            t = time.time()
            plot_parameter_distributions(pars, subset="All", name="all_parameter_distributions", figures_dir=figures_dir)
            t = time.time() - t
            print("Time elapsed: %.2f minutes" % (t/60), flush=True)

            # print("Plotting parameter grid", flush=True)
            # start = time.time()
            
            # df_pars = pd.DataFrame(pars, columns=par_names)
            # df_pars["is_top_20"] = [i in best_20 for i in range(len(pars))]
            # not_top_20 = np.array([i for i in range(len(pars)) if i not in best_20])
            # np.random.seed(0)
            # random_100 = np.random.choice(not_top_20, size=100, replace=False)
            # df_pars["is_random_100"] = [i in random_100 for i in range(len(pars))]
            # df_pars_subset = df_pars[df_pars["is_top_20"] | df_pars["is_random_100"]]

            # fig, ax = plt.subplots()
            # sns.pairplot(df_pars_subset, diag_kind="kde", plot_kws={"alpha":0.5}, diag_kws={"alpha":0.5}, hue="is_top_20", palette="viridis")
            # plt.savefig("%s/%s_parameter_grid_sample.png" % (figures_dir, model))
            # plt.close()


            # fig, ax = plt.subplots()
            # sns.pairplot(df_pars, diag_kind="kde", plot_kws={"alpha":0.5}, diag_kws={"alpha":0.5}, hue="is_top_20", palette="viridis")
            # plt.savefig("%s/%s_parameter_grid.png" % (figures_dir, model))
            # plt.close()
            # end = time.time()
            # t = end - start
            # print("Time elapsed: %.2f minutes" % (t/60), flush=True)

            ## Constraints ##
            # Filter for rows with the following constraints
            print("Filtering for constraints", flush=True)
            # LPS vs pIC nfkb KO
            rows_to_keep = ifnb_predicted[:,0] - ifnb_predicted[:,4] > 0.05
            rows_to_keep = rows_to_keep & ((ifnb_predicted[:,0] - ifnb_predicted[:,4]) - (ifnb_predicted[:,6] - ifnb_predicted[:,7]) > 0.05)
            # LPS vs CpG p50 KO 
            rows_to_keep = rows_to_keep & (ifnb_predicted[:,3] - ifnb_predicted[:,2] > 0.05)
            rows_to_keep = rows_to_keep & ((ifnb_predicted[:,3] - ifnb_predicted[:,2]) - (ifnb_predicted[:,1] - ifnb_predicted[:,0]) > 0.05)

            residuals_filtered = residuals[rows_to_keep,:]
            pars_filtered = pars[rows_to_keep,:]
            ifnb_filtered = ifnb_predicted[rows_to_keep,:]

            print("Number of rows after filtering: %d (%.2f%%)" % (len(residuals_filtered), 100*len(residuals_filtered)/len(residuals)), flush=True)
            if len(residuals_filtered) < 20:
                raise ValueError("Number of rows after filtering is less than 20")

            rmsd = np.sqrt(np.mean(residuals_filtered**2, axis=1))  

            # Plot distribution of RMSD
            print("Plotting distribution of RMSD", flush=True)
            fig, ax = plt.subplots()
            sns.histplot(rmsd, kde=True, color="black", alpha=0.5, bins=20)
            sns.despine()
            plt.savefig("%s/%s_rmsd_filtered.png" % (figures_dir, model))
            plt.close()

            qntl = np.round(20/len(rmsd), 3)
            print("RMSD stats:\n", pd.Series(rmsd).describe([qntl,0.05, 0.25, 0.5, 0.75, 0.95]), flush=True)

            # Plot a scatter plot of smallest 25% of rmsd values
            print("Plotting scatter plot of smallest 5k rmsd values", flush=True)
            df = pd.DataFrame({"rmsd":rmsd, "filtered_par_set":np.arange(len(rmsd))})
            df = df.sort_values("rmsd")
            df["sorted_par_set"] = np.arange(len(rmsd))
            df = df.iloc[:5000,:]

            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x="sorted_par_set", y="rmsd", color="black", alpha=0.8, linewidth=0)
            sns.despine()
            plt.savefig("%s/%s_rmsd_filtered_scatter.png" % (figures_dir, model))
            plt.close()

            # Threshold for rmsd is nth percentile of rmsd
            n=1
            threshold = np.percentile(rmsd, n)
            best_20= np.where(rmsd < threshold)[0]
            num_kept = len(best_20)
            num_best_pars.append(num_kept)
            print("Keeping %d rows with rmsd < %.3f (bottom %s%% of total rmsd)" % (num_kept, threshold, n), flush=True)

            if num_kept < 5:
                raise ValueError("Number of rows after RMSD cutoff is less than 5")

            best_20_params = pars_filtered[best_20]
            best_20_kpars = best_20_params[:,num_t_pars:]
            best_20_k_indices = np.array([np.where(np.all(kgrid == kp, axis=1))[0][0] for kp in best_20_kpars])
            best_20_predicted = ifnb_filtered[best_20]

            

            # best 1000
            df = df.iloc[:1000,:]

            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x="sorted_par_set", y="rmsd", color="black", alpha=0.8, linewidth=0)
            sns.despine()
            plt.savefig("%s/%s_rmsd_filtered_scatter_1000.png" % (figures_dir, model))
            plt.close()

            # Best 50
            df = df.iloc[:50,:]

            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x="sorted_par_set", y="rmsd", color="black", alpha=0.8, linewidth=0)
            sns.despine()
            plt.savefig("%s/%s_rmsd_filtered_scatter_50.png" % (figures_dir, model))
            plt.close()

            # Save best 20 results
            print("Saving best 20 results", flush=True)
            np.savetxt("%s/%s_best_20_parameters_filtered.csv" % (results_dir, model), best_20_params, delimiter=",")
            np.savetxt("%s/%s_best_20_ifnb_filtered.csv" % (results_dir, model), best_20_predicted, delimiter=",")

            best_20_params_rd = np.round(best_20_params, 3)
            df = pd.DataFrame(best_20_params_rd, columns=par_names)
            df["rmsd"] = rmsd[best_20]
            df.to_csv("%s/%s_best_20_parameters_filtered_df.csv" % (results_dir, model), index=False)

            best_20_ifnb_rd = np.round(best_20_predicted, 3)
            df = pd.DataFrame(best_20_ifnb_rd, columns=conditions)
            df.to_csv("%s/%s_best_20_ifnb_filtered_df.csv" % (results_dir, model), index=False)

            # Plot best 20 state probabilities
            print("Plotting best 20 state probabilities", flush=True)
            for cond in probabilities.keys():
                plot_state_probabilities(probabilities[cond][best_20_k_indices], state_names,
                                        "%s_%s_best_20_state_probabilities_filtered" % (model, cond), figures_dir=figures_dir)
            print("Done plotting best 20 state probabilities", flush=True)

            # Plot best 20 ifnb predictions
            print("Plotting best 20 IFNb predictions", flush=True)
            plot_predictions(ifnb_filtered, beta, conditions, subset=best_20, name="ifnb_predictions_best_20_filtered", figures_dir=figures_dir)
            print("Done plotting best 20 IFNb predictions", flush=True)

            # Plot best 20 parameters
            print("Plotting best 20 parameters", flush=True)
            plot_parameters(pars_filtered, subset=best_20, name="parameters_best_20_filtered", figures_dir=figures_dir)
           

    end_end = time.time()
    t = end_end - start_start
    print("\n###############################################")
    if t < 60*60:
        print("Total time elapsed: %.2f minutes" % (t/60), flush=True)
    else:
        print("Total time elapsed: %.2f hours" % (t/3600), flush=True)

if __name__ == "__main__":
    main()