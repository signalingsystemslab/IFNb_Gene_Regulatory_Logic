# For each synthetic dataset, fit best parameters and plot parameters
# Optimize the parameters of the model: t parameters and k parameters
# Initial parameters from parameter scan

from p50_model_force_t import *
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import os
import scipy.optimize as opt
import time
from multiprocessing import Pool
import argparse
import seaborn as sns
import scipy.stats.qmc as qmc

mpl.rcParams["figure.dpi"] = 600
mpl.rcParams["font.sans-serif"] = "Arial"
# irf_color = "#5D9FB5"
# nfkb_color = "#BA4961"
data_color = "#6F5987"

states_cmap_pars = "ch:s=2.2,r=0.75,h=0.6,l=0.8,d=0.25"
models_cmap_pars = "ch:s=-0.0,r=0.6,h=1,d=0.3,l=0.8,g=1_r"

heatmap_cmap = sns.cubehelix_palette(as_cmap=True, light=0.95, dark=0, reverse=True, rot=0.4,start=-.2, hue=0.6)
dataset_cmap = sns.cubehelix_palette(as_cmap=True, start=0.9, rot=-.75, dark=0.1, light=0.8, hue=0.6)

plot_rc_pars = {"axes.labelsize":7, "font.size":6, "legend.fontsize":6, "xtick.labelsize":6, 
                                          "ytick.labelsize":6, "axes.titlesize":7, "legend.title_fontsize":7,
                                          "lines.markersize": 3, "axes.linewidth": 0.5,
                                            "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.minor.width": 0.5,
                                            "ytick.minor.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2,
                                            "xtick.minor.size": 1, "ytick.minor.size": 1, "legend.labelspacing": 0.2,
                                            "legend.columnspacing": 0.5, "legend.handletextpad": 0.5, "legend.handlelength": 1.5}
rc_pars={"xtick.major.pad": 1, "ytick.major.pad": 1, "legend.labelspacing": 0.2}
mpl.rcParams.update(rc_pars)

def get_N_I_P(data, stimulus, genotype):
    row = data.loc[(data["Stimulus"] == stimulus) & (data["Genotype"] == genotype)]
    N = row["NFkB"].values[0]
    I = row["IRF"].values[0]
    P = row["p50"].values[0]
    return N, I, P

def calc_state_prob(k_pars, N, I, P, num_t=6, h_pars=None):
    # print(N, I, P, flush=True)
    t_pars = [1 for _ in range(num_t)]
    probabilities, state_names = get_state_prob(t_pars, k_pars, N, I, P, h_pars=h_pars)
    return probabilities, state_names

# def plot_state_probabilities(state_probabilities, state_names, name, figures_dir):
#         stimuli = ["basal", "CpG", "LPS", "polyIC"]
#         stimulus = [s for s in stimuli if s in name]
#         if len(stimulus) == 0:
#             stimulus = "No Stim"
#         elif len(stimulus) > 1:
#             raise ValueError("More than one stimulus in name")
#         else:
#             stimulus = stimulus[0]

#         condition = name.split("_")[-2:]
#         condition = " ".join(condition)
#         df_state_probabilities = pd.DataFrame(state_probabilities, columns=state_names)
#         df_state_probabilities["par_set"] = np.arange(len(df_state_probabilities))
#         df_state_probabilities = df_state_probabilities.melt(var_name="State", value_name="Probability", id_vars="par_set")

#         with sns.plotting_context("talk", rc={"lines.markersize": 7}):
#             fig, ax = plt.subplots(figsize=(6,5))
#             p = sns.lineplot(data=df_state_probabilities, x = "State", y="Probability", color="black", alpha=0.2,
#                                 estimator=None, units="par_set", legend=False).set_title(condition)
#             sns.scatterplot(data=df_state_probabilities, x = "State", y="Probability", color="black", alpha=0.2, ax=ax, legend=False, zorder=10)
#             sns.despine()
#             plt.xticks(rotation=90)
#             # Save plot
#             plt.savefig("%s/%s.png" % (figures_dir, name), bbox_inches="tight")

def calculate_rmsd(ifnb_predicted, beta):
    residuals = ifnb_predicted - beta
    rmsd = np.sqrt(np.mean(residuals**2))
    return rmsd

def calculate_grid(t_bounds=(0,1), k_bounds=(10**-3,10**3), seed=0, num_samples=10**6, num_threads=60, num_t_pars=5, num_k_pars=4, num_h_pars=2, c_par=False):
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

    return grid_tk


def parameter_scan2(training_data, grid, results_dir, h1=3, h2=1, num_threads=40, num_t_pars=5, num_k_pars=4, c_par=False, num_to_keep=100):
    # Load data points
    datasets = training_data["Dataset"].unique()
    num_datasets = len(datasets)
    num_data_points = int(len(training_data)/num_datasets)

    if len(training_data) % num_datasets != 0:
        raise ValueError("Number of data points (%d) is not divisible by number of datasets (%d)" % (len(training_data), num_datasets))

    # print("\nStarting parameter scan for %d datasets with %d data points each" % (num_datasets, num_data_points), flush=True)

    # all_best_fits_pars = np.zeros((num_datasets,num_to_keep, num_t_pars+num_k_pars))
    # all_best_fits_results = np.zeros((num_datasets,num_to_keep, len(training_data)))
    # all_best_fits_rmsd = np.zeros((num_datasets,num_to_keep))

    # print("Sizes of results arrays:\nBest fits pars: %s\nBest fits results: %s\nBest fits rmsd: %s" % (all_best_fits_pars.shape, all_best_fits_results.shape, all_best_fits_rmsd.shape), flush=True)

    # Only use up to 25 datasets at a time
    num_chunks = -(-num_datasets // 25)  

    chunks = np.array_split(datasets, num_chunks)
    for c, chunk in enumerate(chunks):
        data_chunk = training_data.loc[training_data["Dataset"].isin(chunk)]
        N, I, P = data_chunk["NFkB"].values, data_chunk["IRF"].values, data_chunk["p50"].values
        beta = data_chunk["IFNb"].values
        dataset_vector = data_chunk["Dataset"].values
        num_pts_times_num_datasets = len(N)
        num_param_sets = len(grid)
        num_datasets_chunk = len(chunk)

        # Calculate IFNb value at each point in grid
        print("Calculating effects at %.1E parameter sets for %d points for %d datasets (%.1E total)" % (num_param_sets, num_data_points, num_datasets_chunk, num_pts_times_num_datasets*num_param_sets), flush=True)

        # raise ValueError("Stop here")

        start = time.time()
        if c_par:
            with Pool(num_threads) as p:
                results = p.starmap(get_f, [(grid[i,0:num_t_pars], grid[i,num_t_pars:-1], N[j], I[j], P[j], grid[i,-1], [h1, h2], False) for i in range(num_param_sets) for j in range(num_pts_times_num_datasets)])
        else:
            with Pool(num_threads) as p:
                results = p.starmap(get_f, [(grid[i,0:num_t_pars], grid[i,num_t_pars:], N[j], I[j], P[j], None, [h1, h2], False) for i in range(num_param_sets) for j in range(num_pts_times_num_datasets)])


        ifnb_predicted = np.array(results).reshape(num_param_sets, num_pts_times_num_datasets)
        del results

        # Make sure all elements for each dataset are the same
        for i in range(num_datasets_chunk):
            if len(np.unique(dataset_vector[i*num_data_points:(i+1)*num_data_points])) > 1:
                raise ValueError("Dataset chunk %d has more than one dataset" % i)
        
        beta_bydset = np.array_split(beta, num_datasets_chunk)

        # Make sure length of each element of beta_bydset is the same as num_data_points
        for i in range(num_datasets_chunk):
            if len(beta_bydset[i]) != num_data_points:
                raise ValueError("Length of beta_bydset element %d (%d) is not equal to num_data_points (%d)" % (i, len(beta_bydset[i]), num_data_points))

        ifnb_predicted_bydset = np.array_split(ifnb_predicted, num_datasets_chunk, axis=1)
        del ifnb_predicted

        # Calculate residuals
        with Pool(num_threads) as p:
            rmsd = p.starmap(calculate_rmsd, [(ifnb_predicted_bydset[j][i], beta_bydset[j]) for i in range(num_param_sets) for j in range(num_datasets_chunk)])
        del beta_bydset

        rmsd = np.array(rmsd).reshape(num_param_sets, num_datasets_chunk)

        best_rmsd_rows = np.argsort(rmsd, axis=0)[:num_to_keep, :]

        if num_datasets_chunk != len(chunk):
            raise ValueError("Number of datasets in chunk (%d) is not equal to number of unique datasets in chunk (%d)" % (num_datasets_chunk, len(dataset_vector.unique())))

        for dset in range(num_datasets_chunk):
            dname = chunk[dset]
            grid_subset = grid[best_rmsd_rows[:, dset]]
            np.savetxt("%s/initial_pars_%s.csv" % (results_dir, dname), grid_subset, delimiter=",")

            # all_best_fits_pars[c*num_datasets_chunk+dset] = grid_subset
            del grid_subset
            
            rmsd_subset = rmsd[best_rmsd_rows[:, dset], dset]
            np.savetxt("%s/rmsd_%s.csv" % (results_dir, dname), rmsd_subset, delimiter=",")
            # all_best_fits_rmsd[c*num_datasets_chunk+dset] = rmsd_subset
            del rmsd_subset

            ifnb_predicted_subset = ifnb_predicted_bydset[dset][best_rmsd_rows[:, dset]]
            np.savetxt("%s/ifnb_predicted_%s.csv" % (results_dir, dname), ifnb_predicted_subset, delimiter=",")
            # all_best_fits_results[c*num_datasets_chunk+dset] = ifnb_predicted_subset
            del ifnb_predicted_subset

            print("Saved parameter scan results for dataset %s" % dname, flush=True)

        del ifnb_predicted_bydset, rmsd
    
    end = time.time()
    t = end - start
    if t < 60*60:
        print("Time elapsed for param scan: %.2f minutes" % (t/60), flush=True)
    else:
        print("Time elapsed for param scan: %.2f hours" % (t/3600), flush=True)

    # return best_fits_pars, best_fits_results, best_fits_rmsd, dataset_names
    return  datasets

def objective_function(pars, *args):
    N, I, P, beta, h_pars, c_par = args
    t_pars = pars[0:4]
    k_pars = pars[4:8]
    if c_par:
        c = pars[8]
    else:
        c = None

    num_pts = len(N)
    
    ifnb_predicted = [get_f(t_pars, k_pars, N[i], I[i], P[i], h_pars=h_pars, c_par=c) for i in range(num_pts)]

    rmsd = calculate_rmsd(ifnb_predicted, beta)

    return rmsd

def minimize_objective(pars, N, I, P, beta, h_pars, c_par, bounds):
    return opt.minimize(objective_function, pars, args=(N, I, P, beta, h_pars, c_par), method="Nelder-Mead", bounds=bounds)

def optimize_model(N, I, P, beta, initial_pars, h, c=False, num_threads=40, num_t_pars=5, num_k_pars=3):
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
        results = p.starmap(minimize_objective, [(pars, N, I, P, beta, h, c, bnds) for pars in initial_pars])

    final_pars = np.array([result.x for result in results]) # each row is a set of optimized parameters
    rmsd = np.array([result.fun for result in results])

    if c:
        with Pool(num_threads) as p:
            ifnb_predicted = p.starmap(get_f, [(final_pars[i,0:num_t_pars], final_pars[i,num_t_pars:-1], N[j], I[j], P[j], final_pars[i,-1], h, False) for i in range(len(final_pars)) for j in range(len(N))])
    else:
        with Pool(num_threads) as p:
            ifnb_predicted = p.starmap(get_f, [(final_pars[i,0:num_t_pars], final_pars[i,num_t_pars:], N[j], I[j], P[j], None, h, False) for i in range(len(final_pars)) for j in range(len(N))])

    ifnb_predicted = np.array(ifnb_predicted).reshape(len(final_pars), len(N))

    end = time.time()

    print("Finished optimization at %s" % time.ctime(), flush=True)
    t = end - start
    if t < 60:
        print("Time elapsed for optimization: %.2f seconds" % t)
    elif t < 3600:
        print("Time elapsed for optimization: %.2f minutes" % (t/60))
    else:
        print("Time elapsed for optimization: %.2f hours" % (t/3600))
    return final_pars, ifnb_predicted, rmsd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o","--optimize", action="store_true")
    # parser.add_argument("-k","--optimize_kp", action="store_true")
    parser.add_argument("-p","--param_scan", action="store_true")
    # parser.add_argument("-s","--state_probabilities", action="store_true")
    parser.add_argument("-m","--make_plots", action="store_true")
    parser.add_argument("-c","--c_parameter", action="store_true")
    parser.add_argument("-1","--h1", type=int, default=3)
    parser.add_argument("-2","--h2", type=int, default=1)
    parser.add_argument("-e","--error_val", type=float, default=0.1)
    parser.add_argument("-t","--test", action="store_true")
    args = parser.parse_args()

    start_start = time.time()

    print("###############################################\n")
    print("Starting at %s\n" % time.ctime(), flush=True)

    # Settings    
    num_threads = 60
    model = "p50_force_t"
    par_type = "k"
    h1, h2 = args.h1, args.h2
    h3 = 1
    h_val = "%d_%d_%d" % (h1, h2, h3)
    print("h_I1: %d, h_I2: %d, h3_N: %d" % (h1, h2, h3), flush=True)
    c_par= args.c_parameter
    num_to_keep = 100

    # Model details
    num_t_pars = 4
    num_k_pars = 4
    # num_c_pars = 1
    # num_h_pars = 2

    # Directories
    figures_dir = "parameter_scan_force_t/figures/"
    results_dir = "parameter_scan_force_t/results/"
    if h_val != "3_1_1":
        figures_dir = figures_dir[:-1] + "_h_%s/" % h_val
        results_dir = results_dir[:-1] + "_h_%s/" % h_val
    if c_par:
        figures_dir = figures_dir[:-1] + "_c_scan/"
        results_dir = results_dir[:-1] + "_c_scan/"
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    # pars_dir = "../three_site_model/optimization/results/seed_0/"
    print("Saving results to %s" % results_dir, flush=True)
    
    # Load training data
    # training_data = pd.read_csv("../data/p50_training_data.csv")
    # print("Using the following training data:\n", training_data)
    # N = training_data["NFkB"]
    # I = training_data["IRF"]
    # P = training_data["p50"]
    # beta = training_data["IFNb"]
    # conditions = training_data["Stimulus"] + "_" + training_data["Genotype"]
    # num_pts = len(N)
    # len_training = len(N)
    # # stimuli = training_data["Stimulus"].unique()
    # # genotypes = training_data["Genotype"].unique()
    synthetic_data = pd.read_csv("../data/p50_training_data_plus_synthetic_e%.1fpct.csv" % (args.error_val))
    datasets = synthetic_data["Dataset"].unique()

    if args.test:
        datasets = datasets[1:4]

    one_dataset = synthetic_data.loc[synthetic_data["Dataset"] == datasets[0]]
    datapoint_names = one_dataset["Stimulus"] + "_" + one_dataset["Genotype"]

    if c_par:
        # result_par_names = [r"$t_I$",r"$t_N$",r"$t_{I\cdotIg}$", r"$t_{I\cdotN}$", "k1", "k2", "kn", "c"]
        result_par_names = ["t_1","t_3","t_4","t_5", "k1", "k2", "kn", "kp", "c"]
    else:
        # result_par_names = [r"$t_I$",r"$t_N$",r"$t_{I\cdotIg}$", r"$t_{I\cdotN}$", "k1", "k2", "kn"]
        result_par_names = ["t_1","t_3","t_4","t_5", "k1", "k2", "kn", "kp"]

    # Optimize
    if args.param_scan:
        print("###############################################\n")
        start = time.time()
        grid = calculate_grid(seed=0, num_threads=num_threads, num_t_pars=num_t_pars, num_k_pars=num_k_pars, c_par=c_par)

        
        # training_data = synthetic_data.copy()
        # For testing, only 3 datasets
        training_data = synthetic_data.loc[synthetic_data["Dataset"].isin(datasets)]
        
        datasets = parameter_scan2(training_data, grid, results_dir, h1=h1, h2=h2, num_threads=num_threads, num_t_pars=num_t_pars, num_k_pars=num_k_pars, c_par=c_par,num_to_keep=num_to_keep)
        np.savetxt("%s/dataset_names.csv" % (results_dir), datasets, delimiter=",", fmt="%s")

        end = time.time()
        t = end - start
        if t < 60*60:
            print("Time elapsed for param scan for all synthetic datasets: %.2f minutes" % (t/60), flush=True)
        else:
            print("Time elapsed for param scan for all synthetic datasets: %.2f hours" % (t/3600), flush=True)
        
        print("###############################################\n\n###############################################\n")

    if args.optimize:

        start = time.time()

        print("###############################################\n")
        print("Optimizing model\n\n")

        if not os.path.exists("%s/dataset_names.csv" % (results_dir)):
            raise FileNotFoundError("Dataset names file (%s/%s_dataset_names.csv) not found" % (results_dir, model))

        dataset_names = np.loadtxt("%s/dataset_names.csv" % (results_dir), delimiter=",", dtype=str)

        # Verify that all dataset names are in the synthetic data
        if not all([d in synthetic_data["Dataset"].values for d in dataset_names]):
            missing_names = [d for d in dataset_names if d not in synthetic_data["Dataset"].values]
            raise ValueError("Dataset names not found in synthetic data: %s" % missing_names)

        dir = "%s/optimization/" % results_dir
        os.makedirs(dir, exist_ok=True)

        for i, dataset in enumerate(dataset_names):
            print("Optimizing model using dataset %s" % dataset, flush=True)
            training_data = synthetic_data.loc[synthetic_data["Dataset"] == dataset]
            N = training_data["NFkB"].values
            I = training_data["IRF"].values
            P = training_data["p50"].values
            beta = training_data["IFNb"].values

            # Optimize the model
            print("Optimizing model...", flush=True)
            initial_pars = np.loadtxt("%s/initial_pars_%s.csv" % (results_dir, dataset), delimiter=",")
            
            final_pars, ifnb_predicted, rmsd = optimize_model(N, I, P, beta, initial_pars, [h1, h2], c=c_par, num_threads=num_threads, num_t_pars=num_t_pars, num_k_pars=num_k_pars)
            # print("Size of final_pars: %s" % str(final_pars.shape), flush=True)

            # Save all results
            np.savetxt("%s/ifnb_predicted_optimized_%s.csv" % (dir, dataset), ifnb_predicted, delimiter=",")
            np.savetxt("%s/rmsd_optimized_%s.csv" % (dir, dataset), rmsd, delimiter=",")
            np.savetxt("%s/optimized_parameters_%s.csv" % (dir, dataset), final_pars, delimiter=",")
            
            top20_pars = final_pars[np.argsort(rmsd)[:20]]
            del final_pars
            top20_pars = pd.DataFrame(top20_pars, columns=result_par_names)
            top20_pars.to_csv("%s/best_optimized_parameters_%s.csv" % (dir, dataset), index=False)

            top20_predictions = ifnb_predicted[np.argsort(rmsd)[:20]]
            del ifnb_predicted
            np.savetxt("%s/best_optimized_predictions_%s.csv" % (dir, dataset), top20_predictions, delimiter=",")

            top20_rmsd = rmsd[np.argsort(rmsd)[:20]]
            del rmsd
            np.savetxt("%s/best_optimized_rmsd_%s.csv" % (dir, dataset), top20_rmsd, delimiter=",")
            
            print("Finished optimization.", flush=True)


        end = time.time()
        t = end - start
        if t < 60*60:
            print("Time elapsed for optimization for all synthetic datasets: %.2f minutes" % (t/60), flush=True)
        else:
            print("Time elapsed for optimization for all synthetic datasets: %.2f hours" % (t/3600), flush=True)

        print("###############################################\n\n###############################################\n")

    if args.make_plots:
        # Plot optimized parameters
        final_pars_df = pd.read_csv("%s/optimization/%s_optimized_parameters.csv" % (results_dir, model))
        final_pars_df = final_pars_df.melt(id_vars=["dataset", "par_set", "rmsd"], value_vars=result_par_names, var_name="parameter", value_name="value")
        fig, ax = plt.subplots(figsize=(2.6,1.7))
        p = sns.lineplot(data=final_pars_df, x="parameter", y="value", hue="dataset", units="par_set", estimator=None, alpha=0.5, ax=ax, palette=dataset_cmap)
        sns.despine()
        plt.xticks(rotation=90)
        plt.savefig("%s/optimized_parameters_separate.png" % figures_dir)

        # Plot optimized predictions (top 20 for each dataset)
        all_predictions = np.loadtxt("%s/%s_ifnb_predicted_optimized.csv" % (results_dir, model), delimiter=",")
        all_rmsd = np.loadtxt("%s/%s_rmsd_optimized.csv" % (results_dir, model), delimiter=",")
        all_rmsd = all_rmsd.reshape(len(datasets), 100)
        all_rmsd = all_rmsd.flatten()
        all_predictions = all_predictions.reshape(len(datasets)*100, -1)
        predictions_df = pd.DataFrame(all_predictions, columns=datapoint_names)
        predictions_df["dataset"] = np.repeat(datasets, 100)
        predictions_df["par_set"] = np.tile(np.arange(100), len(datasets))
        predictions_df["rmsd"] = all_rmsd
        predictions_df = predictions_df.sort_values(by=["dataset", "par_set"]).groupby("dataset").head(20)

        predictions_df = predictions_df.melt(id_vars=["dataset", "par_set", "rmsd"], value_vars=datapoint_names, var_name="datapoint", value_name="IFNb")
        fig, ax = plt.subplots(figsize=(2.6,1.7))
        p = sns.lineplot(data=predictions_df, x="datapoint", y="IFNb", hue="dataset", units="par_set", estimator=None, alpha=0.5, ax=ax, palette=dataset_cmap)
        sns.despine()
        plt.xticks(rotation=90)
        plt.savefig("%s/optimized_predictions_separate.png" % figures_dir)

        print("###############################################\n\n###############################################\n")
        

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
