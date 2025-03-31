# For each synthetic dataset, fit best parameters and plot parameters
# Optimize the parameters of the model: t parameters and k parameters
# Initial parameters from parameter scan

from p50_model_distal_synergy import *
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
import gc

mpl.rcParams["figure.dpi"] = 600
mpl.rcParams["font.sans-serif"] = "Arial"
# irf_color = "#5D9FB5"
# nfkb_color = "#BA4961"
data_color = "#6F5987"

states_cmap_pars = "ch:s=2.2,r=0.75,h=0.6,l=0.8,d=0.25"
models_cmap_pars = "ch:s=-0.0,r=0.6,h=1,d=0.3,l=0.8,g=1_r"
dataset_cmap_pars = "ch:s=0.9,r=-0.75,h=0.6,l=0.8,d=0.1"

heatmap_cmap = sns.cubehelix_palette(as_cmap=True, light=0.95, dark=0, reverse=True, rot=0.4,start=-.2, hue=0.6)
# dataset_cmap = sns.cubehelix_palette(as_cmap=True, start=0.9, rot=-.75, dark=0.1, light=0.8, hue=0.6)

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

def get_t_pars(l_bounds, u_bounds):
    new_t_pars = np.random.uniform(l_bounds, u_bounds)
    j = 0
    while (new_t_pars[0]*2 > new_t_pars[2]) or (new_t_pars[0] + new_t_pars[1] > new_t_pars[3] or (new_t_pars[0] + new_t_pars[1] > new_t_pars[4])):
        new_t_pars = np.random.uniform(l_bounds, u_bounds)
        j += 1
        if j % 300 == 0:
            print("Number of attempts to fix this parameter set: %d" % j, flush=True)
    return new_t_pars

def calculate_grid(t_bounds=(0,1), k_bounds=(10**-3,10**3), seed=0, num_samples=10**6, num_threads=60, num_t_pars=5, num_k_pars=4, num_h_pars=2, c_par=False, restrict_t=False):
    start = time.time()

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
    if restrict_t:
        # self.t[1] = self.parsT[0] # IRF - t1
        # self.t[2] = self.parsT[0] # IRF_G - t1
        # self.t[3] = self.parsT[0] # IRF + p50 - t1
        # self.t[4] = self.parsT[1] # NFkB - t3
        # self.t[5] = self.parsT[1] # NFkB + p50 - t3
        # # 6 is zero
        # self.t[7] = self.parsT[2] # IRF + IRF_G - t4
        # self.t[8] = self.parsT[4] # IRF + NFkB - t6
        # self.t[9] = self.parsT[3] # IRF_G + NFkB - t5
        # self.t[10] = self.parsT[4] # IRF + NFkB + p50 - t6
        # self.t[11] = 1

        # 1. Get all rows where tI*2>tI1I2 or tI+tN>tIN (n rows)
        # 2. Generate n random values of t parameters such that tI*2<=tI1I2 and tI+tN<=tIN 
        # 3. Replace rows in grid
        out_of_bounds = np.where((grid_tk[:,0]*2 > grid_tk[:,2]) | (grid_tk[:,0] + grid_tk[:,1] > grid_tk[:,3]) | (grid_tk[:,0] + grid_tk[:,1] > grid_tk[:,4]))[0]
        num_out_of_bounds = len(out_of_bounds)
        if num_out_of_bounds > 0:
            new_t_pars = np.zeros((num_out_of_bounds, num_t_pars))
            print("Correcting %d parameter sets violating t constraints (%d %%)" % (num_out_of_bounds, num_out_of_bounds/num_samples*100), flush=True)
            with Pool(num_threads) as p:
                new_t_pars = p.starmap(get_t_pars, [(l_bounds[:num_t_pars], u_bounds[:num_t_pars]) for _ in range(num_out_of_bounds)])
            grid_tk[out_of_bounds,:num_t_pars] = np.array(new_t_pars)

        # Verify that all rows are within bounds
        out_of_bounds = np.where((grid_tk[:,0]*2 > grid_tk[:,2]) | (grid_tk[:,0] + grid_tk[:,1] > grid_tk[:,3]) | (grid_tk[:,0] + grid_tk[:,1] > grid_tk[:,4]))[0]
        if len(out_of_bounds) > 0:
            raise ValueError("Number of out of bounds parameter sets after correction: %d" % len(out_of_bounds))

    end = time.time()
    t = end - start
    if t < 60:
        print("Time elapsed for grid calculation: %.2f seconds" % t, flush=True)
    elif t < 3600:
        print("Time elapsed for grid calculation: %.2f minutes" % (t/60), flush=True)
    else:
        print("Time elapsed for grid calculation: %.2f hours" % (t/3600), flush=True)

    return grid_tk


def parameter_scan(training_data, grid, results_dir, h1=3, h2=1, num_threads=40, num_t_pars=5, num_k_pars=4, c_par=False, num_to_keep=100):
    print("\n\nBeginning parameter scan", flush=True)
    # Load data points
    datasets = training_data["Dataset"].unique()
    num_datasets = len(datasets)
    num_data_points = int(len(training_data)/num_datasets)

    if len(training_data) % num_datasets != 0:
        raise ValueError("Number of data points (%d) is not divisible by number of datasets (%d)" % (len(training_data), num_datasets))

    # Reduce type to float32
    training_data = training_data.astype({
        "NFkB": np.float32,
        "IRF": np.float32,
        "p50": np.float32,
        "IFNb": np.float32
    })

    # Only use some datasets at a time
    max_datasets = 10
    num_chunks = -(-num_datasets // max_datasets)  

    chunks = np.array_split(datasets, num_chunks)
    for c, chunk in enumerate(chunks):
        data_chunk = training_data.loc[training_data["Dataset"].isin(chunk)]
        N, I, P = data_chunk["NFkB"].values, data_chunk["IRF"].values, data_chunk["p50"].values
        beta = data_chunk["IFNb"].values
        dataset_vector = data_chunk["Dataset"].values
        num_pts_times_num_datasets = len(N)
        num_param_sets = len(grid)
        num_datasets_chunk = len(chunk)
        del data_chunk
        gc.collect()

        # Calculate IFNb value at each point in grid
        print("Calculating effects at %.1E parameter sets for %d points for %d datasets (%.1E total)" % (num_param_sets, num_data_points, num_datasets_chunk, num_pts_times_num_datasets*num_param_sets), flush=True)
        grid = grid.astype(np.float32)

        start = time.time()
        if c_par:
            with Pool(num_threads) as p:
                results = p.starmap(get_f, [(grid[i,0:num_t_pars], grid[i,num_t_pars:-1], N[j], I[j], P[j], grid[i,-1], [h1, h2], False) for i in range(num_param_sets) for j in range(num_pts_times_num_datasets)])
        else:
            with Pool(num_threads) as p:
                results = p.starmap(get_f, [(grid[i,0:num_t_pars], grid[i,num_t_pars:], N[j], I[j], P[j], None, [h1, h2], False) for i in range(num_param_sets) for j in range(num_pts_times_num_datasets)])


        ifnb_predicted = np.array(results).reshape(num_param_sets, num_pts_times_num_datasets)
        del results
        gc.collect()

        # Change to float32
        ifnb_predicted = ifnb_predicted.astype(np.float32)

        print("Memory used by ifnb_predicted: %.2f GB" % (ifnb_predicted.nbytes/10**9), flush=True)

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
        gc.collect()

        # Calculate residuals
        with Pool(num_threads) as p:
            rmsd = p.starmap(calculate_rmsd, [(ifnb_predicted_bydset[j][i], beta_bydset[j]) for i in range(num_param_sets) for j in range(num_datasets_chunk)])
        del beta_bydset
        gc.collect()

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
            gc.collect()

            print("Saved parameter scan results for dataset %s" % dname, flush=True)

        num_complete = (c+1)*num_datasets_chunk
        # print("Finished with %d datasets\n" % num_complete, flush=True)
        print("%s: Finished with %d datasets" % (time.ctime(), num_complete), flush=True)

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
    N, I, P, beta, h_pars, c_par, restrict_t = args
    t_pars = pars[0:5]
    k_pars = pars[5:9]
    if c_par:
        c = pars[9]
    else:
        c = None

    num_pts = len(N)
    
    ifnb_predicted = [get_f(t_pars, k_pars, N[i], I[i], P[i], h_pars=h_pars, c_par=c) for i in range(num_pts)]

    rmsd = calculate_rmsd(ifnb_predicted, beta)

    # t0: tI, t1:tN, t2: tII, t3: tIN
    # out_of_bounds = np.where((grid_tk[:,0]*2 > grid_tk[:,2]) | (grid_tk[:,0] + grid_tk[:,1] > grid_tk[:,3]) | (grid_tk[:,0] + grid_tk[:,1] > grid_tk[:,4]))[0]
    if restrict_t:
        if (t_pars[2] < 2*t_pars[0]) or (t_pars[3] < t_pars[0] + t_pars[1]) or (t_pars[4] < t_pars[0] + t_pars[1]):
            rmsd = 100

    return rmsd

def minimize_objective(pars, N, I, P, beta, h_pars, c_par, bounds, restrict_t):
    return opt.minimize(objective_function, pars, args=(N, I, P, beta, h_pars, c_par, restrict_t), method="Nelder-Mead", bounds=bounds)

def optimize_model(N, I, P, beta, initial_pars, h, c=False, num_threads=40, num_t_pars=5, num_k_pars=3, restrict_t=True):
    start = time.time()    
    min_k_order = -3
    max_k_order = 3
    min_c_order = -3
    max_c_order = 3

    # Define bounds
    bnds = [(0, 1) for i in range(num_t_pars)] + [(10**min_k_order, 10**max_k_order) for i in range(num_k_pars)]
    if c:
        bnds.append((10**min_c_order, 10**max_c_order))
    bnds = tuple(bnds)

    # Optimize
    with Pool(num_threads) as p:
        results = p.starmap(minimize_objective, [(pars, N, I, P, beta, h, c, bnds, restrict_t) for pars in initial_pars])

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

def make_parameters_data_frame(pars):
    df_pars = pars.drop(columns=["h1", "h2", "h3", "rmsd"], errors="ignore")

    df_pars["t_0"] = 0
    df_pars["t_I1I2N"] = 1
    df_pars = df_pars.melt(var_name="Parameter", value_name="Value", id_vars="Dataset")
    # df_t_pars = df_pars[df_pars["Parameter"].str.startswith("t")]
    df_t_pars = df_pars.loc[df_pars["Parameter"].str.startswith("t")].copy()
    num_t_pars = len(df_t_pars["Parameter"].unique())

    new_t_par_names = [r"$t_{I}$", r"$t_{I}$", r"$t_N$", r"$t_{I_1I_2}$", r"$t_{I_1N}$", r"$t_{I_2N}$"]
    # Rename t parameters
    df_t_pars["Parameter"] = df_t_pars["Parameter"].replace(["t1", "t2", "t3", "t4", "t5", "t6"], new_t_par_names)
    df_t_pars["Parameter"] = df_t_pars["Parameter"].replace(["t_1", "t_2", "t_3", "t_4", "t_5", "t_6"], new_t_par_names)
    df_t_pars["Parameter"] = df_t_pars["Parameter"].replace(["t_0", "t_I1I2N"], [r"$t_0$", r"$t_{I_1I_2N}$"])
    new_t_par_order = [r"$t_0$",r"$t_{I}$", r"$t_N$", r"$t_{I_1I_2}$", r"$t_{I_1N}$", r"$t_{I_2N}$", r"$t_{I_1I_2N}$"]
    df_t_pars["Parameter"] = pd.Categorical(df_t_pars["Parameter"], categories=new_t_par_order, ordered=True)

    df_k_pars = df_pars.loc[df_pars["Parameter"].str.startswith("k") | df_pars["Parameter"].str.startswith("c")].copy()
    num_k_pars = len(df_k_pars["Parameter"].unique())
    # df_k_pars["Parameter"] = df_k_pars["Parameter"].str.replace("k3", r"$K_N$")
    # df_k_pars["Parameter"] = df_k_pars["Parameter"].str.replace("k2", r"$k_2$")
    # df_k_pars["Parameter"] = df_k_pars["Parameter"].str.replace("k1", r"$k_1$")
    # df_k_pars["Parameter"] = df_k_pars["Parameter"].str.replace("kn", r"$K_N$")
    df_k_pars.loc[df_k_pars["Parameter"] == "k1", "Parameter"] = r"$k_{I_2}$" # Rename
    df_k_pars.loc[df_k_pars["Parameter"] == "k2", "Parameter"] = r"$k_{I_1}$" # Rename
    df_k_pars.loc[df_k_pars["Parameter"] == "kn", "Parameter"] = r"$K_N$"
    df_k_pars.loc[df_k_pars["Parameter"] == "k3", "Parameter"] = r"$K_N$"
    df_k_pars.loc[df_k_pars["Parameter"] == "kp", "Parameter"] = r"$K_P$"
    df_k_pars.loc[df_k_pars["Parameter"] == "k4", "Parameter"] = r"$K_P$"
    df_k_pars.loc[df_k_pars["Parameter"] == "c", "Parameter"] = r"$C$"
    df_k_pars["Parameter"] = pd.Categorical(df_k_pars["Parameter"], categories=[r"$k_{I_1}$", r"$k_{I_2}$", r"$K_N$", r"$K_P$",r"$C$"], ordered=True)
    return df_t_pars, df_k_pars, num_t_pars, num_k_pars

def make_predictions_data_frame(df_ifnb_predicted):
    df_ifnb_predicted["Stimulus"] = df_ifnb_predicted["Data point"].str.split("_", expand=True)[0]
    df_ifnb_predicted["Stimulus"] = df_ifnb_predicted["Stimulus"].replace("polyIC", "PolyIC")

    df_ifnb_predicted["Genotype"] = df_ifnb_predicted["Data point"].str.split("_", expand=True)[1]
    df_ifnb_predicted["Category"] = "Stimulus specific"
    df_ifnb_predicted.loc[df_ifnb_predicted["Genotype"].str.contains("rela"), "Category"] = "NFκB dependence"
    df_ifnb_predicted.loc[df_ifnb_predicted["Genotype"].str.contains("irf"), "Category"] = "IRF dependence"
    df_ifnb_predicted.loc[df_ifnb_predicted["Genotype"].str.contains("p50"), "Category"] = "p50 dependence"

    df_ifnb_predicted["Genotype"] = df_ifnb_predicted["Genotype"].replace("relacrelKO", r"NFκBko")
    df_ifnb_predicted["Genotype"] = df_ifnb_predicted["Genotype"].replace("irf3irf7KO", "IRF3/7ko")
    df_ifnb_predicted["Genotype"] = df_ifnb_predicted["Genotype"].replace("irf3irf5irf7KO", "IRF3/5/7ko")
    df_ifnb_predicted["Genotype"] = df_ifnb_predicted["Genotype"].replace("p50KO", "p50ko")
    df_ifnb_predicted["Data point"] = df_ifnb_predicted["Stimulus"] + " " + df_ifnb_predicted["Genotype"]    
    stimuli_levels = ["basal", "CpG", "LPS", "PolyIC"]
    # genotypes_levels = ["WT", "irf3irf7KO", "irf3irf5irf7KO", "relacrelKO"]
    genotypes_levels = ["WT","p50ko", "IRF3/7ko", "IRF3/5/7ko", r"NFκBko"]
    df_ifnb_predicted["Stimulus"] = pd.Categorical(df_ifnb_predicted["Stimulus"], categories=stimuli_levels, ordered=True)
    df_ifnb_predicted["Genotype"] = pd.Categorical(df_ifnb_predicted["Genotype"], categories=genotypes_levels, ordered=True)
    df_ifnb_predicted = df_ifnb_predicted.sort_values(["Stimulus", "Genotype"])
    # print(df_ifnb_predicted)
    return df_ifnb_predicted

def make_all_plots(synthetic_data, results_dir, figures_dir, err):
        datasets = np.loadtxt("%s/dataset_names.csv" % results_dir, delimiter=",", dtype=str)
        dir = "%s/optimization/" % results_dir
        all_optimized_pars = pd.DataFrame()
        all_optimized_predictions = pd.DataFrame()

        for i, dataset in enumerate(datasets):
            optimized_pars = pd.read_csv("%s/best_optimized_parameters_%s.csv" % (dir, dataset))
            optimized_pars["Dataset"] = dataset
            all_optimized_pars = pd.concat([all_optimized_pars, optimized_pars])

            training_data = synthetic_data.loc[synthetic_data["Dataset"] == dataset]
            datapoint_names = training_data["Stimulus"] + "_" + training_data["Genotype"]

            optimized_predictions = np.loadtxt("%s/best_optimized_predictions_%s.csv" % (dir, dataset), delimiter=",")
            optimized_predictions = pd.DataFrame(optimized_predictions, columns=datapoint_names)
            optimized_predictions["Dataset"] = dataset
            all_optimized_predictions = pd.concat([all_optimized_predictions, optimized_predictions])

        # # Save combined data
        dir = "%s_combined/" % results_dir.rstrip("/")
        os.makedirs(dir, exist_ok=True)
        print("Saving combined data to %s" % dir, flush=True)
        all_optimized_pars.to_csv("%s/best_optimized_pars_combined.csv" % dir, index=False)
        all_optimized_predictions.to_csv("%s/best_optimized_predictions_combined.csv" % dir, index=False)

        optimized_predictions_df = all_optimized_predictions.melt(id_vars="Dataset", var_name="Data point", value_name=r"IFN$\beta$")

        # Plot optimized parameters
        print("Plotting optimized parameters", flush=True)

        df_all_t_pars, df_all_k_pars, num_t_pars, num_k_pars = make_parameters_data_frame(all_optimized_pars)

        colors = sns.color_palette(dataset_cmap_pars, len(datasets))

        # with sns.plotting_context("paper",rc=plot_rc_pars):
        #     width = 2.8
        #     height = 1
        #     fig, ax = plt.subplots(1,2, figsize=(width, height), 
        #                         gridspec_kw={"width_ratios":[num_t_pars, num_k_pars]})
            
        #     legend_handles = []
        #     for i, dset in enumerate(datasets):
        #         # Filter data for the current model
        #         df_model = df_all_t_pars[df_all_t_pars["Dataset"] == dset]
        #         l = sns.stripplot(data=df_model, x="Parameter", y="Value", color=colors[i], ax=ax[0], zorder = i, alpha=0.2)

        #         legend_handles.append(l)

        #         df_model = df_all_k_pars[df_all_k_pars["Dataset"] == dset]
        #         sns.stripplot(data=df_model, x="Parameter", y="Value", color=colors[i], ax=ax[1], zorder = i, alpha=0.2)
            
        #     ax[1].set_yscale("log")
        #     ax[1].set_ylabel("")

        #     ax[0].set_ylabel("Parameter Value")
        #     ax[0].set_xlabel("")
        #     ax[1].set_xlabel("")

        #     sns.despine()
        #     plt.tight_layout()
        #     plt.savefig("%s/optimized_parameters_old_plot.png" % figures_dir, bbox_inches="tight")
        #     plt.close()

        #######
        col  = sns.color_palette(models_cmap_pars, n_colors=4)[1]
        k_parameters = [r"$k_{I_1}$",r"$K_N$",r"$K_P$"]

        df_all_t_pars["Source"] = "Synthetic"
        df_all_t_pars.loc[df_all_t_pars["Dataset"].str.contains(datasets[0]), "Source"] = "Original"
        df_all_t_pars["Source"] = pd.Categorical(df_all_t_pars["Source"], categories=["Original", "Synthetic"], ordered=True)

        df_all_k_pars["Source"] = "Synthetic"
        df_all_k_pars.loc[df_all_k_pars["Dataset"].str.contains(datasets[0]), "Source"] = "Original"
        df_all_k_pars["Source"] = pd.Categorical(df_all_k_pars["Source"], categories=["Original", "Synthetic"], ordered=True)

        with sns.plotting_context("paper",rc=plot_rc_pars):
            width = 2.8
            height = 1
            fig, ax = plt.subplots(1,2, figsize=(width, height), 
                                gridspec_kw={"width_ratios":[num_t_pars, 2.5]})
        
            # unique_models = np.unique(df_all_t_pars["Model"])
            # legend_handles = []

            cols = ["black", col]

            s = sns.stripplot(data=df_all_t_pars, x="Parameter", y="Value", hue ="Source", palette=cols, ax=ax[0], zorder = 0, linewidth=0,
                                alpha=0.2, jitter=0.1, dodge=True, legend=False)

            legend_handles = s.collections
        

            df2 = df_all_k_pars[(df_all_k_pars["Parameter"].isin(k_parameters))]
            df2 = df2.copy()
            df2["Parameter"] = df2["Parameter"].cat.remove_unused_categories()
            s = sns.stripplot(data=df2, x="Parameter", y="Value", hue ="Source", palette=cols, ax=ax[1], zorder = 0, linewidth=0, 
                            alpha=0.2, jitter=0.1, dodge=True, legend=False)        
        
            ax[1].set_yscale("log")
            ax[1].set_ylabel(r"Value (MNU$^{-1}$)")
            
            ax1_xtick_labels = ax[1].get_xticklabels()
            # Replace ", " with "\n" in xtick labels
            new_xtick_labels = [label.get_text().replace(", $h_{I_2}$=", "\n") for label in ax1_xtick_labels]
            new_xtick_labels = [label.replace("$h_{I_1}$=", "") for label in new_xtick_labels]

            ax[1].set_xticklabels(new_xtick_labels)

            ax[0].set_ylabel("Parameter Value")
            
            for x in ax[0], ax[1]:
                x.set_xlabel("")

            sns.despine()
            plt.tight_layout()
            leg = fig.legend(legend_handles, ["Original", "Synthetic"], loc="lower center", bbox_to_anchor=(0.5, 1), frameon=False, 
                            ncol=4, columnspacing=1, handletextpad=0.5, handlelength=1.5)

            for i in range(len(leg.legend_handles)):
                leg.legend_handles[i].set_alpha(1)
                leg.legend_handles[i].set_color(cols[i])

            plt.savefig("%s/optimized_parameters_combined.png" % figures_dir, bbox_inches="tight")
            plt.close()



            # # Plot original dataset only
            # fig, ax = plt.subplots(1,2, figsize=(width, height),
            #                     gridspec_kw={"width_ratios":[num_t_pars, num_k_pars]})
            # legend_handles = []
            # df_model = df_all_t_pars[df_all_t_pars["Dataset"] == datasets[0]]
            # l = sns.stripplot(data=df_model, x="Parameter", y="Value", color=colors[0], ax=ax[0], zorder = 0, alpha=0.2)
            # legend_handles.append(l)
            # df_model = df_all_k_pars[df_all_k_pars["Dataset"] == datasets[0]]
            # sns.stripplot(data=df_model, x="Parameter", y="Value", color=colors[0], ax=ax[1], zorder = 0, alpha=0.2)
            # ax[1].set_yscale("log")
            # ax[1].set_ylabel("")
            # ax[0].set_ylabel("Parameter Value")
            # ax[0].set_xlabel("")
            # ax[1].set_xlabel("")
            # sns.despine()
            # plt.tight_layout()
            # plt.savefig("%s/optimized_parameters_original.png" % figures_dir, bbox_inches="tight")
            # plt.close()

        # Plot optimized predictions
        print("Plotting optimized predictions", flush=True)

        optimized_predictions_df = make_predictions_data_frame(optimized_predictions_df)

        data_df = pd.read_csv("../data/p50_training_data.csv")
        data_df["Data point"] = data_df["Stimulus"] + "_" + data_df["Genotype"]
        data_df = data_df.rename(columns={"IFNb": r"IFN$\beta$"})
        data_df = make_predictions_data_frame(data_df)
        new_data_color = "#A7535E"
        
        with sns.plotting_context("paper", rc=plot_rc_pars):
            # Scatterplot, x = datapoint name, y= IFNb value, color = dataset
            fig, ax = plt.subplots(figsize=(3,2))
            p = sns.stripplot(data=optimized_predictions_df, x="Data point", y=r"IFN$\beta$", hue="Dataset", palette=sns.color_palette(dataset_cmap_pars, len(datasets)),
                              alpha=0.7, jitter=0.2, ax=ax, legend=False)
            sns.stripplot(data=data_df, x="Data point", y=r"IFN$\beta$", color=new_data_color, alpha=0.7, jitter=0.2, ax=ax)
            plt.xticks(rotation=45)
            # plt.legend(title="Dataset", bbox_to_anchor=(1,1), loc="upper left")
            plt.tight_layout()
            plt.savefig("%s/optimized_predictions_combined.png" % figures_dir, bbox_inches="tight")
            plt.close()

            # Plot original dataset only
            data_subset = optimized_predictions_df.loc[optimized_predictions_df["Dataset"] == datasets[0]]
            fig, ax = plt.subplots(figsize=(3,2))
            p = sns.stripplot(data=data_subset, x="Data point", y=r"IFN$\beta$", color=colors[0],
                                alpha=0.7, jitter=0.2, ax=ax, legend=False)
            sns.stripplot(data=data_df, x="Data point", y=r"IFN$\beta$", color=new_data_color, alpha=0.7, jitter=0.2, ax=ax)
            plt.xticks(rotation=45)
            # plt.legend(title="Dataset", bbox_to_anchor=(1,1), loc="upper left")
            plt.tight_layout()
            plt.savefig("%s/optimized_predictions_original.png" % figures_dir, bbox_inches="tight")
            plt.close()


        # Plot measurements of conclusions
        # 1. tI1I2, tI1N > 0,1 & tI, tN < 0.1
        # 2. tI1I2 > 2*tI & tI1N > tI1 + tN

        # Measure two modes of activation (#1)
        # df_all_t_pars = df_all_t_pars.loc[df_all_t_pars["Dataset"] == datasets[0]] # for now, only use first dataset
        df_all_t_pars["par_set"] = df_all_t_pars.groupby(["Dataset", "Parameter"]).cumcount()
        # Count number of parameter sets where tI1I2 > 0.1
        df_all_t_pars = df_all_t_pars.pivot_table(index=["Dataset", "par_set"], columns="Parameter", values="Value")
        
        df_all_t_pars = df_all_t_pars.reset_index()

        df_all_t_pars["tI1I2_tI1N_high"] = (df_all_t_pars["$t_{I_1I_2}$"] > 0.5) & (df_all_t_pars["$t_{I_1N}$"] > 0.5)
        df_all_t_pars["tI_tN_low"] = (df_all_t_pars["$t_{I}$"] < 0.2) & (df_all_t_pars["$t_N$"] < 0.2)
        df_all_t_pars["two_modes"] = df_all_t_pars["tI1I2_tI1N_high"] & df_all_t_pars["tI_tN_low"]

        # Measure synergy (#2)
        df_all_t_pars["tI1I2_gt_2tI"] = df_all_t_pars["$t_{I_1I_2}$"] > 2*df_all_t_pars["$t_{I}$"]
        df_all_t_pars["tI1N_gt_tI1_tN"] = df_all_t_pars["$t_{I_1N}$"] > df_all_t_pars["$t_{I}$"] + df_all_t_pars["$t_N$"]
        df_all_t_pars["synergy"] = df_all_t_pars["tI1I2_gt_2tI"] & df_all_t_pars["tI1N_gt_tI1_tN"]
        # print(df_all_t_pars)

        # For each dataset, calculate % of parameter sets that satisfy each condition
        conclusions_df = df_all_t_pars.copy()
        conclusions_df = conclusions_df.groupby("Dataset").agg({"two_modes": "mean", "synergy": "mean"}).reset_index()
        conclusions_df.to_csv("%s/conclusions.csv" % dir, index=False)

        # Plot datasets colored by conclusion
        conclusions_df_dsets = conclusions_df.copy()
        # conclusions_df_dsets["two_modes"] = conclusions_df_dsets["two_modes"] > 0.5
        # conclusions_df_dsets["synergy"] = conclusions_df_dsets["synergy"] > 0.5
        # conclusions_df_dsets["both"] = conclusions_df_dsets["two_modes"] & conclusions_df_dsets["synergy"]
        # conclusions_df_dsets["none"] = ~(conclusions_df_dsets["two_modes"]) & ~(conclusions_df_dsets["synergy"])
        # conclusions_df_dsets["two_modes"] = conclusions_df_dsets["two_modes"] & ~(conclusions_df_dsets["both"])
        # conclusions_df_dsets["synergy"] = conclusions_df_dsets["synergy"] & ~(conclusions_df_dsets["both"])
        # conclusions_df_dsets["Conclusion"] = ["Both" if b else "Synergy" if s else "Two modes of activation" if t else "Neither" for b, s, t in zip(conclusions_df_dsets["both"], conclusions_df_dsets["synergy"], conclusions_df_dsets["two_modes"])]
        # conclusions_df_dsets = conclusions_df_dsets.drop(columns=["two_modes", "synergy", "both", "none"])
        conclusions_df_dsets["Conclusion"] = np.where((conclusions_df_dsets["two_modes"] > 0.5) & (conclusions_df_dsets["synergy"] > 0.5), "Both", 
                                                      np.where(conclusions_df_dsets["two_modes"] > 0.5, "Two modes of activation", 
                                                               np.where(conclusions_df_dsets["synergy"] > 0.5, "Synergy", 
                                                                        "Neither")))
        conclusions_df_dsets.to_csv("%s/conclusions_summary.csv" % dir, index=False)
        
        # Join conclusions with all datasets
        all_datasets = synthetic_data.merge(conclusions_df_dsets, on="Dataset", how="left")

        # Plot IRF vs NFkB, colored by conclusion
        with sns.plotting_context("paper", rc=plot_rc_pars):
            fig, ax = plt.subplots(figsize=(2,2))
            sns.scatterplot(data=all_datasets, x="NFkB", y="IRF", hue="Conclusion", palette=sns.color_palette("Set2", 4), ax=ax)
            plt.tight_layout()
            sns.move_legend(ax, "lower center", bbox_to_anchor=(0.5, 1), ncol=2)
            plt.savefig("%s/irf_nfkb_by_conclusions.png" % figures_dir, bbox_inches="tight")
            plt.close()

        # Plot conclusions
        conclusions_df = pd.read_csv("%s/conclusions.csv" % dir)
        conclusions_df = pd.melt(conclusions_df, id_vars="Dataset", var_name="Conclusion", value_name="Fraction")
        conclusions_df["Conclusion"] = conclusions_df["Conclusion"].replace("two_modes", "Two modes of activation")
        conclusions_df["Conclusion"] = conclusions_df["Conclusion"].replace("synergy", "Synergy")
        color = sns.color_palette(dataset_cmap_pars, len(datasets))[0]
        with sns.plotting_context("paper", rc=plot_rc_pars):
            fig, ax = plt.subplots(figsize=(2,2))
            # sns.barplot(data=conclusions_df, x="Conclusion", y="Fraction", ax=ax, color=color,
            #             estimator=np.mean, errorbar="sd")
            sns.boxplot(data=conclusions_df, x="Conclusion", y="Fraction", ax=ax, color=color)
            plt.tight_layout()
            plt.ylim(-0.05, 1.05)
            plt.savefig("%s/conclusions.png" % figures_dir, bbox_inches="tight")
            plt.close()

        
        # # Plot conclusions as violin plot
        # with sns.plotting_context("paper", rc=plot_rc_pars):
        #     fig, ax = plt.subplots(figsize=(2,2))
        #     sns.swarmplot(data=conclusions_df, x="Conclusion", y="Fraction", ax=ax, color=color, size=2)
        #     sns.violinplot(data=conclusions_df, x="Conclusion", y="Fraction", ax=ax, inner=None, fill=False, color="black")
        #     plt.tight_layout()
        #     plt.ylim(-0.05, 1.05)
        #     plt.savefig("%s/conclusions_violin.png" % figures_dir, bbox_inches="tight")
        #     plt.close()
    
        # Plot as side by side histograms
        num_par_sets_per_dset = len(all_optimized_pars.loc[all_optimized_pars["Dataset"] == datasets[0]])
        with sns.plotting_context("paper", rc=plot_rc_pars):
            sns.displot(data=conclusions_df, y="Fraction", col="Conclusion", kind="hist", bins = num_par_sets_per_dset, color=color,
                        height=2, aspect=1.2)
            plt.tight_layout()
            plt.ylim(-0.05, 1.05)
            plt.savefig("%s/conclusions_hist.png" % figures_dir, bbox_inches="tight")

        # # Count # of rows where two_modes is true, synergy is true, no k value is > 1000
        # print("Plotting acceptable and bad parameter sets", flush=True)
        # all_optimized_pars = pd.read_csv("%s/best_optimized_pars_combined.csv" % dir)
        # # print(all_optimized_pars)
        # all_optimized_pars["low_rows"] = (all_optimized_pars["t_1"] <= 0.2) & (all_optimized_pars["t_3"] <= 0.2)
        # all_optimized_pars["high_rows"] = (all_optimized_pars["t_4"] > 0.5) & (all_optimized_pars["t_5"] > 0.5)
        # all_optimized_pars["k_reasonable"] = (all_optimized_pars["k1"] <= 1000) & (all_optimized_pars["k2"] <= 1000) & (all_optimized_pars["kn"] <= 1000) & (all_optimized_pars["kp"] <= 1000)
        # all_optimized_pars["acceptable"] = (all_optimized_pars["low_rows"]) & (all_optimized_pars["high_rows"]) & (all_optimized_pars["k_reasonable"])
        # all_optimized_pars["bad"] = ~(all_optimized_pars["acceptable"])

        # # Plot count of acceptable parameter sets and bad parameter sets per dataset
        # summary = all_optimized_pars.groupby("Dataset").agg({"acceptable": "sum", "bad": "sum"}).reset_index()
        # summary = pd.melt(summary, id_vars="Dataset", var_name="Conclusion", value_name="Count")
        # summary["Conclusion"] = summary["Conclusion"].replace("acceptable", r"Good $t$")
        # summary["Conclusion"] = summary["Conclusion"].replace("bad", r"Bad $t$")
        # # Remove "synthetic_" from dataset names
        # summary["Dataset"] = summary["Dataset"].str.replace("synthetic_", "")

        # with sns.plotting_context("paper", rc=plot_rc_pars):
        #     fig, ax = plt.subplots(figsize=(13,2))
        #     # Stack barplot
        #     # ax = sns.histplot(data=contrib_df, x="Condition", hue="state", weights="contribution", multiple="stack", shrink=0.8,
        #     #               palette=states_colors, ax=ax, linewidth=0.5)
        #     sns.histplot(data=summary, x="Dataset", hue="Conclusion", weights="Count", multiple="stack", shrink=0.8, ax=ax, linewidth=0.5,
        #                  palette=sns.color_palette("Set2", 2))
        #     plt.tight_layout()
        #     sns.move_legend(ax, "lower center", bbox_to_anchor=(0.5, 1), ncol=3)
        #     plt.savefig("%s/acceptable_bad_pars.png" % figures_dir, bbox_inches="tight")
        #     plt.close()

        # # For each data set count the percantage of bad t reasonable k and acceptable
        # summary_by_dset = all_optimized_pars.groupby("Dataset").agg({"acceptable": "mean", "bad": "mean"}).reset_index()
        # summary_by_dset = pd.melt(summary_by_dset, id_vars="Dataset", var_name="Conclusion", value_name="Fraction")
        # summary_by_dset["Conclusion"] = summary_by_dset["Conclusion"].replace("acceptable","Good")
        # summary_by_dset["Conclusion"] = summary_by_dset["Conclusion"].replace("bad", "Bad")

        # print(summary_by_dset)

        # with sns.plotting_context("paper", rc=plot_rc_pars):
        #     fig, ax = plt.subplots(figsize=(2,2))
        #     sns.boxplot(data=summary_by_dset, x="Conclusion", y="Fraction", ax=ax)
        #     plt.tight_layout()
        #     plt.xticks(rotation=45)
        #     plt.ylabel("Fraction of dataset")
        #     plt.savefig("%s/acceptable_bad_pars_boxplot.png" % figures_dir, bbox_inches="tight")
        #     plt.close()

        # print("Plotting bad predictions", flush=True)
        # predictions_df = pd.read_csv("%s/best_optimized_predictions_combined.csv" % dir)

        # if len(all_optimized_pars) != len(predictions_df):
        #     raise ValueError("Number of parameter sets (%d) does not match number of predictions (%d)" % (len(all_optimized_pars), len(predictions_df)))

        # bad_predictions = predictions_df.loc[all_optimized_pars["acceptable"] == False]

        # num_bad_pars = len(bad_predictions)
        # if num_bad_pars > 0:
        #     bad_predictions = bad_predictions.melt(id_vars="Dataset", var_name="Data point", value_name=r"IFN$\beta$")
        #     bad_predictions = make_predictions_data_frame(bad_predictions)

        #     with sns.plotting_context("paper", rc=plot_rc_pars):
        #         fig, ax = plt.subplots(figsize=(3,2))
        #         p = sns.stripplot(data=bad_predictions, x="Data point", y=r"IFN$\beta$", hue="Dataset", palette=sns.color_palette(dataset_cmap_pars, len(bad_predictions["Dataset"].unique())),
        #                         alpha=0.7, jitter=0.2, ax=ax, legend=False)
        #         sns.stripplot(data=data_df, x="Data point", y=r"IFN$\beta$", color=new_data_color, alpha=0.7, jitter=0.2, ax=ax)
        #         plt.xticks(rotation=45)
        #         # plt.legend(title="Dataset", bbox_to_anchor=(1,1), loc="upper left")
        #         plt.tight_layout()
        #         plt.savefig("%s/bad_predictions.png" % figures_dir, bbox_inches="tight")
        #         plt.close()


        #     bad_pars = all_optimized_pars.loc[all_optimized_pars["acceptable"] == False]
        #     bad_pars = bad_pars.loc[:,:"Dataset"]
        #     df_bad_t_pars, df_bad_k_pars, num_t_pars, num_k_pars = make_parameters_data_frame(bad_pars)

        #     with sns.plotting_context("paper", rc=plot_rc_pars):
        #         width = 2.8
        #         height = 1
        #         fig, ax = plt.subplots(1,2, figsize=(width, height), 
        #                             gridspec_kw={"width_ratios":[num_t_pars, num_k_pars]})
                
        #         legend_handles = []
        #         for i, dset in enumerate(datasets):
        #             # Filter data for the current model
        #             if dset in bad_pars["Dataset"].values:
        #                 df_model = df_bad_t_pars[df_bad_t_pars["Dataset"] == dset]
        #                 l = sns.stripplot(data=df_model, x="Parameter", y="Value", color=colors[i], ax=ax[0], zorder = i, alpha=0.2)

        #                 legend_handles.append(l)

        #                 df_model = df_bad_k_pars[df_bad_k_pars["Dataset"] == dset]
        #                 sns.stripplot(data=df_model, x="Parameter", y="Value", color=colors[i], ax=ax[1], zorder = i, alpha=0.2)

        #         ax[1].set_yscale("log")
        #         ax[1].set_ylabel("")
        #         ax[0].set_ylabel("Parameter Value")
        #         ax[0].set_xlabel("")
        #         ax[1].set_xlabel("")
        #         sns.despine()
        #         plt.tight_layout()
        #         plt.savefig("%s/bad_parameters.png" % figures_dir, bbox_inches="tight")
        # else:
        #     print("No bad predictions found", flush=True)

        # print("Plotting good predictions", flush=True)
        # good_predictions = predictions_df.loc[all_optimized_pars["acceptable"] == True]
        # good_predictions = good_predictions.melt(id_vars="Dataset", var_name="Data point", value_name=r"IFN$\beta$")
        # good_predictions = make_predictions_data_frame(good_predictions)

        # with sns.plotting_context("paper", rc=plot_rc_pars):
        #     fig, ax = plt.subplots(figsize=(3,2))
        #     p = sns.stripplot(data=good_predictions, x="Data point", y=r"IFN$\beta$", hue="Dataset", palette=sns.color_palette(dataset_cmap_pars, len(datasets)),
        #                       alpha=0.7, jitter=0.2, ax=ax, legend=False)
        #     sns.stripplot(data=data_df, x="Data point", y=r"IFN$\beta$", color=new_data_color, alpha=0.7, jitter=0.2, ax=ax)
        #     plt.xticks(rotation=45)
        #     # plt.legend(title="Dataset", bbox_to_anchor=(1,1), loc="upper left")
        #     plt.tight_layout()
        #     plt.savefig("%s/good_predictions.png" % figures_dir, bbox_inches="tight")
        #     plt.close()

        # good_pars = all_optimized_pars.loc[all_optimized_pars["acceptable"] == True]
        # good_pars = good_pars.loc[:,:"Dataset"]
        # df_good_t_pars, df_good_k_pars, num_t_pars, num_k_pars = make_parameters_data_frame(good_pars)

        # with sns.plotting_context("paper", rc=plot_rc_pars):
        #     width = 2.8
        #     height = 1
        #     fig, ax = plt.subplots(1,2, figsize=(width, height), 
        #                         gridspec_kw={"width_ratios":[num_t_pars, num_k_pars]})
            
        #     legend_handles = []
        #     for i, dset in enumerate(datasets):
        #         # Filter data for the current model
        #         if dset in good_pars["Dataset"].values:
        #             df_model = df_good_t_pars[df_good_t_pars["Dataset"] == dset]
        #             l = sns.stripplot(data=df_model, x="Parameter", y="Value", color=colors[i], ax=ax[0], zorder = i, alpha=0.2)

        #             legend_handles.append(l)

        #             df_model = df_good_k_pars[df_good_k_pars["Dataset"] == dset]
        #             sns.stripplot(data=df_model, x="Parameter", y="Value", color=colors[i], ax=ax[1], zorder = i, alpha=0.2)

        #     ax[1].set_yscale("log")
        #     ax[1].set_ylabel("")
        #     ax[0].set_ylabel("Parameter Value")
        #     ax[0].set_xlabel("")
        #     ax[1].set_xlabel("")
        #     sns.despine()
        #     plt.tight_layout()
        #     plt.savefig("%s/good_parameters.png" % figures_dir, bbox_inches="tight")

        # # Plot good and bad predictions in different colors
        # print("Plotting good and bad predictions", flush=True)
        # good_predictions["Prediction"] = "Good"
        # bad_predictions["Prediction"] = "Bad"
        # combined_predictions = pd.concat([good_predictions, bad_predictions])

        # with sns.plotting_context("paper", rc=plot_rc_pars):
        #     fig, ax = plt.subplots(figsize=(3,2))
        #     p = sns.stripplot(data=combined_predictions, x="Data point", y=r"IFN$\beta$", hue="Prediction", palette=sns.cubehelix_palette(2, rot=0.9),
        #                       alpha=0.7, jitter=0.2, ax=ax)
        #     sns.stripplot(data=data_df, x="Data point", y=r"IFN$\beta$", color=new_data_color, alpha=0.7, jitter=0.2, ax=ax)
        #     plt.xticks(rotation=45)
        #     plt.legend(title="Prediction", bbox_to_anchor=(1,1), loc="upper left", frameon=False)
        #     plt.tight_layout()
        #     plt.savefig("%s/good_bad_predictions.png" % figures_dir, bbox_inches="tight")
        #     plt.close()

        # # Filter parameter sets for predictions where CpG p50ko - CpG WT > 0.05 and CpG_WT < 0.1
        # predictions_df = pd.read_csv("%s/best_optimized_predictions_combined.csv" % dir)
        # predictions_df["par_set"] = predictions_df.groupby(["Dataset"]).cumcount()

        # parameters_df = pd.read_csv("%s/best_optimized_pars_combined.csv" % dir)
        # parameters_df["par_set"] = parameters_df.groupby(["Dataset"]).cumcount()

        # # print(predictions_df)
        # predictions_df["CpG_p50ko_minus_CpG_WT"] = predictions_df["CpG_p50KO"] - predictions_df["CpG_WT"]
        # cpg_rows_to_keep = predictions_df.loc[(predictions_df["CpG_p50ko_minus_CpG_WT"] > 0.15) & (predictions_df["CpG_WT"] < 0.1), ["Dataset", "par_set"]]
        # par_sets_omitted = pd.DataFrame(columns=["Dataset", "par_set"])

        # for dset in datasets:
        #     par_sets_kept = cpg_rows_to_keep.loc[cpg_rows_to_keep["Dataset"] == dset, "par_set"].values
        #     num_par_sets = len(predictions_df.loc[predictions_df["Dataset"] == dset])
        #     par_sets_left = [i for i in range(num_par_sets) if i not in par_sets_kept]
        #     if len(par_sets_kept) < num_par_sets:
        #         # print([dset]*len(par_sets_left))
        #         par_sets_omitted = pd.concat([par_sets_omitted, pd.DataFrame({"Dataset": [dset.replace("synthetic_", "")]*len(par_sets_left), "par_set": par_sets_left})])

        # par_sets_omitted["Dataset"] = pd.Categorical(par_sets_omitted["Dataset"], categories= np.char.replace(datasets, "synthetic_", ""), ordered=True)

        # with sns.plotting_context("paper", rc=plot_rc_pars):
        #     fig, ax = plt.subplots(figsize=(10,2))
        #     sns.stripplot(data=par_sets_omitted, x="Dataset", y="par_set", ax=ax, jitter=False)
        #     sns.despine()
        #     plt.tight_layout()
        #     plt.title("Omitted parameter sets")
        #     plt.savefig("%s/par_sets_omitted_CpG_filter.png" % figures_dir, bbox_inches="tight")


        # predictions_df_filtered = pd.merge(predictions_df, cpg_rows_to_keep, on=["Dataset", "par_set"], how="inner")
        # parameters_df_filtered = pd.merge(parameters_df, cpg_rows_to_keep, on=["Dataset", "par_set"], how="inner")
        
        # # print(predictions_df_filtered.columns)
        # # print(parameters_df_filtered.columns)
        
        # # raise ValueError("Stop here")


        # # Filter predictions and parameters for LPS IRF3/7ko < 0.1, pIC IRF 3/5/7ko < 0.1, LPS WT- LPS NFkBko > 0.05, pIC IRF 3/7 ko - pIC IRF 3/5/7 ko > 0.05
        # predictions_df_filtered["LPS_wt_minus_LPS_nfkbko"] = predictions_df_filtered["LPS_WT"] - predictions_df_filtered["LPS_relacrelKO"]
        # predictions_df_filtered["pIC_irf3irf7KO_minus_pIC_irf3irf5irf7KO"] = predictions_df_filtered["polyIC_irf3irf7KO"] - predictions_df_filtered["polyIC_irf3irf5irf7KO"]
        # irf_rows_to_keep = predictions_df_filtered.loc[(predictions_df_filtered["LPS_irf3irf7KO"] < 0.1) & 
        #                                                (predictions_df_filtered["polyIC_irf3irf5irf7KO"] < 0.05) & 
        #                                                (predictions_df_filtered["LPS_wt_minus_LPS_nfkbko"] > 0.2) &
        #                                                (predictions_df_filtered["pIC_irf3irf7KO_minus_pIC_irf3irf5irf7KO"] > 0.1), ["Dataset", "par_set"]]
        # predictions_df_filtered = predictions_df_filtered.drop(columns=["LPS_wt_minus_LPS_nfkbko", "pIC_irf3irf7KO_minus_pIC_irf3irf5irf7KO", "CpG_p50ko_minus_CpG_WT"])
        # # print(irf_rows_to_keep)
        # # par_sets_omitted = pd.DataFrame(columns=["Dataset", "par_set"])

        # for dset in datasets:
        #     par_sets_kept = irf_rows_to_keep.loc[irf_rows_to_keep["Dataset"] == dset, "par_set"].values
        #     num_par_sets = len(predictions_df.loc[predictions_df["Dataset"] == dset])
        #     par_sets_left = [i for i in range(num_par_sets) if i not in par_sets_kept]
        #     if len(par_sets_kept) < num_par_sets:
        #         par_sets_omitted = pd.concat([par_sets_omitted, pd.DataFrame({"Dataset": [dset]*len(par_sets_left), "par_set": par_sets_left})])

        # # print(par_sets_omitted)
        # # Remove synthetic_ from dataset names
        # par_sets_omitted["Dataset"] = pd.Categorical(par_sets_omitted["Dataset"].str.replace("synthetic_", ""), categories=np.char.replace(datasets, "synthetic_", ""), ordered=True)
        # # print(par_sets_omitted)
        
        # # raise ValueError("Stop here")

        # with sns.plotting_context("paper", rc=plot_rc_pars):
        #     fig, ax = plt.subplots(figsize=(10,2))
        #     sns.stripplot(data=par_sets_omitted, x="Dataset", y="par_set", ax=ax, jitter=False)
        #     sns.despine()
        #     plt.tight_layout()
        #     plt.title("Omitted parameter sets")
        #     plt.savefig("%s/par_sets_omitted_CpG_LPS_pIC_filter.png" % figures_dir, bbox_inches="tight")

        # predictions_df_filtered = pd.merge(predictions_df_filtered, irf_rows_to_keep, on=["Dataset", "par_set"], how="inner")
        # parameters_df_filtered = pd.merge(parameters_df_filtered, irf_rows_to_keep, on=["Dataset", "par_set"], how="inner")


        # # Plot filtered predictions
        # predictions_df_filtered = predictions_df_filtered.melt(id_vars=["Dataset", "par_set"], var_name="Data point", value_name=r"IFN$\beta$")
        # # print(predictions_df_filtered["Data point"].unique())

        # predictions_df_filtered = make_predictions_data_frame(predictions_df_filtered)
        # # print(predictions_df_filtered["Data point"].unique())
        
        # # raise ValueError("Stop here")

        # predictions_df_filtered["Dataset"] = pd.Categorical(predictions_df_filtered["Dataset"], categories= datasets, ordered=True)

        # data_df = pd.read_csv("../data/p50_training_data.csv")
        # data_df["Data point"] = data_df["Stimulus"] + "_" + data_df["Genotype"]
        # data_df = data_df.rename(columns={"IFNb": r"IFN$\beta$"})
        # data_df = make_predictions_data_frame(data_df)
        # new_data_color = "#A7535E"

        # with sns.plotting_context("paper", rc=plot_rc_pars):
        #     # Scatterplot, x = datapoint name, y= IFNb value, color = dataset
        #     fig, ax = plt.subplots(figsize=(3,2))
        #     p = sns.stripplot(data=predictions_df_filtered, x="Data point", y=r"IFN$\beta$", hue="Dataset", palette=sns.color_palette(dataset_cmap_pars, len(datasets)),
        #                       alpha=0.7, jitter=0.2, ax=ax, legend=False)
        #     sns.stripplot(data=data_df, x="Data point", y=r"IFN$\beta$", color=new_data_color, alpha=0.7, jitter=0.2, ax=ax)
        #     plt.xticks(rotation=45)
        #     # plt.legend(title="Dataset", bbox_to_anchor=(1,1), loc="upper left")
        #     plt.tight_layout()
        #     plt.savefig("%s/predictions_combined_CpG_LPS_pIC_filter.png" % figures_dir, bbox_inches="tight")
        #     plt.close()

        # # Plot filtered parameters
        # df_all_t_pars, df_all_k_pars, num_t_pars, num_k_pars = make_parameters_data_frame(parameters_df_filtered)

        # colors = sns.color_palette(dataset_cmap_pars, len(datasets))

        # with sns.plotting_context("paper",rc=plot_rc_pars):
        #     width = 2.8
        #     height = 1
        #     fig, ax = plt.subplots(1,2, figsize=(width, height), 
        #                         gridspec_kw={"width_ratios":[num_t_pars, num_k_pars]})
            
        #     legend_handles = []
        #     for i, dset in enumerate(datasets):
        #         # Filter data for the current model
        #         df_model = df_all_t_pars[df_all_t_pars["Dataset"] == dset]
        #         l = sns.stripplot(data=df_model, x="Parameter", y="Value", color=colors[i], ax=ax[0], zorder = i, alpha=0.2)

        #         legend_handles.append(l)

        #         df_model = df_all_k_pars[df_all_k_pars["Dataset"] == dset]
        #         sns.stripplot(data=df_model, x="Parameter", y="Value", color=colors[i], ax=ax[1], zorder = i, alpha=0.2)
            
        #     ax[1].set_yscale("log")
        #     ax[1].set_ylabel("")

        #     ax[0].set_ylabel("Parameter Value")
        #     ax[0].set_xlabel("")
        #     ax[1].set_xlabel("")

        #     sns.despine()
        #     plt.tight_layout()
        #     plt.savefig("%s/optimized_parameters_combined_CpG_LPS_pIC_filter.png" % figures_dir, bbox_inches="tight")
        #     plt.close()

        print("Finished saving plots to %s" % figures_dir, flush=True)

        print("###############################################\n\n###############################################\n")

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
    parser.add_argument("-n","--num_threads", type=int, default=60)
    parser.add_argument("-s","--num_samples_power", type=int, default=6)
    args = parser.parse_args()

    start_start = time.time()

    print("###############################################\n")
    print("Starting at %s\n" % time.ctime(), flush=True)

    # Settings    
    num_threads = args.num_threads
    model = "p50_dist_syn"
    h1, h2 = args.h1, args.h2
    h3 = 1
    h_val = "%d_%d_%d" % (h1, h2, h3)
    print("h_I1: %d, h_I2: %d, h3_N: %d" % (h1, h2, h3), flush=True)
    print("Error value: %.1f" % args.error_val, flush=True)
    c_par= args.c_parameter
    num_to_keep = 100
    num_samples = 10**args.num_samples_power
    restrict_t = False

    # Model details
    num_t_pars = 5
    num_k_pars = 4
    # num_c_pars = 1
    # num_h_pars = 2

    # Directories
    insert_dir = "" if restrict_t else "no_restrict/"
    if h_val != "3_1_1":
        insert_dir = insert_dir + "h_%s/" % h_val
    if c_par:
        insert_dir = insert_dir + "c_scan/"

    figures_dir = "parameter_scan_dist_syn/%sfigures_%.1f/" % (insert_dir, args.error_val)
    results_dir = "parameter_scan_dist_syn/%sresults_%.1f/" % (insert_dir, args.error_val)
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
        datasets = datasets[1:10]
        print("IN TEST MODE: Using only %d datasets" % len(datasets), flush=True)
        num_samples = 10**4
        print("Using only %d samples" % num_samples, flush=True)

    one_dataset = synthetic_data.loc[synthetic_data["Dataset"] == datasets[0]]
    datapoint_names = one_dataset["Stimulus"] + "_" + one_dataset["Genotype"]

    if c_par:
        # result_par_names = [r"$t_I$",r"$t_N$",r"$t_{I\cdotIg}$", r"$t_{I\cdotN}$", "k1", "k2", "kn", "c"]
        result_par_names = ["t_1","t_3","t_4","t_5","t_6", "k1", "k2", "kn", "kp", "c"]
    else:
        # result_par_names = [r"$t_I$",r"$t_N$",r"$t_{I\cdotIg}$", r"$t_{I\cdotN}$", "k1", "k2", "kn"]
        result_par_names = ["t_1","t_3","t_4","t_5","t_6", "k1", "k2", "kn", "kp"]

    # Optimize
    if args.param_scan:
        print("###############################################\n")
        start = time.time()
        grid = calculate_grid(seed=0, num_threads=num_threads, num_t_pars=num_t_pars, num_k_pars=num_k_pars, c_par=c_par, num_samples=num_samples, restrict_t=restrict_t)

        training_data = synthetic_data.loc[synthetic_data["Dataset"].isin(datasets)]
        
        datasets = parameter_scan(training_data, grid, results_dir, h1=h1, h2=h2, num_threads=num_threads, num_t_pars=num_t_pars, num_k_pars=num_k_pars, c_par=c_par,num_to_keep=num_to_keep)
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
            
            final_pars, ifnb_predicted, rmsd = optimize_model(N, I, P, beta, initial_pars, [h1, h2], c=c_par, num_threads=num_threads, 
                                                              num_t_pars=num_t_pars, num_k_pars=num_k_pars, restrict_t=restrict_t)
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
        # Load and combine optimized parameters, predictions
        make_all_plots(synthetic_data, results_dir, figures_dir, args.error_val)

          

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
