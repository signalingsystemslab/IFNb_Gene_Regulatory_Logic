# Optimize p50 model with grid search for each synthetic dataset
from p50_model import *
import matplotlib.pyplot as plt
import pandas as pd
import os
import scipy.optimize as opt
from scipy.stats import norm, lognorm, expon, gamma, beta
import time
from multiprocessing import Pool
plt.style.use("~/IFN_paper/src/theme_bw.mplstyle")

figures_dir = "./opt_syn_datasets/figures/"
results_dir = "./opt_syn_datasets/results/"
os.makedirs(results_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

def p50_objective(pars, *args):
    N, I, P, beta, model_name, strategy = args
    t_pars = pars[0:6]
    K = 1
    C = 1
    num_pts = len(N)
    if model_name == "B2":
        K = pars[6]
    elif model_name == "B3":
        C = pars[6]
    elif model_name == "B4":
        K = pars[6]
        C = pars[7]

    f_list = [get_f(t_pars, K, C, N[i], I[i], P[i], model_name) for i in range(num_pts)]
    # # Normalize to highest value
    if np.max(f_list) != 0:
        f_list = f_list / np.max(f_list)
    residuals = np.array(f_list) - beta
    if strategy != "rmsd":
        return residuals
    else:
        rmsd = np.sqrt(np.mean(residuals**2))
        return rmsd

def optimize_model(N, I, P, beta, model_name, num_threads=40):
    print("###############################################\n")
    print("Optimizing p50 model %s globally" % model_name)
    start = time.time()
    print("Starting brute force optimization at %s" % time.ctime())
    trgs = [slice(0, 1, 0.1) for i in range(6)]
    par_rgs = []
    if model_name == "B2":
        par_rgs.append(slice(0, 2, 0.1))
    elif model_name == "B3":
        par_rgs.append((10**-2, 10**2))
    elif model_name == "B4":
        par_rgs.append(slice(0, 2, 0.1))
        par_rgs.append((10**-2, 10**2))
    rgs = tuple(trgs+par_rgs)

    res = opt.brute(p50_objective, rgs, args=(N, I, P, beta, model_name, "rmsd"), Ns=10, full_output=True, finish=None,
                    workers=num_threads)
    end = time.time()
    params = res[0]
    rmsd = res[1]
    grid = res[2]
    jout = res[3]
    print("Finished optimization at %s" % time.ctime())
    t = end - start
    if t < 60:
        print("Time elapsed: %.2f seconds" % t)
    elif t < 3600:
        print("Time elapsed: %.2f minutes" % (t/60))
    else:
        print("Time elapsed: %.2f hours" % (t/3600))
    return params, rmsd, grid, jout

def optimize_model_local(N, I, P, beta, model_name, pars):
    print("###############################################\n")
    print("Optimizing model %s locally" % model_name)
    start = time.time()
    if True:
        method = "trf"
    else:
        method = "lm"
    m_name = {"lm": "Levenberg-Marquardt", "trf": "Trust Region Reflective"}
    print("Using %s method to locally optimize model %s" % (m_name[method], model_name))
    print("Starting local optimization at ", time.ctime())
    upper = np.array([1 for i in range(6)])
    lower = np.array([0 for i in range(6)])
    if model_name == "B2":
        upper = np.append(upper, 2)
        lower = np.append(lower, 0)
    elif model_name == "B3":
        upper = np.append(upper, 10**2)
        lower = np.append(lower, 10**-2)
    elif model_name == "B4":
        upper = np.append(upper, [2, 10**2])
        lower = np.append(lower, [0, 10**-2])

    print("Upper bounds: ", upper)
    print("Lower bounds: ", lower)
    print("Initial parameters: ", pars)
    res = opt.least_squares(p50_objective, pars, bounds = (lower, upper),
                            args=(N, I, P, beta, model_name, "residuals"), method=method, loss = "linear")
    end = time.time()
    print("Optimized parameters:\n", res.x)
    t = end - start
    if t < 60:
        print("Time elapsed: %.2f seconds" % t)
    elif t < 3600:
        print("Time elapsed: %.2f minutes" % (t/60))
    else:
        print("Time elapsed: %.2f hours" % (t/3600))
    pars = res.x
    residuals = res.fun
    cost = res.cost
    return pars, cost, residuals

def main():
    print("###############################################\n")
    print("Optimizing p50 model with grid optimization at %s" % time.ctime())
    training_data = pd.read_csv("../data/p50_training_data_plus_synthetic.csv")
    datasets = training_data["Dataset"].unique()
    num_datasets = len(datasets)
    dataset = training_data.loc[training_data["Dataset"] == datasets[0]]
    num_pts = dataset.shape[0]
    num_threads = 30
    model_par_numbers = {"B1": 6, "B2": 7, "B3": 7, "B4": 8}
    model = "B1"
    num_pars = model_par_numbers[model]
    print("The total number of data points is %d" % num_pts)


    pars_all = np.zeros((num_datasets, num_pars))
    residuals_all = np.zeros((num_datasets, num_pts))
    cost_all = np.zeros(num_datasets)

    pars_global_all = np.zeros((num_datasets, num_pars))
    aic_global_all = np.zeros(num_datasets)
    rmsd_global_all = np.zeros(num_datasets)

    print("\n\n###############################################")
    print("OPTIMIZING MODEL %s" % model)
    print("###############################################\n", flush=True)

    # for i in range(num_datasets):
    #     dataset = training_data.loc[training_data["Dataset"] == datasets[i]]
    #     print("###############################################\n")
    #     dataset_name = datasets[i]
    #     print("Globally optimizing model %s for dataset %s" % (model, dataset_name), flush=True)
    #     print(dataset, flush=True)
    #     start = time.time()
    #     print("Starting at %s" % time.ctime(), flush=True)
    #     N = dataset["NFkB"].values
    #     I = dataset["IRF"].values
    #     P = dataset["p50"].values
    #     beta = dataset["IFNb"].values
        
    #     # Grid search
    #     pars, min_rmsd, grid, jout = optimize_model(N, I, P, beta, model, num_threads=num_threads)
    #     pars_global_all[i] = pars
    #     rmsd_global_all[i] = min_rmsd
    #     np.save("%s/p50_all_datasets_grid_%s.npy" % (results_dir, dataset_name), grid)
    #     np.save("%s/p50_all_datasets_jout_%s.npy" % (results_dir, dataset_name), jout)
        
    #     # Calculate AIC from rmsd and number of parameters
    #     min_aic = num_pts * np.log(min_rmsd) + (2 * num_pars)
    #     aic_global_all[i] = min_aic
        
    #     end = time.time()
    #     # print("Finished at %s, after %.2f minutes" % (time.ctime(), (end-start)/60), flush=True)
    #     print("Minimum rmsd: %.4f, minimum aic: %.4f" % (min_rmsd, min_aic), flush=True)

    #     # Local optimization
    #     print("###############################################\n")
    #     print("Locally optimizing model %s for dataset %s" % (model, dataset_name), flush=True)

    #     pars, cost, residuals = optimize_model_local(N, I, P, beta, model, pars)
    #     pars_all[i] = pars
    #     residuals_all[i] = residuals
    #     cost_all[i] = cost
        

    # # Save results
    # np.savetxt("%s/p50_all_datasets_cost_local.csv" % results_dir, cost_all, delimiter=",")
    # np.savetxt("%s/p50_all_datasets_residuals_local.csv" % results_dir, residuals_all, delimiter=",")
    # np.savetxt("%s/p50_all_datasets_pars_local.csv" % results_dir, pars_all, delimiter=",")
    # np.savetxt("%s/p50_all_datasets_aic_global.csv" % results_dir, aic_global_all, delimiter=",")
    # np.savetxt("%s/p50_all_datasets_rmsd_global.csv" % results_dir, rmsd_global_all, delimiter=",")
    # np.savetxt("%s/p50_all_datasets_pars_global.csv" % results_dir, pars_global_all, delimiter=",")

    # Load results
    pars_all = np.loadtxt("%s/p50_all_datasets_pars_local.csv" % results_dir, delimiter=",")
    residuals = np.loadtxt("%s/p50_all_datasets_residuals_local.csv" % results_dir, delimiter=",")

    # Calculate rmsd and aic for each dataset
    rmsd = np.sqrt(np.mean(residuals**2, axis=1))
    np.savetxt("%s/p50_all_datasets_rmsd_local.csv" % results_dir, rmsd, delimiter=",")
    aic = num_pts * np.log(rmsd) + (2 * num_pars)
    np.savetxt("%s/p50_all_datasets_aic_local.csv" % results_dir, aic, delimiter=",")

    print("\n\n###############################################")
    print("PLOTTING RESULTS")
    print("###############################################\n", flush=True)

    # Plot parameters
    def jitter_dots(dots, jitter=0.3):
        offsets = dots.get_offsets()
        jittered_offsets = offsets
        # only jitter in the x-direction
        jittered_offsets[:, 0] += np.random.uniform(-jitter, jitter, size=offsets.shape[0])
        dots.set_offsets(jittered_offsets)
        return dots

    par_names = [r"IRF_1", r"IRF_2", r"NF$\kappa$B", r"IRF_1 IRF_2", r"IRF_1 NF$\kappa$B", r"IRF_2 NF$\kappa$B"]
    fig, ax = plt.subplots(figsize=(8,6), dpi=300)
    colors = rmsd[1:]
    rmsd_og = rmsd[0]
    for i in range(num_pars):
        # print(np.full(num_datasets-1, i))
        # print(colors)
        # print("##################")
        dots = ax.scatter(np.full(num_datasets-1, i), pars_all[1:,i], c=colors, cmap="viridis", s=50, alpha=0.5)
        dots = jitter_dots(dots)
        ax.scatter(i, pars_all[0,i], c=rmsd_og, cmap="viridis", s=50, alpha=1, edgecolors="red", linewidths=1, vmin=np.min(colors), vmax=np.max(colors))
    ax.set_xticks(np.arange(num_pars))
    ax.set_xticklabels(par_names)
    ax.set_xlim([-0.5, num_pars-0.5])
    ax.set_ylabel("Optimized parameter (t) value")
    ax.set_xlabel("Parameter")
    ax.set_title("Optimized parameter values for model %s" % model)
    plt.colorbar(dots, label="RMSD")
    plt.savefig("%s/p50_all_datasets_pars_local.png" % figures_dir)


    # Plot parameters all black
    fig, ax = plt.subplots(figsize=(8,6), dpi=300)
    for i in range(num_pars):
        dots = ax.scatter(np.full(num_datasets-1, i), pars_all[1:,i], c="k", s=50, alpha=0.5)
        dots = jitter_dots(dots)
        ax.scatter(i, pars_all[0,i], c="r", s=50, alpha=0.5)
    ax.set_xticks(np.arange(num_pars))
    ax.set_xticklabels(par_names)
    ax.set_xlim([-0.5, num_pars-0.5])
    ax.set_ylabel("Optimized parameter (t) value")
    ax.set_xlabel("Parameter")
    ax.set_title("Optimized parameter values for model %s" % model)
    plt.savefig("%s/p50_all_datasets_pars_local_black.png" % figures_dir)
    
    # Plot contributions for each parameter
    num_states = 12
    parameter_contributions_LPS = np.zeros((num_datasets, num_states))
    # LPS
    N = 1.0
    I = 0.25
    P = 1.0
    for i in range(num_datasets):
        parameter_contributions_LPS[i,:], state_names = get_f_contribution(pars_all[i,:], 1, 1, N, I, P, model)
        parameter_contributions_LPS[i,:] = parameter_contributions_LPS[i,:] / np.sum(parameter_contributions_LPS[i,:])
    fig, ax = plt.subplots(figsize=(8,6), dpi=300)
    for i in range(num_states):
        dots = ax.scatter(np.full(num_datasets, i), parameter_contributions_LPS[:,i], c="k", s=50, alpha=0.5)
        dots = jitter_dots(dots)
    ax.set_xticks(np.arange(num_states))
    ax.set_xticklabels(state_names, rotation=90)
    ax.set_xlim([-0.5, num_states-0.5])
    ax.set_ylabel(r"Contribution to IFN$\beta$")
    ax.set_xlabel("Parameter")
    ax.set_title("Contribution of each TF (fraction) for LPS stimulation")
    plt.savefig("%s/p50_all_datasets_contributions_LPS_WT.png" % figures_dir)
    ylim = ax.get_ylim()

    # CpG
    N = 0.75
    I = 0.05
    parameter_contributions_CpG = np.zeros((num_datasets, num_states))
    for i in range(num_datasets):
        parameter_contributions_CpG[i,:], state_names = get_f_contribution(pars_all[i,:], 1, 1, N, I, P, model)
        parameter_contributions_CpG[i,:] = parameter_contributions_CpG[i,:] / np.sum(parameter_contributions_CpG[i,:])
    fig, ax = plt.subplots(figsize=(8,6), dpi=300)
    for i in range(num_states):
        dots = ax.scatter(np.full(num_datasets, i), parameter_contributions_CpG[:,i], c="k", s=50, alpha=0.5)
        dots = jitter_dots(dots)
    ax.set_xticks(np.arange(num_states))
    ax.set_xticklabels(state_names, rotation=90)
    ax.set_xlim([-0.5, num_states-0.5])
    ax.set_ylim(ylim)
    ax.set_ylabel(r"Contribution to IFN$\beta$")
    ax.set_xlabel("Parameter")
    ax.set_title("Contribution of each TF (fraction) for CpG stimulation")
    plt.savefig("%s/p50_all_datasets_contributions_CpG_WT.png" % figures_dir)
    
    # both LPS and CpG
    # get first viridis color
    color1 = plt.cm.viridis(0)
    color2 = plt.cm.viridis(0.5)
    fig, ax = plt.subplots(figsize=(8,6), dpi=300)
    for i in range(num_states):
        dots = ax.scatter(np.full(num_datasets, i) - 0.2, parameter_contributions_LPS[:,i], color=color1, s=30, alpha=0.5, label="LPS" if i==0 else None)
        dots = jitter_dots(dots, jitter=0.15)
        dots = ax.scatter(np.full(num_datasets, i) + 0.2, parameter_contributions_CpG[:,i], color=color2, s=30, alpha=0.5, label="CpG" if i==0 else None)
        dots = jitter_dots(dots, jitter=0.15)
    ax.set_xticks(np.arange(num_states))
    ax.set_xticklabels(state_names, rotation=90)
    ax.set_xlim([-0.7, num_states-0.3])
    ax.set_ylim(ylim)
    ax.set_ylabel(r"Contribution to IFN$\beta$")
    ax.set_xlabel("Parameter")
    ax.set_title("Contribution of each TF (fraction) for LPS and CpG stimulation")
    fig.legend(bbox_to_anchor=(1.1,0.5))
    plt.savefig("%s/p50_all_datasets_contributions_LPS_CpG_WT.png" % figures_dir)


    # # Plot f-value for each parameter set in WT
    # N = training_data.loc[training_data["Genotype"] == "WT"]["NFkB"].values
    # I = training_data.loc[training_data["Genotype"] == "WT"]["IRF"].values
    # P = 1.0
    # f_values_WT = np.zeros((num_datasets, num_pts))
    # max_f = np.zeros(num_datasets)
    # for i in range(num_datasets):
    #     f_values_WT[i,:] = [get_f(pars_all[i,:], 1, 1, N[j], I[j], P, model) for j in range(num_pts)]
    #     max_f[i] = np.max(f_values_WT[i,:])
    #     f_values_WT[i,:] = f_values_WT[i,:] / max_f[i]

    # cmap = plt.cm.viridis
    # fig, ax = plt.subplots(figsize=(8,6), dpi=300)
    # for i in range(num_datasets):
    #     ax.scatter(N, I, c=f_values_WT[i,:], cmap=cmap, s=50, alpha=0.5, vmin=0, vmax=1)
    # ax.set_xlabel(r"$NF\kappa B$")
    # ax.set_ylabel(r"$IRF$")
    # ax.set_title("Model %s f-values for WT dataset" % model)
    # plt.colorbar(label="f-value")
    # plt.savefig("%s/p50_all_datasets_f_values_WT.png" % figures_dir)

    # # Plot top 10 parameters with lowest rmsd
    # fig, ax = plt.subplots(figsize=(8,6), dpi=300)
    # colors = rmsd[1:]
    # top_10 = np.argsort(colors)[0:10]
    # for i in range(num_pars):
    #     dots = ax.scatter(np.full(10, i), pars_all[top_10,i], c=colors[top_10], cmap="viridis", s=50, alpha=0.5,
    #                         vmin=np.min(colors), vmax=np.max(colors))
    #     dots = jitter_dots(dots)
    #     ax.scatter(i, pars_all[0,i], c="k", s=50, alpha=1)
    # ax.set_xticks(np.arange(num_pars))
    # ax.set_xticklabels(par_names)
    # ax.set_xlim([-0.5, num_pars-0.5])
    # ax.set_ylabel("Optimized parameter (t) value")
    # ax.set_xlabel("Parameter")
    # ax.set_title("Optimized parameter values for model %s" % model)
    # plt.colorbar(dots, label="RMSD")
    # plt.savefig("%s/p50_all_datasets_pars_local_top_10.png" % figures_dir)

    # # Plot rmsd for each dataset
    # fig, ax = plt.subplots(figsize=(8,6), dpi=300)
    # ax.scatter(np.arange(num_datasets), rmsd, c="k", s=50)
    # ax.set_ylabel("RMSD")
    # ax.set_xlabel("Dataset")
    # ax.set_xticks(np.arange(num_datasets))
    # ax.set_xticklabels(datasets, rotation=90)
    # ax.set_title("RMSD for each dataset")
    # plt.savefig("%s/p50_all_datasets_rmsd_local.png" % figures_dir)

    # # Plot each parameter distribution as density plot
    # fig, ax = plt.subplots(2,3, figsize=(12,8), dpi=300)
    # ax = ax.flatten()
    # for i in range(num_pars):
    #     ax[i].hist(pars_all[:,i], bins=20, density=True, color="k", alpha=0.5)
    #     ax[i].set_title(par_names[i])
    #     ax[i].set_xlabel("Optimized parameter (t) value")
    #     ax[i].set_ylabel("Density")
    # plt.tight_layout()
    # plt.suptitle("Parameter distributions for model %s" % model)
    # plt.savefig("%s/p50_all_datasets_pars_local_hist.png" % figures_dir) 

    # # Plot joint distribution of parameters
    # fig, ax = plt.subplots(num_pars, num_pars, figsize=(12,12), dpi=300)
    # for i in range(num_pars):
    #     for j in range(num_pars):
    #         ax[i,j].scatter(pars_all[:,i], pars_all[:,j], c="k", s=50, alpha=0.5)
    #         ax[i,j].set_xlabel(par_names[i])
    #         ax[i,j].set_ylabel(par_names[j])
    # plt.tight_layout()
    # plt.savefig("%s/p50_all_datasets_pars_local_joint.png" % figures_dir)

    # # Make contour plots from top 10 parameters
    # top_pars = pars_all[top_10, :]
    # for i in range(10):
    #     t_pars = top_pars[i]
    #     K = 1
    #     C = 1
    #     N = np.linspace(0, 1, 50)
    #     I = np.linspace(0, 1, 50)
    #     P = np.ones(50)
    #     f_values = np.zeros((len(N), len(I)))
    #     for j in range(len(N)):
    #         for k in range(len(I)):
    #             f_values[j,k] = get_f(t_pars, K, C, N[j], I[k], P[k], model)
    #     plot_contour(f_values, model, I, N, figures_dir, "top_10_%d" % i, condition="top 10 #%d" % i, normalize=True)



if __name__ == "__main__":
    main()