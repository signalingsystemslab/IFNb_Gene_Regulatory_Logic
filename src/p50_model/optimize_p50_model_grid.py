# Optimize p50 model with grid search
from p50_model import *
import matplotlib.pyplot as plt
import pandas as pd
import os
import scipy.optimize as opt
import time
from multiprocessing import Pool
plt.style.use("~/IFN_paper/src/theme_bw.mplstyle")

figures_dir = "grid_opt_B1/figures/"
results_dir = "grid_opt_B1/results/"
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
        # pars = np.hstack([pars[6], pars[0:6]])
    elif model_name == "B3":
        C = pars[6]
        # pars = np.hstack([pars[6], pars[0:6]])
    elif model_name == "B4":
        K = pars[6]
        C = pars[7]
        # pars = np.hstack([pars[6:8], pars[0:6]])

    f_list = [get_f(t_pars, K, C, N[i], I[i], P[i], model_name, True) for i in range(num_pts)]
    residuals = np.array(f_list) - beta
    if strategy == "residuals":
        return residuals
    elif strategy == "rmsd":
        # Enforce fitting to CpG points
        if np.abs(residuals[2]) > 0.1 or np.abs(residuals[3]) > 0.1:
            rmsd = 100
        else:
            rmsd = np.sqrt(np.mean(residuals**2))
        return rmsd
    elif strategy == "cost":
        if np.abs(residuals[2]) > 0.1 or np.abs(residuals[3]) > 0.1:
            cost = 100
        else:
            cost = 0.5 * np.sum(residuals**2)
        return cost
    else:
        raise ValueError("Strategy %s not implemented" % strategy)

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

def plot_optimization(grid, jout, model_name, measure="RMSD"):
    #  Plot each individual param against corresponding RMSD
    print("Plotting optimization results")
    fig, ax = plt.subplots(2,3, figsize=(10,6))
    ax = ax.flatten()
    for i in range(6):
        par_vals = grid[i].flatten()
        rmsd_vals = jout.flatten()
        ax[i].plot(par_vals, rmsd_vals, 'o')
        ax[i].set_xlabel("t%d" % (i+1))
        ax[i].set_ylabel("%s" % measure)
    fig.suptitle("%s optimization results for model %s" % (measure, model_name))
    plt.tight_layout(pad=2)
    plt.savefig("%s/p50_grid_%s_%s.png" % (figures_dir, measure, model_name), bbox_inches="tight") 


def optimize_model_local(N, I, P, beta, model_name, pars, full_output=True):
    if full_output:
        print("###############################################\n")
        print("Optimizing model %s locally" % model_name)
    start = time.time()
    # if True:
    #     method = "trf"
    # else:
    #     method = "lm"
    # m_name = {"lm": "Levenberg-Marquardt", "trf": "Trust Region Reflective"}
    method = "trust-constr"
    if full_output:
        print("Using %s method to locally optimize model %s" % (method, model_name))
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

    if full_output:
        print("Upper bounds: ", upper)
        print("Lower bounds: ", lower)
        print("Initial parameters: ", pars)

    bnds = [(lower[i], upper[i]) for i in range(len(upper))]
    bnds = tuple(bnds)

    res = opt.minimize(p50_objective, pars, args=(N, I, P, beta, model_name, "cost"), method=method, bounds = bnds)

    # res = opt.least_squares(p50_objective, pars, bounds = (lower, upper),
    #                         args=(N, I, P, beta, model_name, "residuals"), method=method, loss = "linear")
    end = time.time()
    if full_output:
        print("Optimized parameters:\n", res.x)
        t = end - start
        if t < 60:
            print("Time elapsed: %.2f seconds" % t)
        elif t < 3600:
            print("Time elapsed: %.2f minutes" % (t/60))
        else:
            print("Time elapsed: %.2f hours" % (t/3600))
    pars = res.x
    cost = res.fun
    residuals = p50_objective(pars, N, I, P, beta, model_name, "residuals")
    return pars, cost, residuals

# def save_results(results_df, model, pars, other):
#     if model == "B1":
#         res = np.hstack([pars, np.nan,np.nan,other])
#     elif model == "B2":
#         res = np.hstack([pars, np.nan,other])
#     elif model == "B3":
#         res = np.hstack([pars[0:6], np.nan, pars[6], other])
#     elif model == "B4":
#         res = np.hstack([pars, other])

#     print("Results to add: %s\n Size of results df %s" % (res, results_df.shape))

#     results_df.loc[model] = res
#     return results_df

def four_model_par_plot(npars, y, names, x_label, y_label, title, figname):
    # Plot parameters (on one axis), states, parameter products, etc. for each model in different color
    fig, ax = plt.subplots()
    ax.set_prop_cycle("color", plt.cm.viridis(np.linspace(0,1,4)))
    i=0
    for model in ["B1", "B2", "B3", "B4"]:
        x= np.arange(i, i+npars)
        ax.plot(x, y[model], label=model, marker="o", linestyle="none")
        i+=0.1
    ax.set_xticks(np.arange(0.15,npars+0.15), names, rotation=45)
    fig.text(0.5, 0.04, x_label, ha='center', size=14)
    fig.text(0.04, 0.5, y_label, va='center', rotation='vertical', size=14)
    plt.grid(False)
    plt.suptitle(title)
    plt.tight_layout(pad = 4)
    fig.legend(bbox_to_anchor=(1.1,0.5))
    plt.savefig("%s/%s.png" % (figures_dir, figname), bbox_inches="tight")

def main():
    print("###############################################\n")
    print("Optimizing p50 model with grid optimization at %s" % time.ctime())
    training_data = pd.read_csv("../data/p50_training_data.csv")
    print("Using the following training data:\n", training_data)
    print("Starting at %s" % time.ctime())

    N = training_data["NFkB"]
    I = training_data["IRF"]
    P = training_data["p50"]
    beta = training_data["IFNb"]
    model_par_numbers = {"B1": 6, "B2": 7, "B3": 7, "B4": 8}
    model = "B1"
    num_pars = model_par_numbers[model]
    num_pts = len(N)
    num_threads = 40
    len_training = len(N)

    if True:
        print("\n\n###############################################")
        print("OPTIMIZING MODEL %s" % model)
        print("###############################################\n")
        
        # # Grid search
        # pars_global, rmsd_global, grid, jout = optimize_model(N, I, P, beta, model, num_threads=num_threads)

        # # Calculate AIC from rmsd and number of parameters
        # min_aic = num_pts * np.log(rmsd_global) + 2 * model_par_numbers[model]
        # np.save("%s/p50_grid_aic_model_%s.npy" % (results_dir, model), min_aic)
        # print("RMSD: %.4f, AIC: %.4f" % (rmsd_global, min_aic))

        # # Plot all rmsd values
        # plot_optimization(grid, jout, model, measure="RMSD")

        # # Local optimization
        # pars, cost, residuals = optimize_model_local(N, I, P, beta, model, pars_global)
        # rmsd = np.sqrt(np.mean(residuals**2))
        # aic = num_pts * np.log(rmsd) + 2 * model_par_numbers[model]
        # print("RMSD: %.4f, AIC: %.4f" % (rmsd, aic))

        # # Save results
        # np.savetxt("%s/p50_grid_optimization_results_%s.csv" % (results_dir, model), pars, delimiter=",")

        # Define grid of parameters (adapted from opt.brute)
        if model != "B1":
            raise ValueError("Model %s not implemented" % model)
        trgs = [slice(0, 1, 0.1) for i in range(6)]
        ranges = tuple(trgs)
        lrange = list(ranges)
        grid = np.mgrid[lrange]
        inpt_shape = grid.shape
        grid = np.reshape(grid, (inpt_shape[0], np.prod(inpt_shape[1:]))).T

        # # Calculate residual at each point in grid
        # print("Calculating residual at each point in grid")
        # start = time.time()
        # with Pool(num_threads) as p:
        #     results = p.starmap(p50_objective, [(grid[i], N, I, P, beta, model, "residuals") for i in range(len(grid))])

        # end = time.time()
        # t = end - start
        # if t < 60*60:
        #     print("Time elapsed: %.2f minutes" % (t/60))
        # else:
        #     print("Time elapsed: %.2f hours" % (t/3600))

        # residuals = np.array(results)
        # rmsd = np.sqrt(np.mean(residuals**2, axis=1))
        # print("Size of residuals: %s, grid: %s" % (residuals.shape, grid.shape))
        # np.savetxt("%s/p50_grid_residuals_%s.csv" % (results_dir, model), residuals, delimiter=",")
        # np.savetxt("%s/p50_grid_pars_%s.csv" % (results_dir, model), grid, delimiter=",")
        # np.savetxt("%s/p50_grid_rmsd_%s.csv" % (results_dir, model), rmsd, delimiter=",")

        # Locally optimize each parameter set
        print("Optimizing %d parameter sets" % len(grid))
        start = time.time()
        with Pool(num_threads) as p:
            results = p.starmap(optimize_model_local, [(N, I, P, beta, model, grid[i], False) for i in range(len(grid))])
        end = time.time()
        t = end - start
        print("Time elapsed: %.2f minutes" % (t/60))

        pars = np.array([results[i][0] for i in range(len(results))])
        rmsd = np.array([results[i][1] for i in range(len(results))])
        residuals = np.array([results[i][2] for i in range(len(results))])
        print("Size of pars: %s, rmsd: %s, residuals: %s" % (pars.shape, rmsd.shape, residuals.shape))

        np.savetxt("%s/p50_locally_optimized_all_pars_%s.csv" % (results_dir, model), pars, delimiter=",")
        np.savetxt("%s/p50_locally_optimized_all_rmsd_%s.csv" % (results_dir, model), rmsd, delimiter=",")
        np.savetxt("%s/p50_locally_optimized_all_residuals_%s.csv" % (results_dir, model), residuals, delimiter=",")

    # Load results
    print("Loading results from grid", flush=True)
    pars = np.loadtxt("%s/p50_grid_pars_%s.csv" % (results_dir, model), delimiter=",")
    rmsd = np.loadtxt("%s/p50_grid_rmsd_%s.csv" % (results_dir, model), delimiter=",")
    residuals = np.loadtxt("%s/p50_grid_residuals_%s.csv" % (results_dir, model), delimiter=",")

    # # Plot residuals as a heatmap where y-axis is the starting parameter set and x-axis is the data point
    # # To the right, plot heatmap of RMSD values
    # rmsd_sorted = np.sort(rmsd)
    # residuals_sorted = np.abs(residuals[np.argsort(rmsd)])

    # print("Plotting residuals as heatmap", flush=True)
    # fig, ax = plt.subplots(1,2, figsize=(10,6), gridspec_kw={'width_ratios': [1, 0.05]})
    # im = ax[0].imshow(residuals_sorted, cmap="viridis", aspect="auto", interpolation = "nearest")
    # ax[0].set_xlabel("Data point")
    # ax[0].set_ylabel("Parameter set")
    # ax[0].set_title("Residuals, absolute value (grid)")
    # cbar = ax[0].figure.colorbar(im, ax=ax[0])
    # im = ax[1].imshow(np.expand_dims(rmsd_sorted, axis=1), cmap="viridis", aspect="auto", interpolation = "nearest")
    # ax[1].set_xlabel("RMSD")
    # ax[1].set_title("RMSD")
    # cbar = ax[1].figure.colorbar(im, ax=ax[1])
    # plt.tight_layout(pad=2)
    # plt.savefig("%s/p50_grid_residuals_%s.png" % (figures_dir, model), bbox_inches="tight")

    # # Plot residuals as line plot
    # print("Plotting residuals as line plot", flush=True)
    # fig, ax = plt.subplots()
    # for i in range(len(rmsd)):
    #     ax.plot(residuals[i], alpha=0.5, color = plt.cm.viridis(rmsd[i]/np.max(rmsd)))
    # ax.set_xlabel("Data point")
    # ax.set_ylabel("Residual")
    # ax.set_title("Residuals for each parameter set (grid)")
    # # add colorbar
    # sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=0, vmax=np.max(rmsd)))
    # sm._A = []
    # cbar = fig.colorbar(sm)
    # plt.savefig("%s/p50_grid_residuals_lineplot_%s.png" % (figures_dir, model), bbox_inches="tight")
    
    # print("Finished plotting residuals", flush=True)

    # # Plot residuals as jittered scatter plot
    # def jitter_dots(dots, jitter=0.3, y_jitter=False):
    #     offsets = dots.get_offsets()
    #     jittered_offsets = offsets
    #     # only jitter in the x-direction
    #     jittered_offsets[:, 0] += np.random.uniform(-jitter, jitter, size=offsets.shape[0])
    #     if y_jitter:
    #         jittered_offsets[:, 1] += np.random.uniform(-jitter, jitter, size=offsets.shape[0])
    #     dots.set_offsets(jittered_offsets)
    #     return dots
    
    # print("Plotting residuals as jittered scatter plot", flush=True)
    # fig, ax = plt.subplots()
    # for i in range(num_pars):
    #     dots = ax.scatter(np.ones(len(residuals[:,i]))*i, residuals[:,i], alpha=0.3, c = rmsd, cmap="viridis")
    #     jitter_dots(dots)
    # ax.set_xlabel("Parameter")
    # ax.set_ylabel("Residual")
    # ax.set_title("Residuals for each parameter set (grid)")
    # plt.savefig("%s/p50_grid_residuals_scatter_%s.png" % (figures_dir, model), bbox_inches="tight")


    ###############################################
    # Load locally optimized results
    print("Loading locally optimized results", flush=True)
    pars = np.loadtxt("%s/p50_locally_optimized_all_pars_%s.csv" % (results_dir, model), delimiter=",")
    rmsd = np.loadtxt("%s/p50_locally_optimized_all_rmsd_%s.csv" % (results_dir, model), delimiter=",")
    residuals = np.loadtxt("%s/p50_locally_optimized_all_residuals_%s.csv" % (results_dir, model), delimiter=",")

    # Plot residuals as a heatmap where y-axis is the starting parameter set and x-axis is the data point
    # To the right, plot heatmap of RMSD values
    rmsd_sorted = np.sort(rmsd)
    residuals_sorted = np.abs(residuals[np.argsort(rmsd)])

    print("Plotting residuals as heatmap", flush=True)
    fig, ax = plt.subplots(1,2, figsize=(10,6), gridspec_kw={'width_ratios': [1, 0.05]})
    im = ax[0].imshow(residuals_sorted, cmap="viridis", aspect="auto", interpolation = "nearest")
    ax[0].set_xlabel("Data point")
    ax[0].set_ylabel("Parameter set")
    ax[0].set_title("Residuals, absolute value (locally optimized)")
    cbar = ax[0].figure.colorbar(im, ax=ax[0])
    im = ax[1].imshow(np.expand_dims(rmsd_sorted, axis=1), cmap="viridis", aspect="auto", interpolation = "nearest")
    ax[1].set_xlabel("RMSD")
    ax[1].set_title("RMSD")
    cbar = ax[1].figure.colorbar(im, ax=ax[1])
    plt.tight_layout(pad=2)
    plt.savefig("%s/p50_locally_optimized_residuals_%s.png" % (figures_dir, model), bbox_inches="tight")

    # Plot residuals as line plot
    print("Plotting residuals as line plot", flush=True)
    fig, ax = plt.subplots()
    for i in range(len(rmsd)):
        ax.plot(residuals[i], alpha=0.5, color = plt.cm.viridis(rmsd[i]/np.max(rmsd)))
    ax.set_xlabel("Data point")
    ax.set_ylabel("Residual")
    ax.set_title("Residuals for each parameter set (locally optimized)")
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=0, vmax=np.max(rmsd)))
    plt.colorbar(sm)
    plt.savefig("%s/p50_locally_optimized_residuals_lineplot_%s.png" % (figures_dir, model), bbox_inches="tight")
    
    print("Finished plotting residuals", flush=True)

    # Plot residuals as jittered scatter plot
    def jitter_dots(dots, jitter=0.3, y_jitter=False):
        offsets = dots.get_offsets()
        jittered_offsets = offsets
        # only jitter in the x-direction
        jittered_offsets[:, 0] += np.random.uniform(-jitter, jitter, size=offsets.shape[0])
        if y_jitter:
            jittered_offsets[:, 1] += np.random.uniform(-jitter, jitter, size=offsets.shape[0])
        dots.set_offsets(jittered_offsets)
        return dots
    
    print("Plotting residuals as jittered scatter plot", flush=True)
    fig, ax = plt.subplots()
    for i in range(residuals.shape[1]):
        dots = ax.scatter(np.ones(len(residuals[:,i]))*i, residuals[:,i], alpha=0.3, c = rmsd, cmap="viridis")
        jitter_dots(dots)
    ax.set_xlabel("Parameter")
    ax.set_ylabel("Residual")
    ax.set_title("Residuals for each parameter set (locally optimized)")
    plt.savefig("%s/p50_locally_optimized_residuals_scatter_%s.png" % (figures_dir, model), bbox_inches="tight")

    # Plot parameters as a jittered scatter plot
    print("Plotting parameters as jittered scatter plot", flush=True)
    fig, ax = plt.subplots()
    for i in range(num_pars):
        dots = ax.scatter(np.ones(len(pars[:,i]))*i, pars[:,i], alpha=0.3, c = rmsd, cmap="viridis")
        jitter_dots(dots)
    ax.set_xlabel("Parameter")
    ax.set_ylabel("Value")
    ax.set_title("Parameters for each parameter set (locally optimized)")
    plt.colorbar(dots)
    plt.savefig("%s/p50_locally_optimized_parameters_scatter_%s.png" % (figures_dir, model), bbox_inches="tight")

    # # old
    # for model in ["B1", "B2", "B3", "B4"]:
    #     t_pars = results_global.loc[model].values[0:6]
    #     K = results_global.loc[model, "K_i2"]
    #     C = results_global.loc[model, "C"]

    #     f_dict = {}
    #     for genotype in P.keys():
    #         N_vals, I_vals = np.meshgrid(N, I)
    #         # N = N_vals.flatten()
    #         # I = I_vals.flatten()
    #         f_values = np.zeros((len(N), len(I)))
    #         with Pool(40) as p:
    #             for i in range(len(N_vals)):
    #                 f_values[i,:] = p.starmap(get_f, [(t_pars, K, C, N_vals[i,j], I_vals[i,j], P[genotype], model) for j in range(len(N_vals[i,:]))]) 
    #             # f_values = p.starmap(get_f, [(t_pars, K, C, N[i], I[i], P[genotype], model) for i in range(len(N))])
    #         title = "best_fit_grid_%s_%s" % (genotype, model)
    #         plot_contour(f_values, model, I_vals, N_vals, results_dir, title, condition=genotype)
    #         f_dict[genotype] = f_values
    #     # Calculate fold change between WT and p50KO
    #     f_dict["WT"][f_dict["WT"] == 0] = 10**-10
    #     f_fold_change = f_dict["p50KO"]/f_dict["WT"]
    #     title = "best_fit_genotype_fold_change_%s" % model
    #     plot_contour(f_fold_change, model, I_vals, N_vals, results_dir, title, condition="fold change (p50 KO/WT)", normalize=False)

    # #  Plot best fit parameters
    # print("Plotting best fit parameters")
    # fig, ax = plt.subplots(1,2, width_ratios=[1,2/6])
    # for axis in ax:
    #     axis.set_prop_cycle("color", plt.cm.viridis(np.linspace(0,1,4)))
    # i=0
    # for model in ["B1", "B2", "B3", "B4"]:
    #     t = results_global.loc[model].values[0:6]
    #     k = results_global.loc[model, "K_i2"]
    #     c = results_global.loc[model, "C"]
    #     x= np.arange(i, i+6)
    #     x2= np.arange(i, i+2)
    #     ax[0].plot(x, t, label=model, marker="o", linestyle="none")
    #     ax[1].plot(x2, [k, c], marker="o", linestyle="none")
    #     i+=0.1
    # ax[0].set_xticks(np.arange(0.15,6.15), par_names,
    #            rotation=45)
    # ax[1].set_xticks(np.arange(0.05,2.05), ["$K_{i2}$", "C"])

    # fig.text(0.5, 0.04, 'Parameter', ha='center', size=14)
    # fig.text(0.04, 0.5, 'Transcription capability (t)', va='center', rotation='vertical', size=14)
    # plt.grid(False)
    # plt.suptitle("Best fit parameters for p50 model with grid optimization")
    # plt.tight_layout(pad = 4)
    # fig.legend(bbox_to_anchor=(1.1,0.5))
    # plt.savefig("%s/p50_grid_best_fit_parameters.png" % figures_dir, bbox_inches="tight")

    # # Plot product of beta and t parameters
    # print("Plotting product of beta and t parameters")
    # y = {}
    # for model in ["B1", "B2", "B3", "B4"]:
    #     t = results_global.loc[model].values[0:6]
    #     k = results_global.loc[model, "K_i2"]
    #     c = results_global.loc[model, "C"]
    #     product, state_names = get_product(t, k, c, model)
    #     y[model] = product
    # four_model_par_plot(len(product), y, state_names, "t-parameter", r'$\beta \odot t$', "Product of beta and t parameters for p50 model with grid optimization", "p50_model_grid_beta_t_product")
    
    # cpg_row = training_data.loc[(training_data["Genotype"] == "WT") & (training_data["Stimulus"] == "CpG")]
    # lps_row = training_data.loc[(training_data["Genotype"] == "WT") & (training_data["Stimulus"] == "LPS")]
    # cpg_values = [training_data.at[cpg_row.index[0], "NFkB"], training_data.at[cpg_row.index[0], "IRF"], training_data.at[cpg_row.index[0], "p50"]]
    # lps_values = [training_data.at[lps_row.index[0], "NFkB"], training_data.at[lps_row.index[0], "IRF"], training_data.at[lps_row.index[0], "p50"]]
    # stimulus_values = {"CpG": cpg_values, "LPS": lps_values}

    # # PLot f-contribution of each state
    # print("Plotting f-contribution of each state")
    # for stimulus in ["CpG", "LPS"]:
    #     I = stimulus_values[stimulus][0]
    #     N = stimulus_values[stimulus][1]
    #     P = stimulus_values[stimulus][2]
    #     y = {}
    #     for model in ["B1", "B2", "B3", "B4"]:
    #         t = results_global.loc[model].values[0:6]
    #         k = results_global.loc[model, "K_i2"]
    #         c = results_global.loc[model, "C"]
    #         f_values, state_names = get_f_contribution(t, k, c, N, I, P, model)
    #         y[model] = f_values
    #     four_model_par_plot(len(f_values), y, state_names, "State", r"IFN$\beta$ f-value", "f-contribution of each state for %s stimulation" % stimulus, "p50_model_grid_f_contribution_%s" % stimulus)

    # # Plot state probabilities
    # print("Plotting state probabilities")
    # for stimulus in ["CpG", "LPS"]:
    #     I = stimulus_values[stimulus][0]
    #     N = stimulus_values[stimulus][1]
    #     P = stimulus_values[stimulus][2]
    #     print(I, N, P)
    #     y = {}
    #     for model in ["B1", "B2", "B3", "B4"]:
    #         probabilities, state_names = get_state_prob(t, k, c, N, I, P, model)
    #         y[model] = probabilities
    #     four_model_par_plot(len(probabilities), y, state_names, "State", "Probability",
    #                         "State probabilities for %s stimulation" % stimulus, "p50_model_grid_state_probabilities_%s" % stimulus)

    # # Plot RMSD, AIC for each model
    # print("Plotting RMSD, AIC for each model")
    # for measure in ["rmsd", "AIC"]:
    #     fig, ax = plt.subplots()
    #     ax.set_prop_cycle("color", plt.cm.viridis(np.linspace(0,1,4)))
    #     m = results_global[measure]
    #     model_names = results_global.index
    #     ax.bar(model_names, m)
    #     plt.xlabel("Model")
    #     plt.ylabel("%s" % measure)
    #     plt.title("%s for each model" % measure.upper())
    #     plt.savefig("%s/p50_grid_%s.png" % (figures_dir, measure), bbox_inches="tight")

    # print("Done")

    # ## local results
    # # Plot best fit parameters
    # print("Plotting best fit parameters from local optimization")
    # fig, ax = plt.subplots(1,2, width_ratios=[1,2/6])
    # for axis in ax:
    #     axis.set_prop_cycle("color", plt.cm.viridis(np.linspace(0,1,4)))
    # i=0
    # for model in ["B1", "B2", "B3", "B4"]:
    #     t = results_local.loc[model].values[0:6]
    #     k = results_local.loc[model, "K_i2"]
    #     c = results_local.loc[model, "C"]
    #     x= np.arange(i, i+6)
    #     x2= np.arange(i, i+2)
    #     ax[0].plot(x, t, label=model, marker="o", linestyle="none")
    #     ax[1].plot(x2, [k, c], marker="o", linestyle="none")
    #     i+=0.1
    # ax[0].set_xticks(np.arange(0.15,6.15), [r"IRF_1", r"IRF_2", r"NF$\kappa$B", r"IRF_1 IRF_2", r"IRF_1 NF$\kappa$B", r"IRF_2 NF$\kappa$B"], 
    #            rotation=45)
    # ax[1].set_xticks(np.arange(0.05,2.05), ["$K_{i2}$", "C"])

    # fig.text(0.5, 0.04, 'Parameter', ha='center', size=14)
    # fig.text(0.04, 0.5, 'Transcription capability (t)', va='center', rotation='vertical', size=14)
    # plt.grid(False)
    # plt.suptitle("Best fit parameters for p50 model with grid+local optimization")
    # plt.tight_layout(pad = 4)
    # fig.legend(bbox_to_anchor=(1.1,0.5))
    # plt.savefig("%s/p50_grid_best_fit_parameters_local.png" % figures_dir, bbox_inches="tight")

    # # Plot product of beta and t parameters
    # print("Plotting product of beta and t parameters from local optimization")
    # y = {}
    # for model in ["B1", "B2", "B3", "B4"]:
    #     t = results_local.loc[model].values[0:6]
    #     k = results_local.loc[model, "K_i2"]
    #     c = results_local.loc[model, "C"]
    #     product, state_names = get_product(t, k, c, model)
    #     y[model] = product
    # four_model_par_plot(len(product), y, state_names, "t-parameter", r'$\beta \odot t$', "Product of beta and t parameters for p50 model with grid+local optimization", "p50_model_grid_beta_t_product_local")

    # # Plot f-contribution of each state
    # print("Plotting f-contribution of each state from local optimization")
    # for stimulus in ["CpG", "LPS"]:
    #     I = stimulus_values[stimulus][0]
    #     N = stimulus_values[stimulus][1]
    #     P = stimulus_values[stimulus][2]
    #     y = {}
    #     for model in ["B1", "B2", "B3", "B4"]:
    #         t = results_local.loc[model].values[0:6]
    #         k = results_local.loc[model, "K_i2"]
    #         c = results_local.loc[model, "C"]
    #         f_values, state_names = get_f_contribution(t, k, c, N, I, P, model)
    #         y[model] = f_values
    #     four_model_par_plot(len(f_values), y, state_names, "State", r"IFN$\beta$ f-value", "f-contribution of each state for %s stimulation" % stimulus, "p50_model_grid_f_contribution_%s_local" % stimulus)

    # # Plot RMSD, AIC for each model from local optimization
    # print("Plotting RMSD, AIC for each model from local optimization")
    # for measure in ["rmsd", "aic"]:
    #     fig, ax = plt.subplots()
    #     ax.set_prop_cycle("color", plt.cm.viridis(np.linspace(0,1,4)))
    #     m = results_local[measure]
    #     model_names = results_local.index
    #     ax.bar(model_names, m)
    #     plt.xlabel("Model")
    #     plt.ylabel("%s" % measure)
    #     plt.title("%s for each model" % measure.upper())
    #     plt.savefig("%s/p50_grid_%s_local.png" % (figures_dir, measure), bbox_inches="tight")

    # # Plot residuals for all models
    # print("Plotting residuals for all models")
    # fig, ax = plt.subplots()
    # ax.set_prop_cycle("color", plt.cm.viridis(np.linspace(0,1,4)))
    # for model in ["B1", "B2", "B3", "B4"]:
    #     result_values = results_local.loc[model]
    #     # residuals are columns starting with "res"
    #     residuals = result_values[result_values.index.str.startswith("res")]
    #     plt.plot(residuals, 'o', label=model)
    # plt.xlabel("Data point")
    # plt.ylabel("Residuals")
    # plt.title("Residuals for each model local optimization")
    # plt.legend()
    # plt.savefig("%s/p50_grid_residuals_local.png" % figures_dir, bbox_inches="tight")

    # # Make contour plots from best fit parameters
    # I = np.linspace(0, 1, 100)
    # N= I.copy()
    # P = {"WT": 1, "p50KO": 0}

    # for model in ["B1", "B2", "B3", "B4"]:
    #     t_pars = results_local.loc[model].values[0:6]
    #     K = results_local.loc[model, "K_i2"]
    #     C = results_local.loc[model, "C"]
    #     f_dict = {}
    #     for genotype in P.keys():
    #         N_vals, I_vals = np.meshgrid(N, I)
    #         f_values = np.zeros((len(N), len(I)))
    #         with Pool(40) as p:
    #             for i in range(len(N_vals)):
    #                 f_values[i,:] = p.starmap(get_f, [(t_pars, K, C, N_vals[i,j], I_vals[i,j], P[genotype], model) for j in range(len(N_vals[i,:]))]) 
    #         title = "best_fit_grid_%s_%s_local" % (genotype, model)
    #         plot_contour(f_values, model, I_vals, N_vals, results_dir, title, condition=genotype)
    #         plot_contour(f_values, model, I_vals, N_vals, results_dir, "%s_unnormalized" % title, condition=genotype, normalize=False)
    #         f_dict[genotype] = f_values
    #     # Calculate fold change between WT and p50KO
    #     f_dict["WT"][f_dict["WT"] == 0] = 10**-10
    #     f_fold_change = f_dict["p50KO"]/f_dict["WT"]
    #     title = "best_fit_genotype_fold_change_%s_local" % model
    #     plot_contour(f_fold_change, model, I_vals, N_vals, results_dir, title, condition="fold change (p50 KO/WT)", normalize=False)

if __name__ == "__main__":
    main()