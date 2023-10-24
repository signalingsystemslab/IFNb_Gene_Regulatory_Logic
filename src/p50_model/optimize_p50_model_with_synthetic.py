# Optimize p50 model with grid search
from p50_model import *
import matplotlib.pyplot as plt
import pandas as pd
import os
import scipy.optimize as opt
import time
from multiprocessing import Pool
plt.style.use("~/IFN_paper/src/theme_bw.mplstyle")

figures_dir = "./opt_synthetic/figures/"
results_dir = "./opt_synthetic/results/"
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

    f_list = [get_f(t_pars, K, C, N[i], I[i], P[i], model_name) for i in range(num_pts)]
    # Normalize to highest value
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

def save_results(results_df, model, pars, other):
    if model == "B1":
        res = np.hstack([pars, np.nan,np.nan,other])
    elif model == "B2":
        res = np.hstack([pars, np.nan,other])
    elif model == "B3":
        res = np.hstack([pars[0:6], np.nan, pars[6], other])
    elif model == "B4":
        res = np.hstack([pars, other])

    print("Results to add: %s\n Size of results df %s" % (res, results_df.shape))

    results_df.loc[model] = res
    return results_df

def main():
    print("###############################################\n")
    print("Optimizing p50 model with grid optimization at %s" % time.ctime())
    training_data = pd.read_csv("../data/p50_training_data_plus_synthetic.csv")
    print("Using the following training data:\n", training_data)
    print("Starting at %s" % time.ctime())

    N = training_data["NFkB"]
    I = training_data["IRF"]
    P = training_data["p50"]
    beta = training_data["IFNb"]
    model_par_numbers = {"B1": 6, "B2": 7, "B3": 7, "B4": 8}
    num_pts = len(N)
    num_threads = 40
    print("The total number of data points is %d" % num_pts)

    res_title = ["t1", "t2", "t3", "t4", "t5", "t6","K_i2", "C","rmsd", "AIC"]
    results = pd.DataFrame(columns=res_title)
    results_local = pd.DataFrame(columns= ["t1", "t2", "t3", "t4", "t5", "t6","K_i2", "C","rmsd","aic"] + ["res_%d" % i for i in range(num_pts)])

    for model in model_par_numbers.keys():
    # for model in ["B1"]:
        print("\n\n###############################################")
        print("OPTIMIZING MODEL %s" % model)
        print("###############################################\n")
        # Grid search
        pars, min_rmsd, grid, jout = optimize_model(N, I, P, beta, model, num_threads=num_threads)
        # print("\n shape of grid: %s, \n shape of jout: %s" % (grid.shape, jout.shape))
        # print("Minimum RMSD: %.4f" % min_rmsd)

        # save results
        np.save("%s/p50_grid_rmsd_model_%s.npy" % (results_dir, model), min_rmsd)
        np.save("%s/p50_grid_params_model_%s.npy" % (results_dir, model), pars)
        np.save("%s/p50_grid_jout_model_%s.npy" % (results_dir, model), jout)
        np.save("%s/p50_grid_grid_model_%s.npy" % (results_dir, model), grid)

        # Calculate AIC from rmsd and number of parameters
        min_aic = num_pts * np.log(min_rmsd) + 2 * model_par_numbers[model]
        np.save("%s/p50_grid_aic_model_%s.npy" % (results_dir, model), min_aic)
        print("Minimum AIC: %.4f" % min_aic)
        # Save results
        results = save_results(results, model, pars, [min_rmsd, min_aic])

        # Plot all rmsd values
        plot_optimization(grid, jout, model, measure="RMSD")

        # Local optimization
        pars, cost, residuals = optimize_model_local(N, I, P, beta, model, pars)
        rmsd = np.sqrt(np.mean(residuals**2))
        aic = num_pts * np.log(rmsd) + 2 * model_par_numbers[model]
        results_local = save_results(results_local, model, pars, np.hstack([rmsd, aic, residuals]))

    results.to_csv("%s/p50_grid_global_optimization_results.csv" % results_dir)
    results_local.to_csv("%s/p50_grid_local_optimization_results.csv" % results_dir)
    print("Saved all results to %s \n\n\n" % results_dir)

    # Save results with best AIC to use
    results_local = pd.read_csv("%s/p50_grid_local_optimization_results.csv" % results_dir, index_col=0)
    best_results = results_local.iloc[np.argmin(results_local["rmsd"])]
    best_results = best_results.fillna(1)
    best_model = best_results.name
    K = best_results.loc["K_i2"]
    C = best_results.loc["C"]
    
    t_pars = best_results[0:6]
    t_pars = np.array(t_pars)
    f_list_best = [get_f(t_pars, K, C, N[i], I[i], P[i], best_model) for i in range(num_pts)]
    print("\n\n ## Best fit model information ## \n\n Model = %s, K = %.2f, C = %.2f" % (best_model, K, C))
    for i in range(len(t_pars)):
        print("t%d = %.4f" % (i+1, t_pars[i]))
    print("Predicted IFNb values:")
    for n, i, p, b, b_train in zip(N, I, P, f_list_best, beta):
        print("NFkB = %.2f, IRF = %.2f, p50= %s, IFNb = %.2f, IFNb training = %.4f" % (n, i, p, b, b_train))

    best_results["scale"] = np.max(training_data["IFNb"]) / np.max(f_list_best)
    ifnb_half_life = 2.5*60 # From Rios 2014
    best_results["p_deg_ifnb"] = np.log(2) / ifnb_half_life

    print(best_results)
    # save without column name
    best_results.to_csv("%s/ifnb_best_params_grid_local.csv" % results_dir, header=False, na_rep=1)
    print("Best results from model %s:\n" % best_results.name, best_results)
    print("Saved best results to %s/ifnb_best_params_grid_local.csv" % os.path.abspath(results_dir))

    ## global results
    results_global = pd.read_csv("%s/p50_grid_global_optimization_results.csv" % results_dir, index_col=0)
    # Make contour plots from best fit parameters
    I = np.linspace(0, 1, 100)
    N= I.copy()
    P = {"WT": 1, "p50KO": 0}

    for model in ["B1", "B2", "B3", "B4"]:
        t_pars = results_global.loc[model].values[0:6]
        K = results_global.loc[model, "K_i2"]
        C = results_global.loc[model, "C"]

        f_dict = {}
        for genotype in P.keys():
            N_vals, I_vals = np.meshgrid(N, I)
            # N = N_vals.flatten()
            # I = I_vals.flatten()
            f_values = np.zeros((len(N), len(I)))
            with Pool(40) as p:
                for i in range(len(N_vals)):
                    f_values[i,:] = p.starmap(get_f, [(t_pars, K, C, N_vals[i,j], I_vals[i,j], P[genotype], model) for j in range(len(N_vals[i,:]))]) 
                # f_values = p.starmap(get_f, [(t_pars, K, C, N[i], I[i], P[genotype], model) for i in range(len(N))])
            title = "best_fit_grid_%s_%s" % (genotype, model)
            plot_contour(f_values, model, I_vals, N_vals, results_dir, title, condition=genotype)
            f_dict[genotype] = f_values
        # Calculate fold change between WT and p50KO
        f_dict["WT"][f_dict["WT"] == 0] = 10**-10
        f_fold_change = f_dict["p50KO"]/f_dict["WT"]
        title = "best_fit_genotype_fold_change_%s" % model
        plot_contour(f_fold_change, model, I_vals, N_vals, results_dir, title, condition="fold change (p50 KO/WT)", normalize=False)

    #  Plot best fit parameters
    print("Plotting best fit parameters")
    fig, ax = plt.subplots(1,2, width_ratios=[1,2/6])
    for axis in ax:
        axis.set_prop_cycle("color", plt.cm.viridis(np.linspace(0,1,4)))
    i=0
    for model in ["B1", "B2", "B3", "B4"]:
        t = results_global.loc[model].values[0:6]
        k = results_global.loc[model, "K_i2"]
        c = results_global.loc[model, "C"]
        x= np.arange(i, i+6)
        x2= np.arange(i, i+2)
        ax[0].plot(x, t, label=model, marker="o", linestyle="none")
        ax[1].plot(x2, [k, c], marker="o", linestyle="none")
        i+=0.1
    ax[0].set_xticks(np.arange(0.15,6.15), [r"IRF_1", r"IRF_2", r"NF$\kappa$B", r"IRF_1 IRF_2", r"IRF_1 NF$\kappa$B", r"IRF_2 NF$\kappa$B"], 
               rotation=45)
    ax[1].set_xticks(np.arange(0.05,2.05), ["$K_{i2}$", "C"])

    fig.text(0.5, 0.04, 'Parameter', ha='center', size=14)
    fig.text(0.04, 0.5, 'Transcription capability (t)', va='center', rotation='vertical', size=14)
    plt.grid(False)
    plt.suptitle("Best fit parameters for p50 model with grid optimization")
    plt.tight_layout(pad = 4)
    fig.legend(bbox_to_anchor=(1.1,0.5))
    plt.savefig("%s/p50_grid_best_fit_parameters.png" % figures_dir, bbox_inches="tight")

    # Plot RMSD, AIC for each model
    print("Plotting RMSD, AIC for each model")
    for measure in ["rmsd", "AIC"]:
        fig, ax = plt.subplots()
        ax.set_prop_cycle("color", plt.cm.viridis(np.linspace(0,1,4)))
        m = results_global[measure]
        model_names = results_global.index
        ax.bar(model_names, m)
        plt.xlabel("Model")
        plt.ylabel("%s" % measure)
        plt.title("%s for each model" % measure.upper())
        plt.savefig("%s/p50_grid_%s.png" % (figures_dir, measure), bbox_inches="tight")

    print("Done")

    ## local results
    # Plot best fit parameters
    print("Plotting best fit parameters from local optimization")
    fig, ax = plt.subplots(1,2, width_ratios=[1,2/6])
    for axis in ax:
        axis.set_prop_cycle("color", plt.cm.viridis(np.linspace(0,1,4)))
    i=0
    for model in ["B1", "B2", "B3", "B4"]:
        t = results_local.loc[model].values[0:6]
        k = results_local.loc[model, "K_i2"]
        c = results_local.loc[model, "C"]
        x= np.arange(i, i+6)
        x2= np.arange(i, i+2)
        ax[0].plot(x, t, label=model, marker="o", linestyle="none")
        ax[1].plot(x2, [k, c], marker="o", linestyle="none")
        i+=0.1
    ax[0].set_xticks(np.arange(0.15,6.15), [r"IRF_1", r"IRF_2", r"NF$\kappa$B", r"IRF_1 IRF_2", r"IRF_1 NF$\kappa$B", r"IRF_2 NF$\kappa$B"], 
               rotation=45)
    ax[1].set_xticks(np.arange(0.05,2.05), ["$K_{i2}$", "C"])

    fig.text(0.5, 0.04, 'Parameter', ha='center', size=14)
    fig.text(0.04, 0.5, 'Transcription capability (t)', va='center', rotation='vertical', size=14)
    plt.grid(False)
    plt.suptitle("Best fit parameters for p50 model with grid+local optimization")
    plt.tight_layout(pad = 4)
    fig.legend(bbox_to_anchor=(1.1,0.5))
    plt.savefig("%s/p50_grid_best_fit_parameters_local.png" % figures_dir, bbox_inches="tight")

    # Plot RMSD, AIC for each model from local optimization
    print("Plotting RMSD, AIC for each model from local optimization")
    for measure in ["rmsd", "aic"]:
        fig, ax = plt.subplots()
        ax.set_prop_cycle("color", plt.cm.viridis(np.linspace(0,1,4)))
        plt.plot(results_local[measure], 'o')
        plt.xlabel("Model")
        plt.ylabel("%s" % measure)
        plt.title("%s for each model grid+local optimization" % measure.upper()) 
        plt.savefig("%s/p50_grid_%s_local.png" % (figures_dir, measure), bbox_inches="tight")

    # Plot residuals for all models
    print("Plotting residuals for all models")
    fig, ax = plt.subplots()
    ax.set_prop_cycle("color", plt.cm.viridis(np.linspace(0,1,4)))
    for model in ["B1", "B2", "B3", "B4"]:
        result_values = results_local.loc[model]
        # residuals are columns starting with "res"
        residuals = result_values[result_values.index.str.startswith("res")]
        plt.plot(residuals, 'o', label=model)
    plt.xlabel("Data point")
    plt.ylabel("Residuals")
    plt.title("Residuals for each model local optimization")
    plt.legend()
    plt.savefig("%s/p50_grid_residuals_local.png" % figures_dir, bbox_inches="tight")

    # Make contour plots from best fit parameters
    I = np.linspace(0, 1, 100)
    N= I.copy()
    P = {"WT": 1, "p50KO": 0}

    for model in ["B1", "B2", "B3", "B4"]:
        t_pars = results_local.loc[model].values[0:6]
        K = results_local.loc[model, "K_i2"]
        C = results_local.loc[model, "C"]
        f_dict = {}
        for genotype in P.keys():
            N_vals, I_vals = np.meshgrid(N, I)
            f_values = np.zeros((len(N), len(I)))
            with Pool(40) as p:
                for i in range(len(N_vals)):
                    f_values[i,:] = p.starmap(get_f, [(t_pars, K, C, N_vals[i,j], I_vals[i,j], P[genotype], model) for j in range(len(N_vals[i,:]))]) 
            title = "best_fit_grid_%s_%s_local" % (genotype, model)
            plot_contour(f_values, model, I_vals, N_vals, results_dir, title, condition=genotype)
            f_dict[genotype] = f_values
        # Calculate fold change between WT and p50KO
        f_dict["WT"][f_dict["WT"] == 0] = 10**-10
        f_fold_change = f_dict["p50KO"]/f_dict["WT"]
        title = "best_fit_genotype_fold_change_%s_local" % model
        plot_contour(f_fold_change, model, I_vals, N_vals, results_dir, title, condition="fold change (p50 KO/WT)", normalize=False)

if __name__ == "__main__":
    main()