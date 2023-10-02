# Optimize p50 model by randomly selecting initial parameters
from p50_model import *
import matplotlib.pyplot as plt
import pandas as pd
import os
import scipy.optimize as opt
import time
from multiprocessing import Pool
plt.style.use("~/IFN_paper/src/theme_bw.mplstyle")

figures_dir = "./figures/random_opt/"
results_dir = "./results/random_opt/"
os.makedirs(results_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

print("Loading data")
training_data = pd.read_csv("../data/p50_training_data.csv")
# num_pts = training_data.shape[0]
# print(training_data)
# t_pars, K, C, N, I, P, model_name="B2", scaling=1
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
    f_list = f_list / np.max(f_list)
    residuals = np.array(f_list) - beta
    if strategy != "rmsd":
        return residuals
    else:
        rmsd = np.sqrt(np.mean(residuals**2))
        return rmsd
    
# def test_objective(pars, *args):
#     N, I, beta, model_name, strategy = args
#     return pars[0]

def select_params(num_params, par_length, seed=5, lower=0, upper=1):
    params = np.zeros((num_params, par_length))
    np.random.seed(seed)
    for i in range(num_params):
        params[i] = np.random.uniform(lower, upper, par_length)
    return params

def optimize_model(N, I, P, beta, model_name, params, num_threads=40):
    print("###############################################\n")
    print("Optimizing p50 model %s globally" % model_name)
    start = time.time()
    print("Starting random optimization at %s" % time.ctime())

    # rmsd = np.zeros(par_length)
    with Pool(num_threads) as p:
            res = p.starmap(p50_objective, [(params[j], N, I, P, beta, model_name, "rmsd") for j in range(params.shape[0])])
            rmsd = np.array(res)

    end = time.time()
    print("Size of rmsd: ", rmsd.shape)
    print("Finished optimization at %s" % time.ctime())
    t = end - start
    if t < 60:
        print("Time elapsed: %.2f seconds" % t)
    elif t < 3600:
        print("Time elapsed: %.2f minutes" % (t/60))
    else:
        print("Time elapsed: %.2f hours" % (t/3600))

    return params, rmsd

def plot_optimization(params, rmsd, model_name):
    #  Plot individual param against rmsd
    for i in range(params.shape[1]):
        plt.figure()
        plt.plot(params[:,i], rmsd, 'o')
        plt.xlabel("Parameter %d" % (i+1))
        plt.ylabel("RMSD")
        plt.title("Random optimization of %s model" % model_name)
        plt.savefig(figures_dir + "random_optimization_%s_model_param_%d.png" % (model_name, i+1))
        plt.close()


    # Sort rmsd
    rmsd = np.sort(rmsd)
    plt.figure()
    plt.plot(rmsd, 'o')
    plt.xlabel("Iteration (sorted)")
    plt.ylabel("RMSD")
    plt.title("Random optimization of %s model" % model_name)
    plt.savefig(figures_dir + "random_optimization_%s_model.png" % model_name)
    plt.close()

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
    return res.x, res.cost, res.fun

def save_results(results_df, model, pars, other):
    if model == "B1":
        res = np.hstack([pars, np.nan,np.nan,other])
    elif model == "B2":
        res = np.hstack([pars, np.nan,other])
    elif model == "B3":
        res = np.hstack([pars[0:6], np.nan, pars[6], other])
    elif model == "B4":
        res = np.hstack([pars, other])

    print("results to add: ", res)
    print("Size of results df: ", results_df.shape)

    results_df.loc[model] = res
    return results_df

def main():
    print("###############################################\n")
    print("Optimizing p50 model with random optimization")
    print("Using the following training data:\n", training_data)
    print("Starting at %s" % time.ctime())
    N = training_data["NFkB"]
    I = training_data["IRF"]
    P = training_data["p50"]
    beta = training_data["IFNb"]
    model_par_numbers = {"B1": 6, "B2": 7, "B3": 7, "B4": 8}
    npars = 100000
    pars_initial = select_params(npars, 6)
    num_pts = len(N)

    res_title = ["t1", "t2", "t3", "t4", "t5", "t6","K_i2", "C","rmsd"]
    results = pd.DataFrame(columns=res_title)
    results_local = pd.DataFrame(columns= ["t1", "t2", "t3", "t4", "t5", "t6","K_i2", "C","rho"] + ["res_%d" % i for i in range(num_pts)])

    for model in model_par_numbers.keys():
        # Random optimization
        num_remaining = model_par_numbers[model] - 6
        params = pars_initial
        if num_remaining > 0:
            if model == "B2":
                upper = 2
                lower = 0
            elif model == "B3":
                upper = 10**2
                lower = 10**-2
            elif model == "B4":
                upper = [2, 10**2]
                lower =  [0, 10**-2]
            params = np.hstack([params, select_params(npars, num_remaining, lower=lower, upper=upper)])

        params, rmsd = optimize_model(N, I, P, beta, model, params)
        plot_optimization(params, rmsd, model)
        np.save("%s/p50_random_rmsd_model_%s.npy" % (results_dir, model), rmsd)
        np.save("%s/p50_random_params_model_%s.npy" % (results_dir, model), params)

        pars = params[np.argmin(rmsd),:]
        min_rmsd = np.min(rmsd)
        # Save results
        results = save_results(results, model, pars, min_rmsd)

        # Local optimization
        pars = params[np.argmin(rmsd),:]
        pars, rho, residuals = optimize_model_local(N, I, P, beta, model, pars)
        results_local = save_results(results_local, model, pars, np.hstack([rho, residuals]))

    results.to_csv("%s/p50_random_global_optimization_results.csv" % results_dir)
    results_local.to_csv("%s/p50_random_local_optimization_results.csv" % results_dir)

    # Save results with best rmsd to use
    results = pd.read_csv("%s/p50_random_global_optimization_results.csv" % results_dir, index_col=0)
    best_results = results.iloc[np.argmin(results["rmsd"]),:]
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
    best_results.to_csv("%s/ifnb_best_params_random_global.csv" % results_dir, header=False, na_rep=1)
    print("Best results from model %s:\n" % best_results.name, best_results)
    print("Saved best results to %s/ifnb_best_params_random_global.csv" % os.path.abspath(results_dir))

    # Make contour plots from best fit parameters
    t_pars = results.loc[best_model].values[0:6]
    K = best_results.loc["K_i2"]
    C = best_results.loc["C"]
    I = np.linspace(0, 1, 100)
    N= I.copy()
    P = {"WT": 1, "p50KO": 0}

    for model in ["B1", "B2", "B3", "B4"]:
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
            title = "best_fit_random_%s_%s" % (genotype, model)
            plot_contour(f_values, model, I_vals, N_vals, results_dir, title, condition=genotype)
            f_dict[genotype] = f_values
        # Calculate fold change between WT and p50KO
        f_dict["WT"][f_dict["WT"] == 0] = 10**-10
        f_fold_change = f_dict["p50KO"]/f_dict["WT"]
        title = "best_fit_genotype_fold_change_%s" % model
        plot_contour(f_fold_change, model, I_vals, N_vals, results_dir, title, condition="fold change (p50 KO/WT)", normalize=False)

    #  Plot best fit parameters
    fig = plt.figure()
    i=0
    for model in ["B1", "B2", "B3", "B4"]:
        x= np.arange(i, i+6)
        plt.plot(x, t_pars, label=model, marker="o", linestyle="none")
        i+=0.1
    plt.legend(bbox_to_anchor=(1.2,0.5))
    plt.ylabel("Transcription capability (t)")
    plt.xlabel("Parameter")
    plt.grid(False)
    # plt.xticks(range(6), list(results)[0:6])
    plt.xticks(np.arange(0.2,6.2), [r"IRF_1", r"IRF_2", r"NF$\kappa$B", r"IRF_1 IRF_2", r"IRF_1 NF$\kappa$B", r"IRF_2 NF$\kappa$B"], rotation=45)
    plt.title("Best fit parameters for p50 model with random optimization")
    plt.savefig("%s/p50_random_best_fit_parameters.png" % figures_dir, bbox_inches="tight")

   # Compare predicted IFNb to testing data
    test_data = pd.read_csv("../data/p50_testing_data.csv")
    testing_rmsd = {}
    f_vals = test_data.copy()
    for model in ["B1", "B2", "B3", "B4"]:
        f = [get_f(t_pars, K, C, test_data["NFkB"][i], test_data["IRF"][i], test_data["p50"][i], model) for i in range(test_data.shape[0])]
        # f = [explore_modelp50(pars[model], test_data["NFkB"][i], test_data["IRF"][i], test_data["p50"][i], model) for i in range(test_data.shape[0])]
        f_vals["IFNb_%s" % model] = f
        rmsd = np.sqrt(np.mean((f / f[1] * f_vals["IFNb"][1] - test_data["IFNb"])**2))
        testing_rmsd[model] = rmsd
    print("Testing RMSD:\n", testing_rmsd)


    p50_vals = np.linspace(0, 2, 100)
    fig = plt.figure()
    for model_name in ["B1", "B2", "B3", "B4"]:
        ifnb = [get_f(t_pars, K, C, test_data["NFkB"][0], test_data["IRF"][0], p50, model_name) for p50 in p50_vals]
        # ifnb = [explore_modelp50(pars[model_name], test_data["NFkB"][0], test_data["IRF"][0], p50, model_name) for p50 in p50_vals]
        # print("IFNb at p50=0: %.2f, at p50=1: %.2f, at p50=2: %.2f" % (ifnb[0], ifnb[50], ifnb[99]))
        #  Normalize to p50 = 1
        ifnb = ifnb / f_vals["IFNb_%s" % model_name][1] * f_vals["IFNb"][1] 
        print("IFNb at p50=0: %.2f, at p50=1: %.2f, at p50=2: %.2f" % (ifnb[0], ifnb[50], ifnb[99]))
        plt.plot(p50_vals, ifnb, label=model_name)
    plt.scatter(test_data["p50"], test_data["IFNb"], s=50, label="Testing Data", zorder =2, color="black")
    plt.xlabel("p50")
    plt.ylabel(r"IFN$\beta$")
    plt.ylim([0,1])
    plt.title("Model predictions for all models")
    plt.grid(False)
    fig.legend(bbox_to_anchor=(1.15,0.5))
    plt.savefig("%s/random_opt_testing_predictions.png" % results_dir, bbox_inches="tight")

if __name__ == "__main__":
    main()