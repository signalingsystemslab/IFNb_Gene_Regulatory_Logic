from two_site_model import *
import matplotlib.pyplot as plt
import pandas as pd
import os
import scipy.optimize as opt
import time
from multiprocessing import Pool
plt.style.use("~/IFN_paper/src/theme_bw.mplstyle")

figures_dir = "two_site_no-coop/figures/"
results_dir = "two_site_no-coop/results/"
os.makedirs(results_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

def two_site_objective(t_pars, *args):
    N, I, beta, model_name, strategy = args
    num_pts = len(N)
    C = 1
    t_pars = np.concatenate(([0], t_pars, [1]))

    f_list = [get_f(C, N[i], I[i], model_name, t_pars) for i in range(num_pts)]
    # Normalize to highest value
    if np.max(f_list) != 0:
        f_list = f_list / np.max(f_list)
    residuals = np.array(f_list) - beta
    if strategy != "rmsd":
        return residuals
    else:
        rmsd = np.sqrt(np.mean(residuals**2))
        return rmsd
    
def optimize_model(N, I, beta, model_name, num_threads=40):
    print("###############################################\n")
    print("Optimizing two site model %s globally" % model_name)
    start = time.time()
    print("Starting brute force optimization at %s" % time.ctime())
    rgs = ((0,1),(0,1))

    res = opt.brute(two_site_objective, rgs, args=(N, I, beta, model_name, "rmsd"), Ns=30, full_output=True, finish=None,
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

def optimize_model_local(N, I, beta, model_name, pars):
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
    upper = np.array([1,1])
    lower = np.array([0,0])

    print("Upper bounds: ", upper)
    print("Lower bounds: ", lower)
    print("Initial parameters: ", pars)
    res = opt.least_squares(two_site_objective, pars, bounds = (lower, upper),
                            args=(N, I, beta, model_name, "residuals"), method=method, loss = "linear")
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
    print("Optimizing two site model for t-parameters at %s" % time.ctime())
    training_data = pd.read_csv("../data/two_site_training_data.csv")
    print("Using the following training data:\n", training_data)
    print("Starting at %s" % time.ctime())

    N = training_data["NFkB"]
    I = training_data["IRF"]
    beta = training_data["IFNb"]
    model_par_number = 2
    num_pts = len(N)
    num_threads = 40

    print("\n\n###############################################")
    print("OPTIMIZING MODEL FOR T-PARAMETERS")
    print("###############################################\n", flush=True)
    model = "Other"

    # Optimize model globally
    global_pars, global_rmsd, grid, jout = optimize_model(N, I, beta, model, num_threads=num_threads)
    g_pars = np.concatenate(([0], global_pars, [1]))
    print("Optimized parameters:\n", g_pars)

    # Optimize model locally
    pars, cost, residuals = optimize_model_local(N, I, beta, model, global_pars)
    pars = np.concatenate(([0], pars, [1]))
    print("Optimized parameters:\n", pars)
    rmsd = np.sqrt(np.mean(residuals**2))
    print("RMSD: ", rmsd)
    aic = 2 * model_par_number + num_pts * np.log(rmsd**2)
    print("AIC: ", aic)

    # Save pars and rmsd to csv
    np.savetxt("%s/two_site_t_pars.csv" % results_dir, pars, delimiter=",")
    np.savetxt("%s/two_site_rmsd.csv" % results_dir, [rmsd], delimiter=",")
    np.savetxt("%s/two_site_residuals.csv" % results_dir, residuals, delimiter=",")

    print("\n\n###############################################")
    print("PLOTTING RESULTS")
    print("###############################################\n", flush=True)

    par_names = ["None", "IRF", r"$NF\kappa B$", r"$IRF + NF\kappa B$"]

    # Plot best-fit parameters
    fig, ax = plt.subplots()
    ax.bar(par_names, pars)
    ax.set_ylabel("t-parameter")
    ax.set_title("Best-fit t-parameters")
    fig.savefig("%s/t_pars.png" % figures_dir)

    # Plot residuals
    fig, ax = plt.subplots()
    ax.scatter(range(len(residuals)), residuals)
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals")
    fig.savefig("%s/residuals.png" % figures_dir)


    # Plot grid
    fig, ax = plt.subplots()
    p = ax.contourf(grid[0], grid[1], jout, 100, cmap="viridis")
    fig.colorbar(p)
    ax.set_xlabel(r"t$_1$")
    ax.set_ylabel(r"t$_2$")
    ax.set_title("Grid search")
    fig.savefig("%s/grid_global_results.png" % figures_dir)

    # Plot contour plot
    N = np.linspace(0, 1, 100)
    I = np.linspace(0, 1, 100)
    f = np.zeros((len(N), len(I)))
    for j in range(len(N)):
        for k in range(len(I)):
            f[j,k] = get_f(1, N[j], I[k], model, pars)

    plot_contour(f, model, I, N, figures_dir, "best_fit_local", condition="")

    # Plot minimum and maximum ifnb for each nfkb
    fig, ax = plt.subplots()
    ax.set_prop_cycle("color", plt.cm.viridis(np.linspace(0, 1, 5)))
    ax.plot(N, np.max(f, axis=0), label=r"Maximum IFN$\beta$", linewidth=3)
    ax.fill_between(N, np.min(f, axis=0), np.max(f, axis=0), alpha=0.2, label = "Contribution of IRF")
    ax.plot(N, np.min(f, axis=0), label=r"Minimum IFN$\beta$", linewidth=3)
    ax.set_xlabel(r"$NF\kappa B$")
    ax.set_ylabel(r"IFN$\beta$")
    # ax.set_title("Model %s" % model)
    fig.legend(bbox_to_anchor=(1.23, 0.5))
    fig.savefig("%s/nfkb_vs_min_max_ifnb_%s.png" % (figures_dir, model))

    # Plot minimum and maximum ifnb for each irf
    fig, ax = plt.subplots()
    ax.set_prop_cycle("color", plt.cm.viridis(np.linspace(0, 1, 5)))
    ax.plot(I, np.max(f, axis=1), label=r"Maximum IFN$\beta$", linewidth=3)
    ax.fill_between(I, np.min(f, axis=1), np.max(f, axis=1), alpha=0.2, label = "Contribution of NF$\kappa$B")
    ax.plot(I, np.min(f, axis=1), label=r"Minimum IFN$\beta$", linewidth=3)
    ax.set_xlabel(r"$IRF$")
    ax.set_ylabel(r"IFN$\beta$")
    # ax.set_title("Model %s" % model)
    fig.legend(bbox_to_anchor=(1.25, 0.5))
    fig.savefig("%s/irf_vs_min_max_ifnb_%s.png" % (figures_dir, model))

    # Plot contributions of each TF
    N = 1.0
    I = 0.25
    f_contributions, state_names = get_f_contribution(1, N, I, model, pars)
    f_contributions = f_contributions / np.sum(f_contributions)
    fig, ax = plt.subplots()
    ax.bar(state_names, f_contributions)
    ax.set_ylabel(r"Contribution to IFN$\beta$")
    ax.set_title("Contribution of each TF (fraction)")
    fig.savefig("%s/best_fit_contributions_LPS_WT.png" % figures_dir)


    print("\n\n###############################################")
    print("OPTIMIZING MODEL FOR FOUR MODEL STRATEGIES")
    print("###############################################\n", flush=True)

    # Calculate RMSD for each strategy
    strategies = ["IRF", "NFkB", "AND", "OR"]
    N = training_data["NFkB"]
    I = training_data["IRF"]
    beta = training_data["IFNb"]

    rmsd_list = []
    for s in strategies:
        f_values = [get_f(1, n, i, s) for n, i in zip(N, I)]
        residuals = f_values - beta
        rmsd = np.sqrt(np.mean(residuals**2))
        rmsd_list.append(rmsd)

        # Plot residuals
        fig, ax = plt.subplots()
        ax.scatter(range(len(residuals)), residuals)
        ax.set_ylabel("Residuals")
        ax.set_title("Residuals for %s" % s)
        fig.savefig("%s/%s_residuals.png" % (figures_dir, s))

    print("Plotting results for each strategy")
    # Plot RMSD for each strategy
    fig, ax = plt.subplots()
    ax.bar(strategies, rmsd_list)
    ax.set_ylabel("RMSD")
    ax.set_title("RMSD for each strategy")
    fig.savefig("%s/rmsd_strategies.png" % figures_dir)

    # Plot contributions of each TF for each strategy
    N = 1.0
    I = 0.25
    for s in strategies:
        f_contributions, state_names = get_f_contribution(1, N, I, s)
        f_contributions = f_contributions / np.sum(f_contributions)
        fig, ax = plt.subplots()
        ax.bar(state_names, f_contributions)
        ax.set_ylabel(r"Contribution to IFN$\beta$")
        ax.set_title("Contribution of each TF for %s (fraction)" % s)
        fig.savefig("%s/%s_contributions_LPS_WT.png" % (figures_dir, s))

        # Plot contour plot
        N_vals = np.linspace(0, 1, 100)
        I_vals = np.linspace(0, 1, 100)
        f = np.zeros((len(N_vals), len(I_vals)))
        for j in range(len(N_vals)):
            for k in range(len(I_vals)):
                f[j,k] = get_f(1, N_vals[j], I_vals[k], s)

        plot_contour(f, s, I_vals, N_vals, figures_dir, "%s_contour" % s, condition="%s" % s)



if __name__ == "__main__":
    main()