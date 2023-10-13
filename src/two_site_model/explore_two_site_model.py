from two_site_model import *
import matplotlib.pyplot as plt
import pandas as pd
import os
import scipy.optimize as opt
import time
from multiprocessing import Pool
plt.style.use("~/IFN_paper/src/theme_bw.mplstyle")

optimization_dir = "./grid_opt/results"
results_dir = "./explore_two_site_model_results/"
os.makedirs(results_dir, exist_ok=True)

def main():
    # Load best fit parameters from "%s/two_site_grid_local_optimization_results.csv" % optimization_dir
    pars = pd.read_csv("%s/ifnb_best_params_grid_global.csv" % optimization_dir, index_col=0, header=None)
    best_model = "IRF"
    C = pars.loc["C", 1]

    # Determine relative contribution of each TF
    N = np.linspace(0, 1, 50)
    I = np.linspace(0, 1, 50)

    n, i = np.meshgrid(N, I)
    f = np.zeros((len(N), len(I)))
    for j in range(len(N)):
        for k in range(len(I)):
            f[j,k] = get_f(C, n[j,k], i[j,k], best_model)
    
    # # save f to csv
    # np.savetxt("%s/f_%s.csv" % (results_dir, best_model), f, delimiter=",")
    # np.savetxt("%s/n_%s.csv" % (results_dir, best_model), n, delimiter=",")
    # np.savetxt("%s/i_%s.csv" % (results_dir, best_model), i, delimiter=",")
    

    # Plot minimum and maximum ifnb for each nfkb
    fig, ax = plt.subplots()
    ax.set_prop_cycle("color", plt.cm.viridis(np.linspace(0, 1, 5)))
    ax.plot(N, np.max(f, axis=0), label=r"Maximum IFN$\beta$", linewidth=3)
    ax.fill_between(N, np.min(f, axis=0), np.max(f, axis=0), alpha=0.2, label = "Contribution of IRF")
    ax.plot(N, np.min(f, axis=0), label=r"Minimum IFN$\beta$", linewidth=3)
    ax.set_xlabel(r"$NF\kappa B$")
    ax.set_ylabel(r"IFN$\beta$")
    ax.set_title("Model %s" % best_model)
    fig.legend(bbox_to_anchor=(1.23, 0.5))
    fig.savefig("%s/nfkb_vs_min_max_ifnb_%s.png" % (results_dir, best_model))

    # Plot minimum and maximum ifnb for each irf
    fig, ax = plt.subplots()
    ax.set_prop_cycle("color", plt.cm.viridis(np.linspace(0, 1, 5)))
    ax.plot(I, np.max(f, axis=1), label=r"Maximum IFN$\beta$", linewidth=3)
    ax.fill_between(I, np.min(f, axis=1), np.max(f, axis=1), alpha=0.2, label = "Contribution of NF$\kappa$B")
    ax.plot(I, np.min(f, axis=1), label=r"Minimum IFN$\beta$", linewidth=3)
    ax.set_xlabel(r"$IRF$")
    ax.set_ylabel(r"IFN$\beta$")
    ax.set_title("Model %s" % best_model)
    fig.legend(bbox_to_anchor=(1.25, 0.5))
    fig.savefig("%s/irf_vs_min_max_ifnb_%s.png" % (results_dir, best_model))

if __name__ == "__main__":
    main()