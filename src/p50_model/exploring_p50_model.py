from p50_model import *
import matplotlib.pyplot as plt
import pandas as pd
import itertools
from multiprocessing import Pool
import os
import time
plt.style.use("~/IFN_paper/src/theme_bw.mplstyle")

def calcuate_f_p_range(params, N, I, P_range, model_name="B2", scaling=1):
    # Calculate f values for a range of p50 values and K values
    t_pars = [params["t%d" % i] for i in range(1,7)]
    C = params["C"]
    # K = params["K_i2"]
    p50_values = np.linspace(P_range[0], P_range[-1], 50)
    K_values = np.arange(0.2, 2.2, 0.2)
    if params["K_i2"] not in K_values:
        K_values = np.sort(np.append(K_values, params["K_i2"]))
    f_values = np.zeros((len(p50_values), len(K_values)))
    for i in range(len(K_values)):
        K = K_values[i]
        with Pool(30) as p:
            f_values[:,i] = p.starmap(get_f, [(t_pars, K, C, N, I, P, "B2", 1) for P in p50_values])

    # Normalize f_values by max f value
    if np.max(f_values) != 0:
        f_values = f_values / np.max(f_values)
    
    return f_values, p50_values, K_values

def make_p50_plot(params, N, I, P_range, filename):
    f_values, p50_values, K_values = calcuate_f_p_range(params, N, I, P_range, model_name="B2", scaling=1)
    dir = "../p50_model/figures/exploring_p50_model/"

    # Plot f as a function of p50 for all values of k
    best_fit_color="#575757"
    # plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(np.linspace(0,1,len(K_values)-1)))
    fig, ax = plt.subplots()
    ax.set_prop_cycle(plt.cycler("color", plt.cm.viridis(np.linspace(0,1,len(K_values)-1))))
    
    for i in range(len(K_values)):
        if K_values[i] != params["K_i2"]:
            plt.plot(p50_values, f_values[:,i], label="K=%.2f" % K_values[i])
        else:
            plt.plot(p50_values, f_values[:,i], color=best_fit_color, label="K=%.2f" % K_values[i])
    plt.xlabel("p50")
    plt.ylabel(r"IFN$\beta$ f-value, I=%.2f, N=%.2f" % (I, N))
    plt.title(r"IFN$\beta$ f-value as a function of p50, model B2")
    plt.legend()
    plt.savefig("%s/p50_plot_%s.png" % (dir, filename))

    # Create a boxplot of f values at 5 p50 values
    p50_subset = [int(np.round(len(p50_values)/5)*i) for i in range(5)] + [len(p50_values)-1]
    f_values_p50 = f_values[p50_subset,:]
    plt.figure()
    plt.boxplot(f_values_p50.T)
    plt.xticks([1,2,3,4,5,6], [("%.2f" % p50_values[i]) for i in p50_subset])
    plt.xlabel("p50")
    plt.ylabel(r"IFN$\beta$ mRNA")
    plt.title(r"IFN$\beta$ mRNA for different K values, model B2")
    plt.savefig("%s/p50_boxplot_%s.png" % (dir, filename))
    plt.close()
    
# Get final parameters
def get_params(file):
    params = {}
    with open(file) as f:
        for line in f:
            (key, val) = line.split(",")
            params[key] = float(val)
    return params

def make_contour_plot(f_values, N_values, I_values, filename, dir, title):
    plt.figure()
    plt.contourf(I_values, N_values, f_values, 100, cmap="RdYlBu_r")
    plt.colorbar(label=r"IFN$\beta$ mRNA")
    plt.xlabel(r"IRF")
    plt.ylabel(r"NF$\kappa$B")
    plt.title(title)
    plt.savefig("%s/contour_plot_%s.png" % (dir, filename))

def main():
    dir = "results/exploring_p50_model/"
    os.makedirs(dir, exist_ok=True)
    best_model = "B2"

    # Get final parameters
    print("Getting parameters")
    params = get_params("../p50_model/results/random_opt/ifnb_best_params_random_global.csv")
    t_pars = [params["t%d" % i] for i in range(1,7)]
    K = params["K_i2"]
    C = params["C"]


    # Make p50 plots for a range of I and N values
    print("Making p50 plots")
    P_range = [0, 2]
    for I in np.arange(0, 1.25, 0.25):
        for N in np.arange(0, 1.25, 0.25):
            filename = "I%.2f_N%.2f" % (I, N)
            make_p50_plot(params, N, I, P_range, filename)

    # Calculate f values for WT and KO for a range of N and I values
    N_list = np.linspace(0, 1, 100)
    I_list = np.linspace(0, 1, 100)
    P_list = {"WT": 1, "KO": 0}
    f_values = np.zeros((len(N_list)*len(I_list), len(P_list)))
    N_values = np.zeros((len(N_list)*len(I_list), len(P_list)))
    I_values = np.zeros((len(N_list)*len(I_list), len(P_list)))

    print("Starting multiprocessing")
    start = time.time()
    with Pool(30) as pl:
        for i in range(len(P_list)):
            p = list(P_list.keys())[i]
            print("Calculating f values for %s" % p)
            # get_f(t_pars, K, C, N, I, P, model_name=best_model, scaling=1)
            f_values[:,i] = pl.starmap(get_f, [(t_pars, K, C, N, I, P_list[p], best_model, 1) for N in N_list for I in I_list])
            N_values[:,i] = [N for N in N_list for I in I_list]
            I_values[:,i] = [I for N in N_list for I in I_list]
    end = time.time()
    print("Finished multiprocessing in %.2f minutes" % ((end-start)/60))

    f_values = f_values.reshape((len(N_list), len(I_list), len(P_list)))
    N_values = N_values.reshape((len(N_list), len(I_list), len(P_list)))
    I_values = I_values.reshape((len(N_list), len(I_list), len(P_list)))

    # Save all values
    np.save("%s/f_values.npy" % dir, f_values)
    np.save("%s/N_values.npy" % dir, N_values)
    np.save("%s/I_values.npy" % dir, I_values)

    # Calculate fold change KO vs WT
    f_values = np.load("%s/f_values.npy" % dir)
    N_values = np.load("%s/N_values.npy" % dir)
    I_values = np.load("%s/I_values.npy" % dir)

    # Replace 0 values in :,:,0 with 10e-10 to avoid divide by 0 error
    f_values[:,:,0][f_values[:,:,0]==0] = 10e-10
    fold_change = f_values[:,:,1] / f_values[:,:,0]

    # print N and I values that maximize fold change (top 5)
    print("Top 5 fold change values:")
    print(np.sort(fold_change.flatten())[-5:])
    print("N values:")
    print(N_values[:,:,0].flatten()[np.argsort(fold_change.flatten())[-5:]])
    print("I values:")
    print(I_values[:,:,0].flatten()[np.argsort(fold_change.flatten())[-5:]])

    # Plot fold change between WT and KO
    print("Plotting fold change heatmap")
    make_contour_plot(fold_change, N_values[:,:,0], I_values[:,:,0], 
                      "WT_KO_fold_change", dir, r"Fold change IFN$\beta$ w/ p50 KO, model B2")

    # Plot f values for WT and KO
    print("Plotting WT and KO heatmap")
    make_contour_plot(f_values[:,:,0], N_values[:,:,0], I_values[:,:,0],
                        "WT_f-values", dir, r"IFN$\beta$ mRNA in WT, model B2")
    make_contour_plot(f_values[:,:,1], N_values[:,:,1], I_values[:,:,1],
                        "KO_f-values", dir, r"IFN$\beta$ mRNA in KO, model B2")
    # Verify that N and I values are correct
    make_contour_plot(N_values[:,:,0], N_values[:,:,0], I_values[:,:,0],
                        "N-values_verify_coordinates", dir, r"NF$\kappa$B in WT, model B2")
    make_contour_plot(I_values[:,:,0], N_values[:,:,0], I_values[:,:,0],
                      "I-values_verify_coordinates", dir, r"IRF in WT, model B2")
    
    # Plot N and I values that maximize fold change
    print("Plotting N and I values that maximize fold change")
    plt.figure()
    plt.scatter(N_values[:,:,0].flatten()[np.argsort(fold_change.flatten())[-5:]],
                I_values[:,:,0].flatten()[np.argsort(fold_change.flatten())[-5:]])
    plt.xlabel(r"NF$\kappa$B")
    plt.ylabel(r"IRF")
    plt.title(r" Top 5 $NF\kappa B$ and $IRF$ values for fold change, model B2")
    plt.savefig("%s/N_I_values_fold_change.png" % dir)

    # Calculate log 2 fold change
    log2_fold_change = np.log2(fold_change)
    make_contour_plot(log2_fold_change, N_values[:,:,0], I_values[:,:,0],
                        "log2_fold_change", dir, r"Log2 fold change IFN$\beta$ w/ p50 KO, model B2")
    print("Finished")


    # Determine relative contribution of each TF
    N = np.linspace(0, 1, 50)
    I = np.linspace(0, 1, 50)

    n, i = np.meshgrid(N, I)
    f_WT = np.zeros((len(N), len(I)))
    f_KO = np.zeros((len(N), len(I)))
    for j in range(len(N)):
        for k in range(len(I)):
            f_WT[j,k] = get_f(t_pars, K, C, n[j,k], i[j,k], 1, best_model, 1)
            f_KO[j,k] = get_f(t_pars, K, C, n[j,k], i[j,k], 0, best_model, 1)
    f_WT_divide = f_WT[f_WT==0] = 10e-10
    f_FC = f_KO / f_WT_divide
    
    f_values = {"WT": f_WT, "KO": f_KO, "FC": f_FC}
    for f_name, f in f_values.items():
        # Plot minimum and maximum ifnb for each nfkb
        fig, ax = plt.subplots()
        ax.set_prop_cycle("color", plt.cm.viridis(np.linspace(0, 1, 5)))
        ax.plot(N, np.max(f, axis=0), label=r"Maximum IFN$\beta$", linewidth=3)
        ax.fill_between(N, np.min(f, axis=0), np.max(f, axis=0), alpha=0.2, label = "Contribution of IRF")
        ax.plot(N, np.min(f, axis=0), label=r"Minimum IFN$\beta$", linewidth=3)
        ax.set_xlabel(r"$NF\kappa B$")
        ax.set_ylabel(r"IFN$\beta$")
        ax.set_title("Model %s, %s" % (best_model, f_name))
        fig.legend(bbox_to_anchor=(1.23, 0.5))
        fig.savefig("%s/nfkb_vs_min_max_ifnb_%s_%s.png" % (dir, best_model, f_name))

        # Plot minimum and maximum ifnb for each irf
        fig, ax = plt.subplots()
        ax.set_prop_cycle("color", plt.cm.viridis(np.linspace(0, 1, 5)))
        ax.plot(I, np.max(f, axis=1), label=r"Maximum IFN$\beta$", linewidth=3)
        ax.fill_between(I, np.min(f, axis=1), np.max(f, axis=1), alpha=0.2, label = "Contribution of NF$\kappa$B")
        ax.plot(I, np.min(f, axis=1), label=r"Minimum IFN$\beta$", linewidth=3)
        ax.set_xlabel(r"$IRF$")
        ax.set_ylabel(r"IFN$\beta$")
        ax.set_title("Model %s, %s" % (best_model, f_name))
        fig.legend(bbox_to_anchor=(1.25, 0.5))
        fig.savefig("%s/irf_vs_min_max_ifnb_%s_%s.png" % (dir, best_model, f_name))


if __name__ == "__main__":
    main()