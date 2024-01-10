# Perform grid search over parameter space to get all possible values of IFNb
from two_IRF_model import *
import matplotlib.pyplot as plt
import pandas as pd
import os
import scipy.optimize as opt
import time
from multiprocessing import Pool
import argparse
plt.style.use("~/IFN_paper/src/theme_bw.mplstyle")

figures_dir = "grid_search/figures/"
results_dir = "grid_search/results/"
os.makedirs(results_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)
num_t_pars = 2
num_k_pars = 3

def calculate_ifnb(pars, data):
    t_pars, k_pars = pars[:num_t_pars], pars[num_t_pars:]
    N, I, P = data["NFkB"], data["IRF"], data["p50"]
    ifnb = [get_f(t_pars, k_pars, n, i, p) for n, i, p in zip(N, I, P)] # no scaling
    ifnb = np.array(ifnb)
    return ifnb

def calculate_grid(training_data, num_threads=40):
    # Define parameter space
    trg = np.arange(0, 1.1, 0.1)
    trgs = np.stack([trg for _ in range(num_t_pars)])
    krg = np.logspace(-2,2,11)
    krgs = np.stack([krg for _ in range(num_k_pars)])

    grid = np.meshgrid(*trgs, *krgs)
    grid = np.array(grid)
    inpt_shape = grid.shape
    grid = np.reshape(grid, (inpt_shape[0], np.prod(inpt_shape[1:]))).T

    kgrid = np.meshgrid(*krgs)
    kgrid = np.array(kgrid)
    k_shape = kgrid.shape
    kgrid = np.reshape(kgrid, (k_shape[0], np.prod(k_shape[1:]))).T

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g","--calc_grid", action="store_true")
    args = parser.parse_args()

    print("###############################################\n")
    print("Starting grid search\n")
    training_data = pd.read_csv("../data/p50_training_data.csv")
    print("Using the following training data:\n", training_data)
    print("Starting at %s" % time.ctime(), flush=True)
    model = "two_IRF"

    N = training_data["NFkB"]
    I = training_data["IRF"]
    P = training_data["p50"]
    beta = training_data["IFNb"]
    conditions = training_data["Stimulus"] + "_" + training_data["Genotype"]
    num_pts = len(N)
    len_training = len(N)

    num_pars = num_t_pars + num_k_pars
    num_threads = 40

    if args.calc_grid:
        ifnb_predicted, grid, kgrid = calculate_grid(training_data, num_threads=num_threads)

        np.savetxt("%s/%s_grid_pars.csv" % (results_dir, model), grid, delimiter=",")
        np.savetxt("%s/%s_grid_ifnb.csv" % (results_dir, model), ifnb_predicted, delimiter=",")
        np.savetxt("%s/%s_grid_kpars.csv" % (results_dir, model), kgrid, delimiter=",")

    pars = np.loadtxt("%s/%s_grid_pars.csv" % (results_dir, model), delimiter=",")
    ifnb_predicted = np.loadtxt("%s/%s_grid_ifnb.csv" % (results_dir, model), delimiter=",")
    kgrid = np.loadtxt("%s/%s_grid_kpars.csv" % (results_dir, model), delimiter=",")

    # # Calculate all states
    # print("Calculating all states", flush=True)
    # probabilities = {}
    # for stimulus, genotype in zip(["LPS", "pIC", "CpG"], ["WT", "WT", "WT"]):
    #     n, i, p = get_N_I_P(training_data, stimulus, genotype)
    #     print("Calculating state probabilities for %s %s" % (stimulus, genotype), flush=True)
    #     print("N=%.2f, I=%.2f, P=%.2f" % (n, i, p), flush=True)

    #     with Pool(num_threads) as p:
    #         results = p.starmap(get_state_prob, [(pars[0,:num_t_pars], kgrid[i], n, i, p) for i in range(len(kgrid))])

    #     state_probabilities = np.array(results)
    #     state_names = state_probabilities[0,1,0]
    #     state_probabilities = state_probabilities[:,0,:]

    #     probabilities[stimulus] = state_probabilities
        
    #     # Plot all state probabilities
    #     fig = plt.figure()
    #     for i in range(len(kgrid)):
    #         plt.plot(state_probabilities[i], color = "black", alpha=0.5)
    #     plt.xticks(range(len(state_names)), state_names, rotation=90)
    #     plt.ylabel("Probability")
    #     plt.title("Probability of each state for all %d k values, %s WT" % (len(kgrid), stimulus))
    #     # plt.ylim([0,1])
    #     plt.tight_layout()
    #     plt.savefig("%s/%s_all_state_probabilities.png" % (figures_dir, stimulus))

    # Plot all ifnb predictions
    print("Plotting all IFNb predictions", flush=True)
    fig = plt.figure()
    for i in range(len(pars)):
        plt.plot(ifnb_predicted[i], color = "black", alpha=0.5, label = "Predicted" if i == 0 else None)
    plt.plot(beta, color="red", label="Data", marker="o", linestyle="None")
    plt.xticks(range(len(conditions)), conditions, rotation=90)
    plt.ylabel(r"IFN$\beta$ production")
    plt.xlabel("Data point")
    plt.title(r"Predicted IFN$\beta$ for all %d parameter values" % len(pars))
    plt.legend()
    plt.tight_layout()
    plt.savefig("%s/%s_all_ifnb_predictions.png" % (figures_dir, model))
    print("Done plotting all IFNb predictions", flush=True)

    # Calculate residuals
    residuals = ifnb_predicted - np.stack([beta for _ in range(len(pars))])
    rmsd = np.sqrt(np.mean(residuals**2, axis=1))

    # Sort rmsd
    sorted_indices = np.argsort(rmsd)
    best_20 = sorted_indices[:20]

    # # Plot best 20 state probabilities
    # print("Plotting best 20 state probabilities", flush=True)
    # for stimulus in probabilities.keys():
    #     fig = plt.figure()
    #     for i in best_20:
    #         plt.plot(probabilities[stimulus][i], color = "black", alpha=0.5)
    #     plt.xticks(range(len(state_names)), state_names, rotation=90)
    #     plt.ylabel("Probability")
    #     plt.title("Probability of each state for best 20 parameter values by RMSD, %s WT" % stimulus)
    #     # plt.ylim([0,1])
    #     plt.tight_layout()
    #     plt.savefig("%s/%s_best_20_state_probabilities.png" % (figures_dir, stimulus))

    # Plot best 20 ifnb predictions
    print("Plotting best 20 IFNb predictions", flush=True)
    fig = plt.figure()
    for i in best_20:
        plt.plot(ifnb_predicted[i], color = "black", alpha=0.5, label = "Predicted" if i == 0 else None)
    plt.plot(beta, color="red", label="Data", marker="o", linestyle="None")
    plt.xticks(range(len(conditions)), conditions, rotation=90)
    plt.ylabel(r"IFN$\beta$ production")
    plt.xlabel("Data point")
    plt.title(r"Predicted IFN$\beta$ for best 20 parameter values by RMSD")
    plt.legend()
    plt.tight_layout()
    plt.savefig("%s/%s_best_20_ifnb_predictions.png" % (figures_dir, model))
    print("Done plotting best 20 IFNb predictions", flush=True)

if __name__ == "__main__":
    main()