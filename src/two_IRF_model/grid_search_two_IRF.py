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
    krg = np.logspace(-2,3,12)
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

def calc_state_prob(k_pars, N, I, P):
    # print(N, I, P, flush=True)
    t_pars = [1,1]
    probabilities, state_names = get_state_prob(t_pars, k_pars, N, I, P)
    return probabilities, state_names

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g","--calc_grid", action="store_true")
    parser.add_argument("-s","--calc_states", action="store_true")
    parser.add_argument("-n","--no_plot", action="store_false") 
    args = parser.parse_args()

    start_start = time.time()

    print("###############################################\n")
    print("Starting at %s\n" % time.ctime(), flush=True)

    print("Starting grid search\n")
    training_data = pd.read_csv("../data/p50_training_data.csv")
    print("Using the following training data:\n", training_data)
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
        print("Calculating grid ON", flush=True)
        ifnb_predicted, grid, kgrid = calculate_grid(training_data, num_threads=num_threads)

        np.savetxt("%s/%s_grid_pars.csv" % (results_dir, model), grid, delimiter=",")
        np.savetxt("%s/%s_grid_ifnb.csv" % (results_dir, model), ifnb_predicted, delimiter=",")
        np.savetxt("%s/%s_grid_kpars.csv" % (results_dir, model), kgrid, delimiter=",")

    pars = np.loadtxt("%s/%s_grid_pars.csv" % (results_dir, model), delimiter=",")
    ifnb_predicted = np.loadtxt("%s/%s_grid_ifnb.csv" % (results_dir, model), delimiter=",")
    kgrid = np.loadtxt("%s/%s_grid_kpars.csv" % (results_dir, model), delimiter=",")

    # Calculate all states
    extra_training_data = pd.DataFrame({"Stimulus":"basal", "Genotype":"WT", "IRF":0.05, "NFkB":0.05, "p50":1}, index=[0])
    training_data_extended = pd.concat([training_data, extra_training_data], ignore_index=True)
    
    stimuli = training_data_extended["Stimulus"].unique()
    genotypes = ["WT" for _ in range(len(stimuli))]

    if args.calc_states:
        print("Calculating state probabilities ON", flush=True)
        for stimulus, genotype in zip(stimuli, genotypes):
            nfkb, irf, p50 = get_N_I_P(training_data_extended, stimulus, genotype)
            print("Calculating state probabilities for %s %s" % (stimulus, genotype), flush=True)
            print("N=%.2f, I=%.2f, P=%.2f" % (nfkb, irf, p50), flush=True)
        
            with Pool(num_threads) as p:
                results = p.starmap(calc_state_prob, [(tuple(kgrid[i]), nfkb, irf, p50) for i in range(len(kgrid))])
        
            state_names = results[0][1]
            state_probabilities = np.array([x[0] for x in results])

            np.savetxt("%s/%s_%s_%s_state_probabilities.csv" % (results_dir, model, stimulus, genotype), state_probabilities, delimiter=",")

        np.savetxt("%s/%s_state_names.csv" % (results_dir, model), state_names, delimiter=",", fmt="%s")

    if args.no_plot:
        print("Plotting ON", flush=True)
        probabilities = {}
        state_names = np.loadtxt("%s/%s_state_names.csv" % (results_dir, model), delimiter=",", dtype=str)
        for stimulus, genotype in zip(stimuli, genotypes):
            state_probabilities = np.loadtxt("%s/%s_%s_%s_state_probabilities.csv" % (results_dir, model, stimulus, genotype), delimiter=",")
            probabilities[stimulus] = state_probabilities
            # Plot all state probabilities
            fig = plt.figure()
            for i in range(len(kgrid)):
                plt.plot(state_probabilities[i], color = "black", alpha=0.5)
            plt.xticks(range(len(state_names)), state_names, rotation=90)
            plt.ylabel("Probability")
            plt.title("Probability of each state for all %d k values, %s WT" % (len(kgrid), stimulus))
            # plt.ylim([0,1])
            plt.tight_layout()
            plt.savefig("%s/%s_all_state_probabilities.png" % (figures_dir, stimulus))
            plt.close()

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
        best_20_params = pars[best_20]
        best_20_kpars = best_20_params[:,num_t_pars:]
        best_20_k_indices = np.array([np.where(np.all(kgrid == kp, axis=1))[0][0] for kp in best_20_kpars])


        # Plot best 20 state probabilities
        print("Plotting best 20 state probabilities", flush=True)
        for stimulus in probabilities.keys():
            fig = plt.figure()
            for i in best_20_k_indices:
                plt.plot(probabilities[stimulus][i], color = "black", alpha=0.5)
            plt.xticks(range(len(state_names)), state_names, rotation=90)
            plt.ylabel("Probability")
            plt.title("Probability of each state for best 20 parameter values by RMSD, %s WT" % stimulus)
            # plt.ylim([0,1])
            plt.tight_layout()
            plt.savefig("%s/%s_best_20_state_probabilities.png" % (figures_dir, stimulus))
            plt.close()

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
        plt.close()
        print("Done plotting best 20 IFNb predictions", flush=True)

        # Plot best 20 parameters
        print("Plotting best 20 parameters", flush=True)
        fig, ax = plt.subplots(1,2, figsize=(10,5))
        for i in best_20:
            ax[0].plot(range(num_t_pars), pars[i,:num_t_pars], color = "black", alpha=0.5) 
            ax[1].plot(range(num_k_pars), pars[i,num_t_pars:], color = "black", alpha=0.5)
        ax[0].set_xticks(range(num_t_pars))
        ax[0].set_xticklabels(["t1", "t2"])
        ax[1].set_xticks(range(num_k_pars))
        ax[1].set_xticklabels(["k1", "k2", "kp"])
        plt.ylabel("Parameter value")
        plt.xlabel("Parameter")
        plt.title("Parameter values for best 20 parameter values by RMSD")
        plt.tight_layout()
        plt.savefig("%s/%s_best_20_parameters.png" % (figures_dir, model))
        plt.close()

        # Calculate RMSD to IRF and p50 points: 0,1,2,3,5,6,8,9
        print("Calculating RMSD to IRF and p50 points training data points", flush=True)
        relevant_points = [0,1,2,3,5,6,8,9]
        rmsd_to_IRF_p50 = np.sqrt(np.mean(residuals[:,relevant_points]**2, axis=1))
        sorted_indices = np.argsort(rmsd_to_IRF_p50)
        best_20 = sorted_indices[:20]
        best_20_params = pars[best_20]
        np.savetxt("%s/%s_best_20_parameters_IRF_p50.csv" % (results_dir, model), best_20_params, delimiter=",")
        best_20_predicted = ifnb_predicted[best_20]
        np.savetxt("%s/%s_best_20_ifnb_predictions_IRF_p50.csv" % (results_dir, model), best_20_predicted, delimiter=",")
        best_20_kpars = best_20_params[:,num_t_pars:]
        best_20_k_indices = np.array([np.where(np.all(kgrid == kp, axis=1))[0][0] for kp in best_20_kpars])
        print("Saved best 20 parameters and IFNb predictions by RMSD to IRF and p50 points", flush=True)

        # Plot best 20 state probabilities
        print("Plotting best 20 state probabilities by RMSD to IRF and p50 points", flush=True)
        for stimulus in probabilities.keys():
            fig = plt.figure()
            for i in best_20_k_indices:
                plt.plot(probabilities[stimulus][i], color = "black", alpha=0.5)
            plt.xticks(range(len(state_names)), state_names, rotation=90)
            plt.ylabel("Probability")
            plt.title("Probability of each state for best 20 parameter values by RMSD to IRF and p50 points, %s WT" % stimulus)
            # plt.ylim([0,1])
            plt.tight_layout()
            plt.savefig("%s/%s_best_20_state_probabilities_IRF_p50.png" % (figures_dir, stimulus))
            plt.close()

        # Plot best 20 ifnb predictions
        print("Plotting best 20 IFNb predictions by RMSD to IRF and p50 points", flush=True)
        fig = plt.figure()
        for i in best_20:
            plt.plot(ifnb_predicted[i], color = "black", alpha=0.5, label = "Predicted" if i == 0 else None)
        plt.plot(beta, color="red", label="Data", marker="o", linestyle="None")
        plt.xticks(range(len(conditions)), conditions, rotation=90)
        plt.ylabel(r"IFN$\beta$ production")
        plt.xlabel("Data point")
        plt.title(r"Predicted IFN$\beta$ for best 20 parameter values by RMSD to IRF and p50 points")
        plt.legend()
        plt.tight_layout()
        plt.savefig("%s/%s_best_20_ifnb_predictions_IRF_p50.png" % (figures_dir, model))
        plt.close()

        # Plot best 20 parameters
        print("Plotting best 20 parameters by RMSD to IRF and p50 points", flush=True)
        fig, ax = plt.subplots(1,2, figsize=(10,5))
        for i in best_20:
            ax[0].plot(range(num_t_pars), pars[i,:num_t_pars], color = "black", alpha=0.5) 
            ax[1].plot(range(num_k_pars), pars[i,num_t_pars:], color = "black", alpha=0.5)
        ax[0].set_xticks(range(num_t_pars))
        ax[0].set_xticklabels(["t1", "t2"])
        ax[1].set_xticks(range(num_k_pars))
        ax[1].set_xticklabels(["k1", "k2", "kp"])
        plt.ylabel("Parameter value")
        plt.xlabel("Parameter")
        plt.title("Parameter values for best 20 parameter values by RMSD (No NFkB KO)")
        plt.tight_layout()
        plt.savefig("%s/%s_best_20_parameters_IRF_p50.png" % (figures_dir, model))
        plt.close()

        # Separate pars from top 20 where t1<0.4 or t1>0.4
        print("Separating parameters where t1<0.4 or t1>0.4", flush=True)
        t1_small_inds = np.where(best_20_params[:,0] < 0.4)[0]
        t1_small_params = best_20_params[t1_small_inds]
        t1_large_inds = np.where(best_20_params[:,0] > 0.4)[0]
        t1_large_params = best_20_params[t1_large_inds]

        # Plot pars and color by t1 value
        print("Plotting parameters and color by t1 value", flush=True)
        # use two viridis colors
        colors = ["#440154FF", "#21908CFF"]
        fig, ax = plt.subplots(1,2, figsize=(10,5))
        for i in t1_small_inds:
            ax[0].plot(range(num_t_pars), best_20_params[i,:num_t_pars], color = colors[0], alpha=0.5)
            ax[1].plot(range(num_k_pars), best_20_params[i,num_t_pars:], color = colors[0], alpha=0.5, label="t1<0.4" if i == t1_small_inds[0] else None)
        for i in t1_large_inds:
            ax[0].plot(range(num_t_pars), best_20_params[i,:num_t_pars], color = colors[1], alpha=0.5)
            ax[1].plot(range(num_k_pars), best_20_params[i,num_t_pars:], color = colors[1], alpha=0.5, label="t1>0.4" if i == t1_large_inds[0] else None)
        ax[0].set_xticks(range(num_t_pars))
        ax[0].set_xticklabels(["t1", "t2"])
        ax[1].set_xticks(range(num_k_pars))
        ax[1].set_xticklabels(["k1", "k2", "kp"])
        ax[1].legend(bbox_to_anchor=(1.05, 0.5), loc="center left")
        ax[0].set_ylabel("Parameter value")
        # fig.text(0.5, 0.04, "Parameter", ha="center")
        plt.suptitle("Parameter values for best 20 parameter values by RMSD (No NFkB KO)")
        plt.tight_layout()
        plt.savefig("%s/%s_best_20_parameters_IRF_p50_t1_small_large.png" % (figures_dir, model))

    end_end = time.time()
    t = end_end - start_start
    print("\n###############################################")
    if t < 60*60:
        print("Total time elapsed: %.2f minutes" % (t/60), flush=True)
    else:
        print("Total time elapsed: %.2f hours" % (t/3600), flush=True)

if __name__ == "__main__":
    main()