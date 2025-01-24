from p50_model_distal_synergy import get_f
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import os
import time
from multiprocessing import Pool
import argparse
import seaborn as sns

mpl.rcParams["figure.dpi"] = 600
mpl.rcParams["font.sans-serif"] = "Arial"

plot_rc_pars = {"axes.labelsize":7, "font.size":6, "legend.fontsize":6, "xtick.labelsize":6, 
                                          "ytick.labelsize":6, "axes.titlesize":7, "legend.title_fontsize":7,
                                          "lines.markersize": 3, "axes.linewidth": 0.5,
                                            "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.minor.width": 0.5,
                                            "ytick.minor.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2,
                                            "xtick.minor.size": 1, "ytick.minor.size": 1, "legend.labelspacing": 0.2,
                                            "legend.columnspacing": 0.5, "legend.handletextpad": 0.5, "legend.handlelength": 1.5}
rc_pars={"xtick.major.pad": 1, "ytick.major.pad": 1, "legend.labelspacing": 0.2}
mpl.rcParams.update(rc_pars)

def get_pars(testing_data, row_num):
    row = testing_data.iloc[row_num]
    t_pars = row[row.index.str.startswith("t_")].values
    k_pars = row[row.index.str.startswith("k")].values
    h_pars = row[row.index.str.startswith("h")].values
    n = row["NFkB"]
    i = row["IRF"]
    p = row["p50"]
    c_par = None
    scaling=False

    pars = (t_pars, k_pars, n, i, p, c_par, h_pars, scaling)
    # print(pars)
    return pars

def calculate_values(num_threads, max_p50,num_p50_values, results_dir):
    start = time.time()
    print("Starting calculation.")
    # Load training data
    training_data = pd.read_csv("../data/p50_training_data.csv")
    # Filter for WT and remove unnecessary columns
    training_data = training_data.loc[training_data["Genotype"] == "WT",["Stimulus","IRF","NFkB"]]
    training_data["Stimulus"] = training_data["Stimulus"].replace("polyIC", "PolyIC")

    # Add p50 values to test
    p50_array = np.linspace(0,max_p50,num_p50_values)
    testing_data = pd.concat([training_data]*len(p50_array), ignore_index=True)
    testing_data["p50"] = p50_array.repeat(len(training_data))
    # print(testing_data)

    # Load best parameters
    best_fit_dir ="parameter_scan_dist_syn/results/"
    model = "p50_dist_syn"
    best_20_pars_df = pd.read_csv("%s/%s_best_fits_pars.csv" % (best_fit_dir, model))
    best_20_pars_df["h1"] = 3
    best_20_pars_df["h2"] = 1
    # print(best_20_pars_df)

    num_pars_repeats = len(testing_data)
    testing_data = testing_data.loc[np.repeat(testing_data.index, len(best_20_pars_df))].reset_index(drop=True)
    pars_repeated = pd.concat([best_20_pars_df]*num_pars_repeats).reset_index(drop=True)

    testing_data = pd.concat([testing_data, pars_repeated], axis=1)

    # print(len(testing_data))

    with Pool(num_threads) as p:
        results = p.starmap(get_f, [get_pars(testing_data, i) for i in range(len(testing_data))])

    # print(training_data)
    testing_data[r"IFN$\beta$"] = results
    # print(testing_data)
    end = time.time()
    print("Finished calculation after %.2f minutes" % ((end-start)/60))

    print("Saving results to %s" % results_dir, flush=True)
    testing_data.to_csv("%s/p50_abundance_params_results.csv" % results_dir)
    return testing_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c","--calculate", action="store_true")
    args = parser.parse_args()

    # Settings    
    num_threads = 40
    num_p50_values = 100
    max_p50 = 2

    # Directories
    figures_dir = "p50_abundance/figures/"
    results_dir = "p50_abundance/results/"
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    

    if args.calculate:
        testing_data = calculate_values(num_threads, max_p50,num_p50_values, results_dir)
    else:
        print("Loading results from %s/p50_abundance_params_results.csv" % results_dir)
        testing_data = pd.read_csv("%s/p50_abundance_params_results.csv" % results_dir)

    print("Making figure")
    with sns.plotting_context("paper", rc=plot_rc_pars):
        fig, ax = plt.subplots(figsize=(2,1.2))
        p = sns.lineplot(testing_data,x="p50",y=r"IFN$\beta$",hue="Stimulus", ax=ax, errorbar="sd")
        sns.despine()
        sns.move_legend(ax, bbox_to_anchor=(1,0.5), title=None, frameon=False, loc="center left", ncol=1)
        plt.tight_layout()
        plt.savefig("%s/predicted_ifnb_p50_abundance.png" % figures_dir)

    print("Done.")

if __name__ == "__main__":
    main()