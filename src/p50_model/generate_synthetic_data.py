# Generate synthetic data based on p50 training data
# Fit p50 model to each set of synthetic data points
# Evaluate the parameter distributions of the fitted models from all synthetic data sets
# Plot the parameter distributions of the fitted models from all synthetic data sets

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import scipy.optimize as opt
import os
import sys
import time
from p50_model import *
from multiprocessing import Pool

def generate_synthetic_dataset(training_data, seed, starting_seed=5):
    lps_wt_loc = training_data.loc[(training_data["Stimulus"]=="LPS") & (training_data["Genotype"]=="WT")].index[0]
    lps_irf_2ko = training_data.loc[(training_data["Stimulus"]=="LPS") & (training_data["Genotype"]=="irf3irf7KO")].index[0]
    pic_irf_2ko = training_data.loc[(training_data["Stimulus"]=="polyIC") & (training_data["Genotype"]=="irf3irf7KO")].index[0]
    pic_irf_3ko = training_data.loc[(training_data["Stimulus"]=="polyIC") & (training_data["Genotype"]=="irf3irf5irf7KO")].index[0]

    IRF_vals = training_data["IRF"].values
    NFkB_vals = training_data["NFkB"].values
    IFNb_vals = training_data["IFNb"].values

    LPS_WT_val = 0
    LPS_IRF_2KO_val = 1
    PIC_IRF_2KO_val = 0
    PIC_IRF_3KO_val = 1
    # Generate synthetic dataset with length corresponding to the number of data points in the training data with +/- 5% noise around IRF and NFkB values
    std_err = 0.05
    std_dev = std_err
    
    i=0
    rng = np.random.default_rng(seed)
    while LPS_WT_val < LPS_IRF_2KO_val or PIC_IRF_2KO_val < PIC_IRF_3KO_val:
        IRF_synthetic = rng.normal(IRF_vals, std_dev)
        IRF_synthetic = np.clip(IRF_synthetic, 0, 1)

        NFkB_synthetic = rng.normal(NFkB_vals, std_dev)
        NFkB_synthetic = np.clip(NFkB_synthetic, 0, 1)


        LPS_WT_val = IRF_synthetic[lps_wt_loc] 
        LPS_IRF_2KO_val = IRF_synthetic[lps_irf_2ko]
        PIC_IRF_2KO_val = IRF_synthetic[pic_irf_2ko]
        PIC_IRF_3KO_val = IRF_synthetic[pic_irf_3ko]
        i += 1
        if i == 100:
            print("Failed to generate synthetic data set with IRF and NFkB values that match the training data")
            sys.exit(1)

    other_cols = training_data.iloc[:,2:]
    dataset_name = "synthetic_%d" % (seed - starting_seed)
    synthetic_data = pd.DataFrame({"IRF": IRF_synthetic, "NFkB": NFkB_synthetic, **other_cols, "Dataset": dataset_name})
    return synthetic_data


def generate_synthetic_data(training_data, num_datasets, original_seed):
    num_pts = training_data.shape[0]
    synthetic_data = pd.DataFrame(columns=training_data.columns)
    seed = original_seed
    for i in range(num_datasets):
        seed += i # oops, oh well
        synthetic_data = pd.concat([synthetic_data, generate_synthetic_dataset(training_data, seed+i, original_seed)])
    return synthetic_data

def main():
    # Load training data from "../data/p50_training_data.csv" with pandas
    training_data = pd.read_csv("../data/p50_training_data.csv")
    dataset_name = "original_data"
    training_data["Dataset"] = dataset_name

    # Generate synthetic data
    num_datasets = 99
    seed = 5
    synthetic_data = generate_synthetic_data(training_data, num_datasets, seed)
    # Rename datasets
    dataset_names = synthetic_data["Dataset"].unique()
    new_dataset_names = ["synthetic_%d" % i for i in range(num_datasets)]
    dataset_name_dict = dict(zip(dataset_names, new_dataset_names))
    synthetic_data["Dataset"] = synthetic_data["Dataset"].map(dataset_name_dict)
    print(synthetic_data)

    all_data = pd.concat([training_data, synthetic_data])
    # Save to csv
    all_data.to_csv("../data/p50_training_data_plus_synthetic.csv", index=False)

    font = {'size'   : 18}
    plt.rc('font', **font)

    markers = {"CpG": "o", "LPS": "s", "polyIC": "^"}

    # Plot all data where p50=1
    fig, ax = plt.subplots(dpi = 300)
    cmap = plt.cm.viridis
    map = ax.scatter(synthetic_data.loc[synthetic_data["p50"]==1]["IRF"], 
               synthetic_data.loc[synthetic_data["p50"]==1]["NFkB"], 
               c=synthetic_data.loc[synthetic_data["p50"]==1]["IFNb"], s=25, alpha=0.5, edgecolor="none", cmap=cmap)
    plt.colorbar(map, label=r"$IFN\beta$")
    for i in range(len(training_data["Stimulus"].unique())):
        stimulus = training_data["Stimulus"].unique()[i]
        marker = markers[stimulus]
        ax.scatter(training_data.loc[(training_data["p50"]==1) & (training_data["Stimulus"]==stimulus)]["IRF"], 
                   training_data.loc[(training_data["p50"]==1) & (training_data["Stimulus"]==stimulus)]["NFkB"], 
                   c="#E85460", s=25, alpha=1, marker=marker, label=stimulus)
    ax.set_xlabel("IRF")
    ax.set_ylabel(r"$NF\kappa B$")
    ax.set_title("All WT training data (with synthetic points)")
    ax.set_aspect("equal")
    fig.legend(bbox_to_anchor=(1.2,0.5))
    plt.tight_layout()
    ax.spines[['right', 'top']].set_visible(False)
    fig.savefig("./figures/all_training_data_with_synthetic_WT.png", bbox_inches="tight")
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    plt.close()

    # Plot all data where p50=0
    fig, ax = plt.subplots(dpi = 300)
    map = ax.scatter(synthetic_data.loc[synthetic_data["p50"]==0]["IRF"], 
               synthetic_data.loc[synthetic_data["p50"]==0]["NFkB"], 
               c=synthetic_data.loc[synthetic_data["p50"]==0]["IFNb"], s=25, alpha=0.5, edgecolor="none", cmap=cmap)
    plt.colorbar(map, label=r"$IFN\beta$")
    for i in range(len(training_data["Stimulus"].unique())):
        stimulus = training_data["Stimulus"].unique()[i]
        marker = markers[stimulus]
        ax.scatter(training_data.loc[(training_data["p50"]==0) & (training_data["Stimulus"]==stimulus)]["IRF"], 
                   training_data.loc[(training_data["p50"]==0) & (training_data["Stimulus"]==stimulus)]["NFkB"], 
                   c="#E85460", s=25, alpha=1, marker=marker, label=stimulus)
    ax.set_xlabel("IRF")
    ax.set_ylabel(r"$NF\kappa B$")
    ax.set_title("All KO training data (with synthetic points)")
    ax.set_aspect("equal")
    fig.legend(bbox_to_anchor=(1.2,0.5))
    plt.tight_layout()
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    fig.savefig("./figures/all_training_data_with_synthetic_KO.png", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()