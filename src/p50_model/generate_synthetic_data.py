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

def generate_synthetic_dataset(training_data, seed):
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
    synthetic_data = pd.DataFrame({"IRF": IRF_synthetic, "NFkB": NFkB_synthetic, **other_cols})
    return synthetic_data


def generate_synthetic_data(training_data, num_datasets, seed):
    num_pts = training_data.shape[0]
    synthetic_data = pd.DataFrame(columns=training_data.columns)
    # row where Stimulus == LPS and Genotype == WT

    # print("LPS WT row #%d\nLPS 2KO row #%d\npIC 2KO row #%d\npIC 3KO row #%d" % (lps_wt_loc, lps_irf_2ko, pic_irf_2ko, pic_irf_3ko))
    for i in range(num_datasets):
        seed += i
        synthetic_data = pd.concat([synthetic_data, generate_synthetic_dataset(training_data, seed+i)])

    # for i in range(num_pts):
    #     IRF = training_data.iloc[i]["IRF"]
    #     NFkB = training_data.iloc[i]["NFkB"]
    #     # geneerate num_datasets synthetic points within 5% of the original point
    #     IRF_synthetic = np.random.normal(max(IRF-0.05,0), min(IRF+0.05,1), size = num_datasets)
    #     NFkB_synthetic = np.random.normal(max(NFkB-0.05,0), min(NFkB+0.05,1), size = num_datasets)
    #     # keep other columns the same
    #     other_cols = training_data.iloc[i][2:]
    #     synthetic_data = pd.concat([synthetic_data, pd.DataFrame({"IRF": IRF_synthetic, "NFkB": NFkB_synthetic, **other_cols})])
    return synthetic_data

def main():
    # Load training data from "../data/p50_training_data.csv" with pandas
    training_data = pd.read_csv("../data/p50_training_data.csv")
    
    # Generate synthetic data
    num_datasets = 99
    seed = 5
    synthetic_data = generate_synthetic_data(training_data, num_datasets, seed)
    print(synthetic_data)

    all_data = pd.concat([training_data, synthetic_data])
    # Save to csv
    all_data.to_csv("../data/p50_training_data_plus_synthetic.csv", index=False)

    # Plot all data where p50=1
    fig, ax = plt.subplots()
    ax.set_prop_cycle('color', plt.cm.viridis(np.linspace(0, 1, num_datasets+1)))
    ax.scatter(synthetic_data.loc[synthetic_data["p50"]==1]["IRF"], synthetic_data.loc[synthetic_data["p50"]==1]["NFkB"], c=synthetic_data.loc[synthetic_data["p50"]==1]["IFNb"], s=10, alpha=0.5)
    ax.scatter(training_data.loc[training_data["p50"]==1]["IRF"], training_data.loc[training_data["p50"]==1]["NFkB"], c="#E85460", s=10, alpha=1)
    ax.set_xlabel("IRF")
    ax.set_ylabel(r"$NF\kappa B$")
    # ax.set_xlim(0,1)
    # ax.set_ylim(0,1)
    ax.set_title("All WT training data (with synthetic points)")
    fig.savefig("./figures/all_training_data_with_synthetic_WT.png")
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    plt.close()

    # Plot all data where p50=0
    fig, ax = plt.subplots()
    ax.set_prop_cycle('color', plt.cm.viridis(np.linspace(0, 1, num_datasets+1)))
    ax.scatter(synthetic_data.loc[synthetic_data["p50"]==0]["IRF"], synthetic_data.loc[synthetic_data["p50"]==0]["NFkB"], c=synthetic_data.loc[synthetic_data["p50"]==0]["IFNb"], s=10, alpha=0.5)
    ax.scatter(training_data.loc[training_data["p50"]==0]["IRF"], training_data.loc[training_data["p50"]==0]["NFkB"], c="#E85460", s=10, alpha=1)
    ax.set_xlabel("IRF")
    ax.set_ylabel(r"$NF\kappa B$")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title("All p50 KO training data (with synthetic points)")
    fig.savefig("./figures/all_training_data_with_synthetic_KO.png")
    plt.close()


if __name__ == "__main__":
    main()