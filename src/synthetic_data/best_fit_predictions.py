# Using best-fit parameters from p50 model, determine how well IFNb expression is fit with
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool
from p50_model_force_t import *

def get_N_I_P(data, stimulus, genotype):
    row = data.loc[(data["Stimulus"] == stimulus) & (data["Genotype"] == genotype)]
    N = row["NFkB"].values[0]
    I = row["IRF"].values[0]
    P = row["p50"].values[0]
    return N, I, P

def find_residual(std_err, best_pars_df, num_t_pars, num_k_pars, h, num_threads, figures_dir):
    synthetic_data = pd.read_csv("../data/p50_training_data_plus_synthetic_e%.1fpct.csv" % (std_err*100))

    # predict IFNb expression using best-fit parameters
    for i in range(len(best_pars_df)):
        t_pars = best_pars_df.iloc[i, :num_t_pars]
        k_pars = best_pars_df.iloc[i, num_t_pars:num_t_pars+num_k_pars]

        with Pool(num_threads) as p:
            results = p.starmap(get_f, [(t_pars, k_pars, N, I, P, None, h, False) for N, I, P in zip(synthetic_data["NFkB"], 
                                                                                                    synthetic_data["IRF"], synthetic_data["p50"])])

        synthetic_data["IFNb%d" % i] = results

    # pivot all the IFNb predictions
    synthetic_data = synthetic_data.melt(id_vars=["Stimulus", "Genotype", "IRF", "NFkB", "IFNb", "Dataset", "p50"], 
                                        var_name="par_set", value_name="IFNb_pred")
    # Average the predictions
    synthetic_data = synthetic_data.groupby(["Stimulus", "Genotype", "Dataset"]).mean(numeric_only=True).reset_index()
    synthetic_data["residual"] = synthetic_data["IFNb_pred"] - synthetic_data["IFNb"]

    synthetic_data["Genotype"] = synthetic_data["Genotype"].replace("relacrelKO", r"NFκBko")
    synthetic_data["Genotype"] = synthetic_data["Genotype"].replace("p50KO", "p50ko")
    synthetic_data["Genotype"] = synthetic_data["Genotype"].replace("irf3irf7KO", "IRF3/7ko")
    synthetic_data["Genotype"] = synthetic_data["Genotype"].replace("irf3irf5irf7KO", "IRF3/5/7ko")
    synthetic_data["Stimulus"] = synthetic_data["Stimulus"].replace("polyIC", "PolyIC")
    synthetic_data["Stimulus"] = synthetic_data["Stimulus"].replace("basal", "Basal")
    stimuli_levels = ["Basal", "CpG", "LPS", "PolyIC"]
    genotypes_levels = ["WT","p50ko", "IRF3/7ko", "IRF3/5/7ko", r"NFκBko"]
    synthetic_data = synthetic_data.sort_values(by=["Stimulus", "Genotype"])
    synthetic_data["Condition"] = synthetic_data["Stimulus"] + " " + synthetic_data["Genotype"]
    synthetic_data["Condition"] = pd.Categorical(synthetic_data["Condition"], ordered=True)

    # print(synthetic_data)

    # Plot residuals vs condition
    # col = sns.cubehelix_palette(light=0.95, dark=0, reverse=True, rot=0.4,start=-.2, hue=0.6, n_colors=4)[2]
    data_color = "#FA4B5C"
    fig, ax = plt.subplots()
    p = sns.stripplot(data=synthetic_data, x="Condition", y="residual", ax=ax, dodge=True, alpha=0.5, color="black")
    ax.tick_params(axis='x', rotation=45)
    sns.despine()
    plt.savefig("%s/residuals_vs_condition_error%.1fpct.png" % (figures_dir, std_err*100), dpi=300, bbox_inches = "tight")
    plt.close()

    synthetic_data_copy = synthetic_data.copy()
    synthetic_data_copy = synthetic_data_copy.melt(id_vars=["Dataset","Condition"], value_vars=["IFNb", "IFNb_pred"],
                                        var_name="ifnb_type", value_name="IFNb_val")
    synthetic_data_copy["ifnb_type"] = synthetic_data_copy["ifnb_type"].replace("IFNb", "Experimental")
    synthetic_data_copy["ifnb_type"] = synthetic_data_copy["ifnb_type"].replace("IFNb_pred", "Predicted")
    synthetic_data_copy["ifnb_type"] = pd.Categorical(synthetic_data_copy["ifnb_type"], categories=["Predicted", "Experimental"], ordered=True)
    synthetic_data_copy = synthetic_data_copy.sort_values(by=["ifnb_type", "Condition"])

    color_dict = {"Predicted": "black", "Experimental": data_color}
    # Plot IFNb vs condition
    fig, ax = plt.subplots()
    p = sns.stripplot(data=synthetic_data_copy, x="Condition", y="IFNb_val", ax=ax, alpha=0.5, hue="ifnb_type", palette=color_dict, 
                      hue_order=["Predicted", "Experimental"])
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylabel(r"IFN$\beta$")
    ax.set_xlabel("")
    sns.despine()
    sns.move_legend(ax, bbox_to_anchor=(0.5, 1), title=None, frameon=False, loc="lower center", ncol=2)
    plt.savefig("%s/IFNb_vs_condition_error%.1fpct.png" % (figures_dir, std_err*100), dpi=300, bbox_inches = "tight")
    plt.close()

    return synthetic_data

def main():
    
    num_threads = 40
    h = [3, 1]
    figures_dir = "figures/"
    results_dir = "results/"
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Load best-fit parameters
    force_t_dir = "../p50_model/parameter_scan_force_t/results/"
    best_pars_df = pd.read_csv("%s/p50_force_t_best_fits_pars.csv" % (force_t_dir))
    # print(best_pars_df)
    num_t_pars = ["t" in col for col in best_pars_df.columns].count(True)
    num_k_pars = ["k" in col for col in best_pars_df.columns].count(True)
    print("Number of t parameters: %d, Number of k parameters: %d" % (num_t_pars, num_k_pars))

    # Load synthetic data
    all_data = pd.DataFrame()

    # Load errors
    e_vals = np.loadtxt("error_percentages_tested.txt")

    # Predict IFNb and calculate residuals
    for e in e_vals:
        std_err = e/100
        print("Finding residuals for standard error %.2f" % std_err, flush=True)
        synthetic_data = find_residual(std_err, best_pars_df, num_t_pars, num_k_pars, h, num_threads, figures_dir)
        synthetic_data["error"] = std_err
        all_data = pd.concat([all_data, synthetic_data])

    all_data.to_csv("%s/p50_force_t_residuals_synthetic_data.csv" % results_dir, index=False)

    # Count for each condition how many datasets have residual < 0.1
    cutoffs = [0.1]
    for r in cutoffs:
        print("Counting number of acceptable fits for residual < %.2f" % r, flush=True)
        all_data["acceptable_fit"] = all_data["residual"] < r
        counts = all_data.copy()
        counts = counts.groupby(["Condition", "error"]).agg({"acceptable_fit": "sum"}).reset_index()
        counts["err_pct"] = np.round(counts["error"]*100, 1)
        # print(counts)

        # Plot # acceptable fits vs error
        fig, ax = plt.subplots()
        sns.stripplot(data=counts, x="err_pct", y="acceptable_fit", ax=ax, dodge=True, color="black")
        sns.despine()
        ax.set_ylim(0, 101)
        ax.set_xlabel("Standard Error (%)")
        ax.set_ylabel("Percent Acceptable Fits (Residual < %.2f)" % r)
        plt.savefig("%s/acceptable_fits_vs_error_cutoff%.2f.png" % (figures_dir, r), dpi=300, bbox_inches = "tight")
        plt.close()

if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print("Time elapsed: %.2f minutes" % ((end-start)/60))