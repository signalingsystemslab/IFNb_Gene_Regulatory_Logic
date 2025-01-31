from p50_model_distal_synergy import get_f, get_contribution, get_state_prob
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
stim_pal=  {"LPS": "#86BB87", "CpG": "#5D9FB5", "PolyIC": "#BA4961"}

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
    return pars

def get_renaming_dict(state_names_old_names):
    state_names = pd.DataFrame(state_names_old_names, columns=["state_only"])
    state_names["state_new"] = state_names["state_only"].replace("$IRF$", "$IRF_2$")
    state_names["state_new"] = state_names["state_new"].replace("none", "Unbound")
    state_names["state_new"] = state_names["state_new"].replace("$IRF_G$", "$IRF_1$")
    state_names["state_new"] = state_names["state_new"].replace("$IRF\cdot IRF_G$", "$IRF_1& IRF_2$")
    state_names["state_new"] = state_names["state_new"].replace("$IRF\cdot NF\kappa B$", "$IRF_2& NF\kappa B$")
    state_names["state_new"] = state_names["state_new"].replace("$IRF_G\cdot NF\kappa B$", "$IRF_1& NF\kappa B$")
    state_names["state_new"] = state_names["state_new"].replace("$IRF\cdot NF\kappa B\cdot p50$", "$IRF_2& NF\kappa B& p50$")
    state_names["state_new"] = state_names["state_new"].replace("$IRF\cdot IRF_G\cdot NF\kappa B$", "$IRF_1& IRF_2& NF\kappa B$")
    state_names["state_new"] = state_names["state_new"].replace("$IRF\cdot p50$", "$IRF_2& p50$")
    state_names["state_new"] = state_names["state_new"].replace("$NF\kappa B\cdot p50$", "$NF\kappa B& p50$")
    state_name_order = ["Unbound", r"$IRF_1$", r"$IRF_2$", r"$IRF_2& p50$", r"$NF\kappa B$",
                        r"$NF\kappa B& p50$", r"$p50$", r"$IRF_1& IRF_2$", r"$IRF_1& NF\kappa B$",
                        r"$IRF_2& NF\kappa B$", r"$IRF_2& NF\kappa B& p50$", r"$IRF_1& IRF_2& NF\kappa B$"]
    state_names["state_new"] = pd.Categorical(state_names["state_new"], categories=state_name_order, ordered=True)
    state_names = state_names.sort_values("state_new")
    state_names["state"] = state_names["state_new"].astype(str)
    # Renaming the states
    state_name_dict = state_names.set_index('state_only')["state"].to_dict()
    # Make all values raw strings
    state_name_dict = {k: r"%s" % v for k, v in state_name_dict.items()}
    return state_name_dict

def calculate_values(num_threads, max_p50,num_p50_values, results_dir, genotype="WT", more_info=""):
    if len(more_info)>0:
        more_info = "_" + more_info

    start = time.time()
    print("Starting calculation.")
    # Load training data
    training_data = pd.read_csv("../data/p50_training_data.csv")
    # Filter for WT and remove unnecessary columns
    training_data = training_data.loc[training_data["Genotype"] == "WT",["Stimulus","IRF","NFkB"]]
    if genotype == "nfkbKO":
        training_data["NFkB"] = 0
    training_data["Stimulus"] = training_data["Stimulus"].replace("polyIC", "PolyIC")

    # Add p50 values to test
    p50_array = np.linspace(0,max_p50,num_p50_values)
    testing_data = pd.concat([training_data]*len(p50_array), ignore_index=True)
    testing_data["p50"] = p50_array.repeat(len(training_data))

    # Load best parameters
    best_fit_dir ="parameter_scan_dist_syn/results/"
    model = "p50_dist_syn"
    best_20_pars_df = pd.read_csv("%s/%s_best_fits_pars.csv" % (best_fit_dir, model))
    best_20_pars_df["h1"] = 3
    best_20_pars_df["h2"] = 1
    best_20_pars_df.reset_index(drop=False,inplace=True,names="par_number")

    num_pars_repeats = len(testing_data)
    testing_data = testing_data.loc[np.repeat(testing_data.index, len(best_20_pars_df))].reset_index(drop=True)
    pars_repeated = pd.concat([best_20_pars_df]*num_pars_repeats).reset_index(drop=True)

    testing_data = pd.concat([testing_data, pars_repeated], axis=1)

    ## Get f values
    with Pool(num_threads) as p:
        results = p.starmap(get_f, [get_pars(testing_data, i) for i in range(len(testing_data))])

    testing_data[r"IFN$\beta$"] = results
    end = time.time()
    print("Finished calculation of f values after %.2f minutes" % ((end-start)/60))

    print("Saving results to %s" % results_dir, flush=True)
    testing_data.to_csv("%s/p50_abundance_params_results%s.csv" % (results_dir, more_info))

    ## Get contributions
    with Pool(num_threads) as p:
        results = p.starmap(get_contribution, [get_pars(testing_data, i)[:-1] for i in range(len(testing_data))])
    # output is ([contributions], [state_names]), ... ([contributions], [state_names])
    state_names = results[0][1]

    end = time.time()
    print("Finished calculation of contrib values after %.2f minutes" % ((end-start)/60))

    # Reshape results
    results = np.array([results[i][0] for i in range(len(results))])
    results = pd.DataFrame(results, columns = state_names)

    # Rename state names
    state_name_dict = get_renaming_dict(state_names)
    state_names = list(state_name_dict.values())
    results.rename(columns=state_name_dict, inplace=True)
    contrib_data = pd.concat([testing_data, results], axis=1)

    contrib_data.to_csv("%s/p50_abundance_params_contributions%s.csv" % (results_dir, more_info))

    ## Get state probabilities
    with Pool(num_threads) as p:
        results = p.starmap(get_state_prob, [get_pars(testing_data, i)[:-1] for i in range(len(testing_data))])

    end = time.time()
    print("Finished calculation of state probs after %.2f minutes" % ((end-start)/60))

    # Reshape results
    results = np.array([results[i][0] for i in range(len(results))])
    results = pd.DataFrame(results, columns = state_names)
    state_probs = pd.concat([testing_data, results], axis=1)

    state_probs.to_csv("%s/p50_abundance_params_probabilities%s.csv" % (results_dir, more_info))

    return testing_data, contrib_data, state_probs

def make_f_figure(testing_data, figures_dir, more_info="",ylab=None):
    if len(more_info) > 0:
        more_info = "_" + more_info
    with sns.plotting_context("paper", rc=plot_rc_pars):
        fig, ax = plt.subplots(figsize=(2,1.2))
        p = sns.lineplot(testing_data,x="p50",y=r"IFN$\beta$",hue="Stimulus", ax=ax, errorbar="sd", palette=stim_pal)
        if ylab is not None:
            p.set_ylabel(r"IFN$\beta$ " + ylab)
        sns.despine()
        sns.move_legend(ax, bbox_to_anchor=(1,0.5), title=None, frameon=False, loc="center left", ncol=1)
        plt.tight_layout()
        plt.savefig("%s/predicted_ifnb_p50_abundance%s.png" % (figures_dir, more_info))

def make_contrib_figure(contrib_data, figures_dir, more_info="",ylab=None):
    if len(more_info) > 0:
        more_info = "_" + more_info

    with sns.plotting_context("paper", rc=plot_rc_pars):
        ncols=3
        p = sns.FacetGrid(contrib_data, col="State", col_wrap=ncols, sharex=True, sharey=True, height=1, aspect=1)
        p.map_dataframe(sns.lineplot, x="p50", y="contribution", hue="Stimulus", errorbar="sd", palette=stim_pal)
        p.set_titles("{col_name}")

        if ylab is not None:
            p.set_ylabels(r"IFN$\beta$ " + ylab)
        else:
            p.set_ylabels(r"IFN$\beta$")
        p.set_xlabels("[p50:p50]")

        sns.despine()
        plt.tight_layout()
        plt.savefig("%s/predicted_ifnb_p50_contributions%s.png" % (figures_dir, more_info))

def make_probs_figure(probs_data, figures_dir, more_info="",ylab=None):
    if len(more_info) > 0:
        more_info = "_" + more_info

    with sns.plotting_context("paper", rc=plot_rc_pars):
        ncols=4
        p = sns.FacetGrid(probs_data, col="State", col_wrap=ncols, sharex=True, sharey=True, height=1, aspect=1)
        p.map_dataframe(sns.lineplot, x="p50", y="probability", hue="Stimulus", errorbar="sd", palette=stim_pal)
        p.set_titles("{col_name}")

        if ylab is not None:
            p.set_ylabels("Probability " + ylab)
        else:
            p.set_ylabels("Probability")
        p.set_xlabels("[p50:p50]")

        sns.despine()
        plt.tight_layout()
        plt.savefig("%s/predicted_ifnb_p50_probability%s.png" % (figures_dir, more_info))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c","--calculate", action="store_true")
    parser.add_argument("-n","--nfkbKO", action="store_true")
    parser.add_argument("-m","--max_p50", type=int, default=1)
    args = parser.parse_args()

    # Settings    
    num_threads = 40
    num_p50_values = 101
    max_p50 = args.max_p50

    # Directories
    figures_dir = "p50_abundance/figures/"
    results_dir = "p50_abundance/results/"
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    

    if args.calculate:
        testing_data, contrib_data, state_prob_data = calculate_values(num_threads, max_p50,num_p50_values, results_dir)
    else:
        print("Loading results from %s/p50_abundance_params_results.csv" % results_dir)
        testing_data = pd.read_csv("%s/p50_abundance_params_results.csv" % results_dir)
        contrib_data = pd.read_csv("%s/p50_abundance_params_contributions.csv" % (results_dir))
        state_prob_data = pd.read_csv("%s/p50_abundance_params_probabilities.csv" % (results_dir))

    print("Making figures")
    ## Making f value figure w/ legend
    testing_data["Stimulus"] = pd.Categorical(testing_data["Stimulus"], categories = ["CpG", "LPS", "PolyIC"], ordered=True)
    make_f_figure(testing_data, figures_dir, more_info="0_to_1")

    ## Making contributions figure
    # Combine values from inactive states
    state_names = ["Unbound", r"$IRF_1$", r"$IRF_2$", r"$IRF_2& p50$", r"$NF\kappa B$",
                        r"$NF\kappa B& p50$", r"$p50$", r"$IRF_1& IRF_2$", r"$IRF_1& NF\kappa B$",
                        r"$IRF_2& NF\kappa B$", r"$IRF_2& NF\kappa B& p50$", r"$IRF_1& IRF_2& NF\kappa B$"]
    active_states = [r"$IRF_1& NF\kappa B$", r"$IRF_2& NF\kappa B& p50$", 
                     r"$IRF_2& NF\kappa B$", r"$IRF_1& IRF_2$", r"$IRF_1& IRF_2& NF\kappa B$"]
    inactive_states = list(set(state_names)-set(active_states))
    contrib_data["Other"] = contrib_data[inactive_states].sum(axis=1)
    contrib_data.drop(columns = inactive_states,inplace=True)
    state_categories = active_states + ["Other"]
    id_cols = list(set(contrib_data.columns) - set(state_categories))

    # Make long
    contrib_data = contrib_data.melt(id_cols, var_name="State", value_name="contribution")
    contrib_data["State"] = pd.Categorical(contrib_data["State"], categories = state_names + ["Other"], ordered=True).remove_unused_categories()

    make_contrib_figure(contrib_data, figures_dir, more_info="0_to_1")

    ## Making probabilities figure
    state_prob_data = state_prob_data.melt(id_cols, var_name="State", value_name="probability")
    state_prob_data["State"] = pd.Categorical(state_prob_data["State"], categories = state_names, ordered=True)

    make_probs_figure(state_prob_data, figures_dir, more_info="0_to_1")

    if max_p50 > 1:
        # Plot figure only from 0-1 for p50
        testing_data_to1 = testing_data.loc[testing_data["p50"]<=1]
        make_f_figure(testing_data_to1, figures_dir, more_info="0_to_1")

        # Normalize IFNb to value at WT p50
        testing_data_WT = testing_data.loc[testing_data["p50"]==1,["Stimulus","par_number",r"IFN$\beta$"]]
        testing_data_WT.rename(columns={r"IFN$\beta$":"WT_p50_beta"}, inplace=True)
        testing_data_norm = testing_data.merge(testing_data_WT,on=["Stimulus","par_number"])
        testing_data_norm[r"IFN$\beta$"] = testing_data_norm[r"IFN$\beta$"]/testing_data_norm["WT_p50_beta"]
        make_f_figure(testing_data_norm, figures_dir, more_info="norm",ylab="Normalized")

        testing_data_norm_to1 = testing_data_norm.loc[testing_data_norm["p50"]<=1]
        make_f_figure(testing_data_norm_to1, figures_dir, more_info="norm_0_to_1",ylab="Normalized")

    if args.nfkbKO:
        testing_data = calculate_values(num_threads, max_p50,num_p50_values, results_dir, "nfkbKO", "nfkbKO")
        make_f_figure(testing_data, figures_dir, more_info="nfkbKO_0_to_2")

        testing_data_to1 = testing_data.loc[testing_data["p50"]<=1]
        make_f_figure(testing_data_to1, figures_dir, more_info="nfkbKO_0_to_1")


    print("Done.")

if __name__ == "__main__":
    main()