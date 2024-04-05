# Make nice version of the plots for the three site model
from three_site_model import *
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
from multiprocessing import Pool
import argparse
import seaborn as sns
import scipy.stats.qmc as qmc
import matplotlib.ticker as ticker

def helper_contrib_heatmap(*args, **kwargs):
    data = kwargs.pop("data")
    d = data.pivot(index=args[1], columns=args[0], values=args[2])
    ax = sns.heatmap(d, **kwargs)
    ax.invert_yaxis()

    # Set x and y tick locators
    x_ticks_labels = np.linspace(np.min(data[args[0]]), np.max(data[args[0]]), 3)
    y_ticks_labels = np.linspace(np.min(data[args[1]]), np.max(data[args[1]]), 3)
    x_ticks = np.linspace(0, len(data[args[0]].unique()) - 1, 3)
    y_ticks = np.linspace(0, len(data[args[1]].unique()) - 1, 3)
    ax.xaxis.set_major_locator(ticker.FixedLocator(x_ticks))
    ax.yaxis.set_major_locator(ticker.FixedLocator(y_ticks))

    # Format x and y tick labels
    ax.set_xticklabels(["%.1f" % i for i in x_ticks_labels])
    ax.set_yticklabels(["%.1f" % i for i in y_ticks_labels])

def make_contribution_plots():
    figures_dir = "three_site_final_figures"
    os.makedirs(figures_dir, exist_ok=True)
    results_dir = "three_site_contrib/results"
    num_t_pars = 5
    num_k_pars = 3
    num_h_pars = 2
    best_fit_dir ="three_site_param_scan_hill/results/"
    model = "three_site_only_hill"
    num_threads = 40
    h="3_3_1"
    best_20_pars_df = pd.read_csv("%s/%s_all_best_20_pars_h_%s.csv" % (best_fit_dir, model, h))
    state_names = np.loadtxt("%s/%s_state_names.txt" % (results_dir, model), dtype=str, delimiter="\0")

    t = time.time()
    print("Making contribution plots, starting at %s" % time.ctime())

    ## Make heatmaps for all states ##
    contrib_df = pd.read_csv("%s/%s_best_params_contributions_sweep.csv" % (results_dir, model))
    # N_vals = np.loadtxt("%s/%s_N_vals.txt" % (results_dir, model))
    # I_vals = np.loadtxt("%s/%s_I_vals.txt" % (results_dir, model))

    contrib_df = pd.melt(contrib_df, id_vars=[r"NF$\kappa$B", "IRF", "par_set"], value_vars=state_names, var_name="state", value_name="contribution")
    contrib_df["state"] = pd.Categorical(contrib_df["state"], categories=state_names, ordered=True)
    contrib_df = contrib_df.groupby([r"NF$\kappa$B", "IRF", "state"])["contribution"].mean().reset_index()

    p = sns.FacetGrid(contrib_df, col="state", col_wrap=4, sharex=False, sharey=False)
    cbar_ax = p.figure.add_axes([.92, .3, .02, .4])
    p.map_dataframe(helper_contrib_heatmap, r"NF$\kappa$B", "IRF", "contribution", data=contrib_df, cbar_ax=cbar_ax, vmin=0, vmax=1)
    # p.set_axis_labels(r"$IRF$", r"$NF\kappa B$")
    plt.subplots_adjust(top=0.93, right=0.9)
    plt.savefig("%s/%s_contrib_sweep_heatmap.png" % (figures_dir, model))
    plt.close()

    ## Make stacked bar plots for LPS/pIC states ##
    contrib_df = pd.read_csv("%s/%s_specific_conds_contributions.csv" % (results_dir, model))
    contrib_df = pd.melt(contrib_df, id_vars=["stimulus", "genotype", "par_set"], value_vars=state_names, var_name="state", value_name="contribution")
    contrib_df["state"] = pd.Categorical(contrib_df["state"], categories=state_names, ordered=True)
    contrib_df = contrib_df.groupby(["stimulus", "genotype", "state"])["contribution"].mean().reset_index()
    contrib_df["stimulus"] = contrib_df["stimulus"].replace("polyIC", "PolyIC")
    contrib_df["genotype"] = contrib_df["genotype"].replace("relacrelKO", r"$rela^{-/-}crel^{-/-}$")
    contrib_df["Condition"] = contrib_df["stimulus"] + " " + contrib_df["genotype"]
    ax = sns.histplot(data=contrib_df, x="Condition", hue="state", weights="contribution", multiple="stack", shrink=0.8, palette="rocket")
    ax.set_ylabel("Contribution")
    labels = [item.get_text().replace(" ", "\n") for item in ax.get_xticklabels()]
    ax.set_xticklabels(labels)
    # plt.xticks(rotation=90)
    sns.despine()
    sns.move_legend(ax, bbox_to_anchor=(1, 0.5), title=None, frameon=False, loc="center left")
    plt.tight_layout()
    plt.savefig("%s/%s_specific_conds_contributions.png" % (figures_dir, model))
    plt.close()

    print("Finished making contribution plots, took %s" % (time.time() - t))

def make_param_scan_plots():
    figures_dir = "three_site_final_figures"
    os.makedirs(figures_dir, exist_ok=True)
    num_t_pars = 5
    num_k_pars = 3
    num_h_pars = 2
    results_dir ="three_site_param_scan_hill/results/"
    model = "three_site_only_hill"
    num_threads = 40
    h_values = np.readtxt("%s/%s_h_values.csv" % (results_dir, model), dtype=str)
    h="3_3_1"
    best_20_pars_df = pd.read_csv("%s/%s_all_best_20_pars_h_%s.csv" % (results_dir, model, h))

    #Which plots to make?


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c","--contributions", action="store_true")
    args = parser.parse_args()

    if args.contributions:
        make_contribution_plots()


if __name__ == "__main__":
    main()
