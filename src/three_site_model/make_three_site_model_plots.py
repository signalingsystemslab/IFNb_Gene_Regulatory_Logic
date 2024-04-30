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
import matplotlib.colors as mcolors

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
    # best_20_pars_df = pd.read_csv("%s/%s_all_best_20_pars_h_%s.csv" % (best_fit_dir, model, h))
    state_names = np.loadtxt("%s/%s_state_names.txt" % (results_dir, model), dtype=str, delimiter="\0")
    # Rename IRFs
    state_names = [state.replace(r"$IRF$", r"$IRF_2$") for state in state_names]
    state_names = [state.replace(r"$IRF_G$", r"$IRF_1$") for state in state_names]

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

def make_predictions_data_frame(ifnb_predicted, beta, conditions):
    df_ifnb_predicted = pd.DataFrame(ifnb_predicted, columns=conditions)
    df_ifnb_predicted["par_set"] = np.arange(len(df_ifnb_predicted))
    df_ifnb_predicted = df_ifnb_predicted.melt(var_name="Data point", value_name=r"IFN$\beta$", id_vars="par_set")

    df_ifnb_predicted_data = pd.DataFrame({"Data point":conditions, r"IFN$\beta$":beta, "par_set":"Data"})
    df_ifnb_predicted = pd.concat([df_ifnb_predicted, df_ifnb_predicted_data], ignore_index=True)
    df_ifnb_predicted["Stimulus"] = df_ifnb_predicted["Data point"].str.split("_", expand=True)[0]
    df_ifnb_predicted["Stimulus"] = df_ifnb_predicted["Stimulus"].replace("polyIC", "PolyIC")
    df_ifnb_predicted["Genotype"] = df_ifnb_predicted["Data point"].str.split("_", expand=True)[1]
    df_ifnb_predicted["Genotype"] = df_ifnb_predicted["Genotype"].replace("relacrelKO", r"$rela^{-/-}crel^{-/-}$")
    df_ifnb_predicted["Genotype"] = df_ifnb_predicted["Genotype"].replace("irf3irf7KO", r"$irf3^{-/-}irf7^{-/-}$")
    df_ifnb_predicted["Genotype"] = df_ifnb_predicted["Genotype"].replace("irf3irf5irf7KO", r"$irf3^{-/-}irf5^{-/-}irf7^{-/-}$")
    df_ifnb_predicted["Data point"] = df_ifnb_predicted["Stimulus"] + " " + df_ifnb_predicted["Genotype"]    
    stimuli_levels = ["basal", "CpG", "LPS", "polyIC"]
    genotypes_levels = ["WT", "irf3irf7KO", "irf3irf5irf7KO", "relacrelKO"]
    df_ifnb_predicted["Stimulus"] = pd.Categorical(df_ifnb_predicted["Stimulus"], categories=stimuli_levels, ordered=True)
    df_ifnb_predicted["Genotype"] = pd.Categorical(df_ifnb_predicted["Genotype"], categories=genotypes_levels, ordered=True)
    df_ifnb_predicted = df_ifnb_predicted.sort_values(["Stimulus", "Genotype"])
    return df_ifnb_predicted

def plot_predictions(ifnb_predicted, beta, conditions, name, figures_dir, lines = True):
    df_ifnb_predicted = make_predictions_data_frame(ifnb_predicted, beta, conditions)
    col = sns.color_palette("rocket", n_colors=7)[4]
    col = mcolors.rgb2hex(col) 
    fig, ax = plt.subplots(figsize=(6, 6))
    
    sns.scatterplot(data=df_ifnb_predicted, x="Data point", y=r"IFN$\beta$", 
                    color="black", alpha=0.5, ax=ax, zorder = 1, label="Predicted")
    if lines:
        sns.lineplot(data=df_ifnb_predicted.loc[df_ifnb_predicted["par_set"] != "Data"], x="Data point", y=r"IFN$\beta$", 
                        units="par_set", color="black", estimator=None, ax=ax, legend=False, zorder = 2, alpha=0.2)
        sns.lineplot(data=df_ifnb_predicted.loc[df_ifnb_predicted["par_set"] == "Data"], x="Data point", y=r"IFN$\beta$",
                        color=col, estimator=None, ax=ax, legend=False, zorder = 3)
    sns.scatterplot(data=df_ifnb_predicted.loc[df_ifnb_predicted["par_set"] == "Data"], x="Data point", y=r"IFN$\beta$", 
                    color=col, marker="o", ax=ax, zorder = 4, label="Observed")
    xticks = ax.get_xticks()
    labels = [item.get_text().replace(" ", "\n") for item in ax.get_xticklabels()]
    ax.set_xticks(xticks)
    ax.set_xticklabels(labels)
    sns.despine()
    plt.xticks(rotation=90)
    plt.tight_layout()
    sns.move_legend(ax, bbox_to_anchor=(1, 0.5), title=None, frameon=False, loc="center left")
    plt.savefig("%s/%s.png" % (figures_dir, name), bbox_inches="tight")
    plt.close()

def plot_parameters(pars, name, figures_dir):
    df_pars = pars.drop(columns=["h1", "h2", "h3", "rmsd"], errors="ignore")

    df_pars["par_set"] = np.arange(len(df_pars))
    df_pars = df_pars.melt(var_name="Parameter", value_name="Value", id_vars="par_set")
    df_t_pars = df_pars[df_pars["Parameter"].str.startswith("t")]
    num_t_pars = len(df_t_pars["Parameter"].unique())
    
    df_k_pars = df_pars[df_pars["Parameter"].str.startswith("k")]
    num_k_pars = len(df_k_pars["Parameter"].unique())
    # df_k_pars["Parameter"] = df_k_pars["Parameter"].str.replace("k3", r"$k_N$")
    # df_k_pars["Parameter"] = df_k_pars["Parameter"].str.replace("k2", r"$k_2$")
    # df_k_pars["Parameter"] = df_k_pars["Parameter"].str.replace("k1", r"$k_1$")
    # df_k_pars["Parameter"] = df_k_pars["Parameter"].str.replace("kn", r"$k_N$")
    df_k_pars.loc[df_k_pars["Parameter"] == "k1", "Parameter"] = r"$k_{I_2}$" # Rename
    df_k_pars.loc[df_k_pars["Parameter"] == "k2", "Parameter"] = r"$k_{I_1}$" # Rename
    df_k_pars.loc[df_k_pars["Parameter"] == "kn", "Parameter"] = r"$k_N$"
    df_k_pars.loc[df_k_pars["Parameter"] == "k3", "Parameter"] = r"$k_N$"

    fig, ax = plt.subplots(1,2, figsize=(10,5), gridspec_kw={"width_ratios":[num_t_pars, num_k_pars]})
    sns.lineplot(data=df_t_pars, x="Parameter", y="Value", units="par_set", estimator=None, legend=False, alpha=0.2, ax=ax[0], color="black")
    sns.scatterplot(data=df_t_pars, x="Parameter", y="Value", color="black", ax=ax[0], legend=False, alpha=0.2, zorder = 10)
    sns.lineplot(data=df_k_pars, x="Parameter", y="Value", units="par_set", estimator=None, ax=ax[1], legend=False,alpha=0.2, color="black")
    sns.scatterplot(data=df_k_pars, x="Parameter", y="Value", color="black", ax=ax[1], legend=False, alpha=0.2, zorder = 10)
    ax[1].set_yscale("log")
    sns.despine()
    plt.tight_layout()
    plt.savefig("%s/%s.png" % (figures_dir, name))
    plt.close()

def plot_rmsd_boxplot(all_opt_rmsd, model, figures_dir, only_opt=False):
    all_opt_rmsd["Hill"] = pd.Categorical(all_opt_rmsd["Hill"], ordered=True)
    all_opt_rmsd = all_opt_rmsd.sort_values("Hill")
    if only_opt:
        # remove rows where RMSD == "rmsd_initial"
        rmsd_df = all_opt_rmsd.copy()
        rmsd_df = rmsd_df[rmsd_df["RMSD"] == "rmsd_final"]
        fig, ax = plt.subplots(figsize=(10,6))
        col = sns.color_palette("rocket", n_colors=2)[1]
        sns.boxplot(data=rmsd_df, x="Hill", y="Value", color="white")

        ax.set_ylabel("RMSD")

        # Remove x-axis labels
        ax.set_xticklabels([])
        # Remove x-axis title
        ax.set_xlabel("")
        # # Remove x-axis ticks
        # ax.set_xticks([])

        # Create a table of h values
        table_data = rmsd_df[[r"$h_{I_1}$", r"$h_{I_1}$", r"$h_N$"]].drop_duplicates().values.tolist()
        table_data = np.array(table_data).T
        table = plt.table(cellText=table_data, cellLoc='center', loc='bottom', rowLabels=[r"$h_{I_1}$", r"$h_{I_2}$", r"$h_N$"], bbox=[0, -0.25, 1, 0.2])

        colors = sns.color_palette("rocket", n_colors=4)
        alpha = 0.5
        colors = [(color[0], color[1], color[2], alpha) for color in colors]
        # Loop through the cells and change their color based on their text
        for i in range(len(table_data)):
            for j in range(len(table_data[i])):
                cell = table[i, j] 
                if table_data[i][j] in [5,"5"]:
                    cell.set_facecolor(colors[0])
                elif table_data[i][j] in [3,"3"]:
                    cell.set_facecolor(colors[1])
                elif table_data[i][j] in [1,"1"]:
                    cell.set_facecolor(colors[2])
                else:
                    cell.set_facecolor(colors[3])

        # Adjust layout to make room for the table:
        plt.subplots_adjust(left=0.2, bottom=0.18)
        sns.despine()
        plt.xticks(rotation=90)
        plt.tight_layout()

        plt.savefig("%s/%s_rmsd_boxplot_optimized.png" % (figures_dir, model))

    else:
        rmsd_df = all_opt_rmsd.copy()
        rmsd_df["RMSD"] = rmsd_df["RMSD"].replace("rmsd_initial", "Sampling")
        rmsd_df["RMSD"] = rmsd_df["RMSD"].replace("rmsd_final", "Optimization")
        fig, ax = plt.subplots(figsize=(10,6))
        sns.boxplot(data=rmsd_df, x="Hill", y="Value", hue="RMSD", palette="rocket")

        ax.set_ylabel("RMSD")

        # Remove x-axis labels
        ax.set_xticklabels([])
        # Remove x-axis title
        ax.set_xlabel("")
        # # Remove x-axis ticks
        # ax.set_xticks([])

        # Create a table of h values
        table_data = rmsd_df[[r"$h_{I_1}$", r"$h_{I_2}$", r"$h_N$"]].drop_duplicates().values.tolist()
        table_data = np.array(table_data).T
        table = plt.table(cellText=table_data, cellLoc='center', loc='bottom', rowLabels=[r"$h_{I_1}$", r"$h_{I_2}$", r"$h_N$"], bbox=[0, -0.25, 1, 0.2])

        # Adjust layout to make room for the table:
        plt.subplots_adjust(left=0.2, bottom=0.18)
        sns.despine()
        plt.xticks(rotation=90)
        plt.tight_layout()
        sns.move_legend(ax, bbox_to_anchor=(1, 0.5), title=None, frameon=False, loc="center left")
        plt.savefig("%s/%s_rmsd_boxplot_pre_post.png" % (figures_dir, model), bbox_inches="tight")

def make_param_scan_plots():
    figures_dir = "three_site_final_figures"
    os.makedirs(figures_dir, exist_ok=True)
    num_t_pars = 5
    num_k_pars = 3
    num_h_pars = 2
    results_dir ="three_site_param_scan_hill/results/seed_0/"
    optimization_dir = "optimization/results/seed_0/"
    model = "three_site_only_hill"
    training_data = pd.read_csv("../data/training_data.csv")
    beta = training_data["IFNb"]
    conditions = training_data["Stimulus"] + "_" + training_data["Genotype"]
    num_threads = 4
    h_values = np.loadtxt("%s/%s_h_values.csv" % (results_dir, model), dtype=str, delimiter=",")

    # # change context, but make dot size smaller
    # sns.set_context("talk", rc={"lines.markersize": 7})

    with sns.plotting_context("talk", rc={"lines.markersize": 7}):
        # Best fit model wth h=1_1_1 (best 20, seed 0)
        print("Plotting predictions for best fit model with h=1_1_1")
        predictions_1_1_1 = np.loadtxt("%s/%s_best_20_ifnb_h_1_1_1.csv" % (results_dir, model), delimiter=",")
        # plot_predictions(predictions_1_1_1, beta, conditions, "best_20_ifnb_h_1_1_1", figures_dir, lines=False)
        plot_predictions(predictions_1_1_1, beta, conditions, "best_20_ifnb_h_1_1_1_lines", figures_dir, lines=True)
        del predictions_1_1_1

        predictions_1_1_1_opt = np.loadtxt("%s/%s_ifnb_predicted_optimized_h_1_1_1.csv" % (optimization_dir, model), delimiter=",")
        # plot_predictions(predictions_1_1_1_opt, beta, conditions, "ifnb_1_1_1_optimized", figures_dir, lines=False)
        plot_predictions(predictions_1_1_1_opt, beta, conditions, "ifnb_1_1_1_optimized_lines", figures_dir, lines=True)
        del predictions_1_1_1_opt

        # RMSD distribution for all hill combinations (select top 20 for each)
        print("Plotting RMSD distributions for all hill combinations")
        all_best_rmsd = pd.DataFrame(columns=["rmsd", r"$h_{I_1}$", r"$h_{I_2}$", r"$h_N$"])
        for row in h_values:
            h_vals_str = "_".join([str(int(x)) for x in row])
            rmsd = np.loadtxt("%s/%s_rmsd_h_%s.csv" % (results_dir, model, h_vals_str), delimiter=",")
            rmsd_sorted = np.sort(rmsd)
            rmsd = rmsd_sorted[:1000]
            h1, h2, hN = row
            all_best_rmsd = pd.concat([all_best_rmsd, pd.DataFrame({"RMSD":rmsd, r"$h_{I_1}$":h2, r"$h_{I_1}$":h1, r"$h_N$":hN})], ignore_index=True)
            del rmsd, rmsd_sorted

        sns.displot(data=all_best_rmsd, x="RMSD", row=r"$h_{I_1}$", col=r"$h_{I_2}$", hue=r"$h_N$", kind="kde", fill=True, alpha=0.5, palette="rocket")
        sns.despine()
        plt.tight_layout()
        plt.savefig("%s/%s_rmsd_distributions_top_1000.png" % (figures_dir, model))
        plt.close()
        del all_best_rmsd

        # Best fit model with h=3_3_1, color lines as rainbow (top 20 from seed 0)
        print("Plotting predictions for best fit model with h=3_3_1")
        predictions_3_3_1 = np.loadtxt("%s/%s_best_20_ifnb_h_3_3_1.csv" % (results_dir, model), delimiter=",")
        # plot_predictions(predictions_3_3_1, beta, conditions, "best_20_ifnb_h_3_3_1", figures_dir, lines=False)
        plot_predictions(predictions_3_3_1, beta, conditions, "best_20_ifnb_h_3_3_1_lines", figures_dir, lines=True)
        del predictions_3_3_1

        predictions_3_3_1_opt = np.loadtxt("%s/%s_ifnb_predicted_optimized_h_3_3_1.csv" % (optimization_dir, model), delimiter=",")
        # plot_predictions(predictions_3_3_1_opt, beta, conditions, "ifnb_3_3_1_optimized", figures_dir, lines=False)
        plot_predictions(predictions_3_3_1_opt, beta, conditions, "ifnb_3_3_1_optimized_lines", figures_dir, lines=True)
        del predictions_3_3_1_opt

        # Plot best-fit parameters for h=3_3_1
        print("Plotting best-fit parameters for h=3_3_1")
        best_20_pars_df = pd.read_csv("%s/%s_best_20_pars_h_3_3_1.csv" % (results_dir, model))
        plot_parameters(best_20_pars_df, "best_20_pars_h_3_3_1", figures_dir)
        del best_20_pars_df

        col_names = ["t%d" % i for i in range(1, num_t_pars+1)] + ["k%d" % i for i in range(1, num_k_pars+1)]
        best_20_pars_opt = pd.read_csv("%s/%s_optimized_parameters_h_3_3_1.csv" % (optimization_dir, model), names=col_names)
        plot_parameters(best_20_pars_opt, "optimized_pars_h_3_3_1", figures_dir)
        del best_20_pars_opt

        # Plot box plot of post-optimization rmsd
        all_opt_rmsd = pd.DataFrame()
        for row in h_values:
            h_vals_str = "_".join([str(int(x)) for x in row])
            rmsd = pd.read_csv("%s/%s_rmsd_h_%s.csv" % (optimization_dir, model, h_vals_str))
            rmsd[r"$h_{I_1}$"] = row[1]
            rmsd[r"$h_{I_2}$"] = row[0]
            rmsd[r"$h_N$"] = row[2]
            rmsd["Hill"] = r"$h_{I_1}$ = " + str(row[1]) + r", $h_{I_2}$ = " + str(row[0]) + r", $h_N$ = " + str(row[2])
            all_opt_rmsd = pd.concat([all_opt_rmsd, rmsd], ignore_index=True)
            del rmsd

        all_opt_rmsd = all_opt_rmsd.melt(value_vars=["rmsd_initial", "rmsd_final"], var_name="RMSD", value_name="Value", id_vars=[r"$h_1$", r"$h_2$", r"$h_N$", "Hill"])
        plot_rmsd_boxplot(all_opt_rmsd, model, figures_dir, only_opt=False)
        plot_rmsd_boxplot(all_opt_rmsd, model, figures_dir, only_opt=True)

        print("Finished making param scan plots")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c","--contributions", action="store_true")
    parser.add_argument("-p","--param_scan", action="store_true")
    args = parser.parse_args()

    t = time.time()
    if args.contributions:
        make_contribution_plots()

    if args.param_scan:
        make_param_scan_plots()

    print("Finished making all plots, took %.2f seconds" % (time.time() - t))

if __name__ == "__main__":
    main()
