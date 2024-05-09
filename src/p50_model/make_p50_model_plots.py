# Make nice version of the plots for the three site model
from p50_model_force_t import get_f
import numpy as np
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

def make_heatmap(contrib_df, cmap, model, name, figures_dir):
    p = sns.FacetGrid(contrib_df, col="state", col_wrap=4, sharex=False, sharey=False)
    cbar_ax = p.figure.add_axes([.92, .3, .02, .4])
    p.map_dataframe(helper_contrib_heatmap, r"NF$\kappa$B", "IRF", "contribution", data=contrib_df, cbar_ax=cbar_ax, vmin=0, vmax=1, cmap=cmap)
    # p.set_axis_labels(r"$IRF$", r"$NF\kappa B$")
    p.set_titles("{col_name}")
    plt.subplots_adjust(top=0.93, right=0.9)
    plt.savefig("%s/%s_%s_heatmap.png" % (figures_dir, model, name))
    plt.close()

def get_renaming_dict(results_dir):
    state_names_old_names = np.loadtxt("%s/p50_state_names.txt" % (results_dir), dtype=str, delimiter="\0")
    state_names = pd.DataFrame(state_names_old_names, columns=["state_old"])
    state_names["state_only"] = state_names["state_old"].str.split(" s", expand=True)[0]
    state_names["state_new"] = state_names["state_only"].replace("$IRF$", "$IRF_2$")
    state_names["state_new"] = state_names["state_new"].replace("$IRF_G$", "$IRF_1$")
    state_names["state_new"] = state_names["state_new"].replace("$IRF\cdot IRF_G$", "$IRF_1\cdot IRF_2$")
    state_names["state_new"] = state_names["state_new"].replace("$IRF\cdot NF\kappa B$", "$IRF_2\cdot NF\kappa B$")
    state_names["state_new"] = state_names["state_new"].replace("$IRF_G\cdot NF\kappa B$", "$IRF_1\cdot NF\kappa B$")
    state_names["state_new"] = state_names["state_new"].replace("$IRF\cdot NF\kappa B\cdot p50$", "$IRF_2\cdot NF\kappa B\cdot p50$")
    state_names["state_new"] = state_names["state_new"].replace("$IRF\cdot IRF_G\cdot NF\kappa B$", "$IRF_1\cdot IRF_2\cdot NF\kappa B$")
    state_names["state_new"] = state_names["state_new"].replace("$IRF\cdot p50$", "$IRF_2\cdot p50$")
    state_name_order = ["Unbound", r"$IRF_1$", r"$IRF_2$", r"$IRF_2\cdot p50$", r"$NF\kappa B$",
                        r"$NF\kappa B\cdot p50$", r"$p50$", r"$IRF_1\cdot IRF_2$", r"$IRF_1\cdot NF\kappa B$",
                        r"$IRF_2\cdot NF\kappa B$", r"$IRF_2\cdot NF\kappa B\cdot p50$", r"$IRF_1\cdot IRF_2\cdot NF\kappa B$"]
    state_names["state_new"] = pd.Categorical(state_names["state_new"], categories=state_name_order, ordered=True)
    state_names = state_names.sort_values("state_new")
    state_names["state"] = state_names["state_new"].astype(str) + " state"
    # Renaming the states
    state_name_dict = state_names.set_index('state_old')["state"].to_dict()
    # Make all values raw strings
    state_name_dict = {k: r"%s" % v for k, v in state_name_dict.items()}
    return state_name_dict


def make_contribution_plots():
    figures_dir = "p50_final_figures"
    os.makedirs(figures_dir, exist_ok=True)
    results_dir = "p50_contrib/results"
    model = "p50"
    num_threads = 40
    h="3_1_1"
    # best_20_pars_df = pd.read_csv("%s/%s_all_best_20_pars_h_%s.csv" % (best_fit_dir, model, h))
    cmap = sns.cubehelix_palette(as_cmap=True, light=0.95, dark=0, reverse=True, rot=0.5)

    t = time.time()
    print("Making contribution plots, starting at %s" % time.ctime(), flush=True)

    ## Make heatmaps for all states, WT p50 ##
    contrib_df = pd.read_csv("%s/%s_best_params_contributions_sweep_p1.csv" % (results_dir, model))
    
    # Rename the columns in contrib_df
    state_name_dict = get_renaming_dict(results_dir)
    state_names = list(state_name_dict.values())
    contrib_df.rename(columns=state_name_dict, inplace=True)

    contrib_df = pd.melt(contrib_df, id_vars=[r"NF$\kappa$B", "IRF", "par_set"], value_vars=state_names, var_name="state", value_name="contribution")
    contrib_df["state"] = pd.Categorical(contrib_df["state"], categories=state_names, ordered=True)
    contrib_df = contrib_df.groupby([r"NF$\kappa$B", "IRF", "state"])["contribution"].mean().reset_index()

    make_heatmap(contrib_df, cmap, model, "contrib_sweep_WT", figures_dir)

    # Make heatmap for all states, KO p50
    contrib_df_KO = pd.read_csv("%s/%s_best_params_contributions_sweep_p0.csv" % (results_dir, model))

    # Rename the columns in contrib_df
    contrib_df_KO.rename(columns=state_name_dict, inplace=True)

    contrib_df_KO = pd.melt(contrib_df_KO, id_vars=[r"NF$\kappa$B", "IRF", "par_set"], value_vars=state_names, var_name="state", value_name="contribution")
    contrib_df_KO["state"] = pd.Categorical(contrib_df_KO["state"], categories=state_names, ordered=True)
    contrib_df_KO = contrib_df_KO.groupby([r"NF$\kappa$B", "IRF", "state"])["contribution"].mean().reset_index()

    make_heatmap(contrib_df_KO, cmap, model, "contrib_sweep_KO", figures_dir)

    ## Make stacked bar plots for LPS/pIC states ##
    with sns.plotting_context("talk", rc={"lines.markersize": 7}):
        contrib_df = pd.read_csv("%s/%s_specific_conds_contributions.csv" % (results_dir, model))
        contrib_df.rename(columns=state_name_dict, inplace=True)
        contrib_df = pd.melt(contrib_df, id_vars=["stimulus", "genotype", "par_set"], value_vars=state_names, var_name="state", value_name="contribution")
        contrib_df["state"] = pd.Categorical(contrib_df["state"], categories=state_names, ordered=True)
        contrib_df = contrib_df.groupby(["stimulus", "genotype", "state"])["contribution"].mean().reset_index()
        # Remove NaN values
        contrib_df = contrib_df.dropna()
        contrib_df["stimulus"] = contrib_df["stimulus"].replace("polyIC", "PolyIC")
        contrib_df["genotype"] = contrib_df["genotype"].replace("relacrelKO", r"$rela^{-/-}crel^{-/-}$")
        contrib_df["genotype"] = contrib_df["genotype"].replace("p50KO", r"$nfkb1^{-/-}$")
        contrib_df["Condition"] = contrib_df["stimulus"] + " " + contrib_df["genotype"]

        # Contributing states
        contrib_states = [r"$IRF_1\cdot IRF_2$ state", r"$IRF_1\cdot NF\kappa B$ state", r"$IRF_1\cdot IRF_2\cdot NF\kappa B$ state"]
        # Sum all non-contributing states into "Other"
        contrib_df["state"] = contrib_df["state"].apply(lambda x: x if x in contrib_states else "Other")
        contrib_df["state"] = pd.Categorical(contrib_df["state"], categories=contrib_states + ["Other"], ordered=True)
        contrib_df = contrib_df.groupby(["Condition", "state"])["contribution"].sum().reset_index()

        fig, ax = plt.subplots(figsize=(15, 5))
        # ax = sns.histplot(data=contrib_df, x="Condition", hue="state", weights="contribution", multiple="stack", shrink=0.8, palette=sns.cubehelix_palette(n_colors=len(contrib_states) + 1), ax=ax)
        ax = sns.histplot(data=contrib_df, x="Condition", hue="state", weights="contribution", multiple="stack", shrink=0.8, palette=sns.color_palette("ch:s=-.2,r=.6", n_colors=len(contrib_states) + 1), ax=ax)
        ax.set_ylabel("Contribution")
        labels = [item.get_text().replace(" ", "\n") for item in ax.get_xticklabels()]
        ax.set_xticklabels(labels)
        # plt.xticks(rotation=90)
        sns.despine()
        sns.move_legend(ax, bbox_to_anchor=(1, 0.5), title=None, frameon=False, loc="center left")
        plt.tight_layout()
        plt.savefig("%s/%s_specific_conds_contributions.png" % (figures_dir, model))
        plt.close()

    print("Finished making contribution plots, took %.2f seconds" % (time.time() - t), flush=True)

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
    df_ifnb_predicted["Genotype"] = df_ifnb_predicted["Genotype"].replace("p50KO", r"$nfkb1^{-/-}$")
    df_ifnb_predicted["Data point"] = df_ifnb_predicted["Stimulus"] + " " + df_ifnb_predicted["Genotype"]    
    stimuli_levels = ["basal", "CpG", "LPS", "polyIC"]
    # genotypes_levels = ["WT", "irf3irf7KO", "irf3irf5irf7KO", "relacrelKO"]
    genotypes_levels = ["WT", r"$irf3^{-/-}irf7^{-/-}$", r"$irf3^{-/-}irf5^{-/-}irf7^{-/-}$", r"$rela^{-/-}crel^{-/-}$",r"$nfkb1^{-/-}$"]
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
    # df_t_pars = df_pars[df_pars["Parameter"].str.startswith("t")]
    df_t_pars = df_pars.loc[df_pars["Parameter"].str.startswith("t")].copy()
    num_t_pars = len(df_t_pars["Parameter"].unique())
    new_t_par_names = [r"$t_{I_2}$", r"$t_{I_1}$", r"$t_N$", r"$t_{I_1I_2}$", r"$t_{I_1N}$"]
    # Rename t parameters
    df_t_pars["Parameter"] = df_t_pars["Parameter"].replace(["t1", "t2", "t3", "t4", "t5"], new_t_par_names)
    df_t_pars["Parameter"] = df_t_pars["Parameter"].replace(["t_1", "t_2", "t_3", "t_4", "t_5"], new_t_par_names)
    new_t_par_order = [r"$t_{I_1}$", r"$t_{I_2}$", r"$t_N$", r"$t_{I_1I_2}$", r"$t_{I_1N}$"]
    df_t_pars["Parameter"] = pd.Categorical(df_t_pars["Parameter"], categories=new_t_par_order, ordered=True)

    
    # df_k_pars = df_pars[df_pars["Parameter"].str.startswith("k")]
    df_k_pars = df_pars.loc[df_pars["Parameter"].str.startswith("k")].copy()
    num_k_pars = len(df_k_pars["Parameter"].unique())
    # df_k_pars["Parameter"] = df_k_pars["Parameter"].str.replace("k3", r"$k_N$")
    # df_k_pars["Parameter"] = df_k_pars["Parameter"].str.replace("k2", r"$k_2$")
    # df_k_pars["Parameter"] = df_k_pars["Parameter"].str.replace("k1", r"$k_1$")
    # df_k_pars["Parameter"] = df_k_pars["Parameter"].str.replace("kn", r"$k_N$")
    df_k_pars.loc[df_k_pars["Parameter"] == "k1", "Parameter"] = r"$k_{I_2}$" # Rename
    df_k_pars.loc[df_k_pars["Parameter"] == "k2", "Parameter"] = r"$k_{I_1}$" # Rename
    df_k_pars.loc[df_k_pars["Parameter"] == "kn", "Parameter"] = r"$k_N$"
    df_k_pars.loc[df_k_pars["Parameter"] == "k3", "Parameter"] = r"$k_N$"
    df_k_pars.loc[df_k_pars["Parameter"] == "kp", "Parameter"] = r"$k_P$"
    df_k_pars.loc[df_k_pars["Parameter"] == "k4", "Parameter"] = r"$k_P$"
    df_k_pars["Parameter"] = pd.Categorical(df_k_pars["Parameter"], categories=[r"$k_{I_1}$", r"$k_{I_2}$", r"$k_N$", r"$k_P$"], ordered=True)

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

def plot_rmsd_boxplot(all_opt_rmsd, model, figures_dir):
    all_opt_rmsd["Hill"] = pd.Categorical(all_opt_rmsd["Hill"], ordered=True)
    all_opt_rmsd = all_opt_rmsd.sort_values("Hill")

    # remove rows where RMSD == "rmsd_initial"
    rmsd_df = all_opt_rmsd.copy()
    fig, ax = plt.subplots(figsize=(6,6))
    col = sns.color_palette("rocket", n_colors=2)[1]
    sns.boxplot(data=rmsd_df, x="Hill", y="rmsd", color="white")

    ax.set_ylabel("RMSD")

    # Remove x-axis labels
    ax.set_xticklabels([])
    # Remove x-axis title
    ax.set_xlabel("")
    # # Remove x-axis ticks
    # ax.set_xticks([])

    # Create a table of h values
    table_data = rmsd_df[[r"$h_{I_1}$", r"$h_{I_2}$"]].drop_duplicates().values.tolist()
    # print(table_data)
    table_data = np.array(table_data).T
    table = plt.table(cellText=table_data, cellLoc='center', loc='bottom', rowLabels=[r"$h_{I_1}$", r"$h_{I_2}$"], bbox=[0, -0.25, 1, 0.2])

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
    plt.savefig("%s/%s_rmsd_boxplot.png" % (figures_dir, model))
    plt.close()

def make_param_scan_plots():
    figures_dir = "p50_final_figures/"
    os.makedirs(figures_dir, exist_ok=True)
    num_t_pars = 5
    num_k_pars = 3
    num_h_pars = 2
    # results_dir ="p50_param_scan_hill/results/seed_0/"
    # results_dir = "parameter_scan/"

    model = "p50_force_t"
    training_data = pd.read_csv("../data/p50_training_data.csv")
    beta = training_data["IFNb"]
    conditions = training_data["Stimulus"] + "_" + training_data["Genotype"]
    num_threads = 4
    # h_scan_dir = "p50_param_scan_hill/results/seed_0/"
    h_values = ["3_1_1","1_1_1", "3_3_1", "1_3_1"]

    force_t_dir = "parameter_scan_force_t/"

    # # change context, but make dot size smaller
    # sns.set_context("talk", rc={"lines.markersize": 7})

    with sns.plotting_context("talk", rc={"lines.markersize": 7}):
        # RMSD distribution for all hill combinations (select top 20 for each)
        print("Plotting RMSD distributions for all hill combinations", flush=True)
        all_best_rmsd = pd.DataFrame(columns=["rmsd", r"$h_{I_1}$", r"$h_{I_2}$"])
        for row in h_values:
            h_vals_str = row
            if row == "3_1_1":
                dir = "%s/results/" % force_t_dir
            else:
                dir = "%s/results_h_%s/" % (force_t_dir, h_vals_str)
            rmsd_df = pd.read_csv("%s/%s_rmsd.csv" % (dir, model))
            h1, h2, _ = row.split("_")
            rmsd_df[r"$h_{I_1}$"] = h2
            rmsd_df[r"$h_{I_2}$"] = h1
            all_best_rmsd = pd.concat([all_best_rmsd, rmsd_df], ignore_index=True)
            del rmsd_df

        col = sns.color_palette("rocket", n_colors=4)[1]
        sns.displot(data=all_best_rmsd, x="rmsd", row=r"$h_{I_1}$", col=r"$h_{I_2}$", kind="kde", fill=True, alpha=0.5, color=col)
        sns.despine()
        plt.tight_layout()
        plt.xlabel("RMSD")
        plt.savefig("%s/%s_rmsd_distributions_top_par_scan.png" % (figures_dir, model))
        plt.close()
        
        # Plot box plot of rmsd
        all_best_rmsd["Hill"] = all_best_rmsd[r"$h_{I_1}$"] + "_" + all_best_rmsd[r"$h_{I_2}$"] 
        plot_rmsd_boxplot(all_best_rmsd, model, figures_dir)
        del all_best_rmsd

        # Best fit model for h=1_1_1, force t constraint
        model_t = "p50_force_t"
        force_t_dir = "%s/results/" % force_t_dir
        print("Plotting predictions for best fit model with h=1_1_1, force t constraint", flush=True)
        predictions_force_t = np.loadtxt("%s/%s_best_fits_ifnb_predicted.csv" % (force_t_dir, model_t), delimiter=",")
        plot_predictions(predictions_force_t, beta, conditions, "best_20_ifnb_force_t_lines", figures_dir, lines=True)
        del predictions_force_t

        # Plot best-fit model for h=3_3_1, force t constraint
        print("Plotting predictions for best fit model with h=3_3_1, force t constraint", flush=True)
        predictions_force_t = np.loadtxt("%s/%s_best_fits_ifnb_predicted.csv" % (force_t_dir, model_t), delimiter=",")
        plot_predictions(predictions_force_t, beta, conditions, "best_20_ifnb_force_t_lines", figures_dir, lines=True)
        del predictions_force_t

        # Best fit model for h=3_1_1, force t constraint
        print("Plotting predictions for best fit model with h=3_1_1, force t constraint", flush=True)
        predictions_force_t = np.loadtxt("%s/%s_best_fits_ifnb_predicted.csv" % (force_t_dir, model_t), delimiter=",")
        plot_predictions(predictions_force_t, beta, conditions, "best_20_ifnb_force_t_lines", figures_dir, lines=True)
        del predictions_force_t

        # Plot best-fit parameters for h=3_1_1, force t constraint
        print("Plotting best-fit parameters for h=3_1_1, force t constraint", flush=True)
        best_20_pars_df = pd.read_csv("%s/%s_best_fits_pars.csv" % (force_t_dir, model_t))
        plot_parameters(best_20_pars_df, "best_20_pars_force_t", figures_dir)


        print("Finished making param scan plots")

def make_state_probabilities_plots():
    figures_dir = "p50_final_figures/"
    os.makedirs(figures_dir, exist_ok=True)
    model = "p50_force_t"
    training_data = pd.read_csv("../data/p50_training_data.csv")
    conditions = training_data["Stimulus"] + "_" + training_data["Genotype"]
    conditions = pd.concat([conditions, pd.Series("basal_WT")], ignore_index=True)
    num_threads = 4
    results_dir = "parameter_scan_force_t/results/"
    names_dir = "p50_contrib/results/"

    cmap = sns.cubehelix_palette(as_cmap=True, light=0.95, dark=0, reverse=True, rot=0.5)

    state_name_dict = get_renaming_dict(names_dir)

    print("Making state probabilities plots", flush=True)
    # Load all state probabilities for the avg. best 20 force t models
    old_state_names = np.loadtxt("%s/p50_state_names.txt" % (names_dir), dtype=str, delimiter="\0")
    # initialize df with state names as index
    probabilities_df = pd.DataFrame(index=state_name_dict.values())
    for condition in conditions:
        state_probs_df = pd.read_csv("%s/%s_state_probabilities_optimized_%s.csv" % (results_dir, model, condition), names=old_state_names)
        state_probs_df.rename(columns=state_name_dict, inplace=True)
        state_probs_df = state_probs_df.mean()
        probabilities_df = pd.concat([probabilities_df, state_probs_df.rename(condition)], axis=1)

    probabilities_df["state"] = probabilities_df.index
    state_probs_df = probabilities_df.melt(var_name="Condition", value_name="Probability", id_vars="state")
    state_probs_df["Stimulus"] = state_probs_df["Condition"].str.split("_", expand=True)[0]
    state_probs_df["Genotype"] = state_probs_df["Condition"].str.split("_", expand=True)[1]
    state_probs_df["Genotype"] = state_probs_df["Genotype"].replace("relacrelKO", r"$rela^{-/-}crel^{-/-}$")
    state_probs_df["Genotype"] = state_probs_df["Genotype"].replace("p50KO", r"$nfkb1^{-/-}$")
    state_probs_df["Genotype"] = state_probs_df["Genotype"].replace("irf3irf7KO", r"$irf3^{-/-}irf7^{-/-}$")
    state_probs_df["Genotype"] = state_probs_df["Genotype"].replace("irf3irf5irf7KO", r"$irf3^{-/-}irf5^{-/-}irf7^{-/-}$")
    state_probs_df["Stimulus"] = state_probs_df["Stimulus"].replace("polyIC", "PolyIC")
    state_probs_df["Stimulus"] = state_probs_df["Stimulus"].replace("basal", "Basal")
    stimuli_levels = ["Basal", "CpG", "LPS", "PolyIC"]
    # stimuli_levels = ["PolyIC", "LPS", "CpG", "Basal"]
    genotypes_levels = ["WT", r"$irf3^{-/-}irf7^{-/-}$", r"$irf3^{-/-}irf5^{-/-}irf7^{-/-}$", r"$rela^{-/-}crel^{-/-}$", r"$nfkb1^{-/-}$"]
    state_probs_df["Condition_old_name"] = state_probs_df["Condition"]
    state_probs_df["Condition"] = state_probs_df["Stimulus"] + " " + state_probs_df["Genotype"]

    condition_renaming_dict = {old: new for old, new in zip(state_probs_df["Condition_old_name"].unique(), state_probs_df["Condition"].unique())}

    state_probs_df["Stimulus"] = pd.Categorical(state_probs_df["Stimulus"], categories=stimuli_levels, ordered=True)
    state_probs_df["Genotype"] = pd.Categorical(state_probs_df["Genotype"], categories=genotypes_levels, ordered=True)
    state_probs_df = state_probs_df.sort_values(["Stimulus", "Genotype"])
    conditions = state_probs_df["Condition"].unique()
    state_probs_df["Condition"] = pd.Categorical(state_probs_df["Condition"], categories=conditions, ordered=True)
    # replace " state" in all state names
    state_probs_df["state"] = state_probs_df["state"].str.replace(" state", "")
    # print(state_probs_df)

    # Plot state probabilities
    with sns.plotting_context("talk", rc={"lines.markersize": 30}):
        # make separate plots for IRF/NFkB KOs and p50 KOs
        irf_kb_data = state_probs_df.loc[state_probs_df["Genotype"].isin(["WT",r"$irf3^{-/-}irf7^{-/-}$", r"$rela^{-/-}crel^{-/-}$"]) & state_probs_df["Stimulus"].isin(["PolyIC", "LPS"])].copy()
        p50_data = state_probs_df.loc[state_probs_df["Genotype"].isin(["WT", r"$nfkb1^{-/-}$"])].copy()

        # Make dictionary of colors
        # genotype_colors = sns.color_palette("rocket", n_colors=len(genotypes_levels))
        genotype_colors = sns.cubehelix_palette(n_colors=len(genotypes_levels), start=-0.2, rot=0.65, dark=0.2, light=0.8, reverse=True)
        genotype_colors = {genotype: color for genotype, color in zip(genotypes_levels, genotype_colors)}

        # First plot
        irf_kb_data["Genotype"] = pd.Categorical(irf_kb_data["Genotype"], categories=["WT", r"$irf3^{-/-}irf7^{-/-}$", r"$rela^{-/-}crel^{-/-}$"], ordered=True)
        irf_kb_data["Stimulus"] = pd.Categorical(irf_kb_data["Stimulus"], categories=["PolyIC", "LPS"], ordered=True)
        p = sns.catplot(data=irf_kb_data, x="state", y="Probability", row="Stimulus", col="Genotype", hue="Genotype", dodge=False, 
                        kind="bar", alpha=0.8, palette=genotype_colors, height=5, aspect=0.9, legend=False)
        p.set_titles("{row_name} {col_name}")
        labels = state_probs_df["state"].unique()
        for ax in p.axes.flat:
            ax.set_xticklabels(labels, rotation=90)
        sns.despine()
        plt.tight_layout()
        plt.savefig("%s/%s_state_probabilities_barplot_irf_kb.png" % (figures_dir, model), bbox_inches="tight")
        plt.close()

        # Second plot
        p50_data["Genotype"] = pd.Categorical(p50_data["Genotype"], categories=["WT", r"$nfkb1^{-/-}$"], ordered=True)
        p50_data["Stimulus"] = pd.Categorical(p50_data["Stimulus"], categories=["LPS", "CpG"], ordered=True)
        p = sns.catplot(data=p50_data, x="state", y="Probability", row="Stimulus", col="Genotype", hue="Genotype", dodge=False,
                        kind="bar", alpha=0.8, palette=genotype_colors, height=5, aspect=0.9, legend=False)
        p.set_titles("{row_name} {col_name}")
        labels = state_probs_df["state"].unique()
        for ax in p.axes.flat:
            ax.set_xticklabels(labels, rotation=90)
        sns.despine()
        plt.tight_layout()
        plt.savefig("%s/%s_state_probabilities_barplot_p50.png" % (figures_dir, model), bbox_inches="tight")
        plt.close()

        # Make heatmap of state probabilities
        # t pars row: 0       t_1    t_1   t_1     t_3     t_3   0      t_4       t_5      t_1 + t_3    t_1 + t_3  1
        best_fit_parameters = pd.read_csv("%s/%s_best_fits_pars.csv" % (results_dir, model)).mean()
        t_pars_df = pd.DataFrame({"state":state_probs_df["state"].unique(), 
                                    "t_value": [0, 
                                            best_fit_parameters.loc["t_1"],
                                            best_fit_parameters.loc["t_1"],
                                            best_fit_parameters.loc["t_1"],
                                            best_fit_parameters.loc["t_3"],
                                            best_fit_parameters.loc["t_3"],
                                            0,
                                            best_fit_parameters.loc["t_4"],
                                            best_fit_parameters.loc["t_5"],
                                            best_fit_parameters.loc["t_1"] + best_fit_parameters.loc["t_3"],
                                            best_fit_parameters.loc["t_1"] + best_fit_parameters.loc["t_3"],
                                            1]})
        t_pars_df = t_pars_df.set_index("state").T
        t_pars_df["Condition"] = [r"Transcription capability ($t$)"]
        t_pars_df = t_pars_df.set_index("Condition")

        # State probability
        state_probs_df["state"] = pd.Categorical(state_probs_df["state"], categories=state_probs_df["state"].unique(), ordered=True)
        state_probs_df = state_probs_df.pivot(index="Condition", columns="state", values="Probability")
    
        # IFNb column
        ifnb_df = pd.read_csv("%s/%s_best_fits_ifnb_predicted.csv" % (results_dir, model), header=None, names=training_data["Stimulus"] + "_" + training_data["Genotype"]).mean()
        t_pars = best_fit_parameters.iloc[:4].values
        k_pars = best_fit_parameters.iloc[4:8].values
        h_pars = [3,1]
        ifnb_basal_wt = get_f(t_pars, k_pars, 0.01, 0.01, 1, h_pars=h_pars)
        ifnb_df.loc["basal_WT"] = ifnb_basal_wt
        ifnb_df = ifnb_df.rename(condition_renaming_dict)
        ifnb_df = ifnb_df.rename(r"IFN$\beta$ mRNA")

        # Add a blank column
        blank_col = pd.DataFrame(np.nan, index=state_probs_df.index, columns=[" "])
        state_probs_df = pd.concat([state_probs_df, blank_col], axis=1)

        # Add the IFNb column
        state_probs_df = pd.concat([state_probs_df, ifnb_df], axis=1)

        # Add a blank row
        blank_row = pd.DataFrame(np.nan, index=[" "], columns=state_probs_df.columns)
        state_probs_df = pd.concat([state_probs_df, blank_row])

        # Add the t_pars row
        t_pars_df = t_pars_df.reindex(columns=state_probs_df.columns)
        state_probs_df = pd.concat([state_probs_df, t_pars_df])

        # Create the heatmap
        # cmap = sns.color_palette("rocket", as_cmap=True)
        fig, ax = plt.subplots(figsize=(10, 8))
        cbar_ax = fig.add_axes([.92, .3, .02, .4])
        sns.heatmap(data=state_probs_df, cbar_ax=cbar_ax, ax=ax, vmin=0, vmax=1, square=True, cmap=cmap)
        ax.set_ylabel("")
        ax.tick_params(axis="both", which="both", length=0)    
        plt.tight_layout()
        plt.subplots_adjust(top=0.93, right=0.9, hspace=0.1, wspace=0.2)
        plt.savefig("%s/%s_state_probabilities_heatmap.png" % (figures_dir, model), bbox_inches="tight")
        plt.close()

        # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 0.2]}, sharex=True)
        # # fig, ax = plt.subplots(figsize=(9, 7))
        # cbar_ax = fig.add_axes([.92, .3, .02, .4])
        # sns.heatmap(data=state_probs_df, cmap="rocket", cbar_ax=cbar_ax, ax=ax1, vmin=0, vmax=1, square=True)
        # ax1.set_ylabel("")
        # ax1.set_xlabel("")
        # ax1.set_xticklabels([])

        # sns.heatmap(data=t_pars_df, cmap="rocket", cbar=False, ax=ax2, vmin=0, vmax=1, square=True)
        # ax2.set_ylabel("")

        # plt.subplots_adjust(top=0.93, right=0.9, hspace=0.1, wspace=0.2)
        # plt.savefig("%s/%s_state_probabilities_heatmap.png" % (figures_dir, model))
        # plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c","--contributions", action="store_true")
    parser.add_argument("-p","--param_scan", action="store_true")
    parser.add_argument("-s","--state_probs", action="store_true")
    args = parser.parse_args()

    t = time.time()
    if args.contributions:
        make_contribution_plots()

    if args.param_scan:
        make_param_scan_plots()

    if args.state_probs:
        make_state_probabilities_plots()

    print("Finished making all plots, took %.2f seconds" % (time.time() - t))

if __name__ == "__main__":
    main()
