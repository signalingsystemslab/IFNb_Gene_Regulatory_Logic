#  Make nice version of the plots for the two site model
from two_site_model import get_f
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
import argparse
import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import matplotlib as mpl
import numpy as np
from multiprocessing import Pool

mpl.rcParams["figure.dpi"] = 400

data_color = "#FA4B5C"
# states_cmap = sns.cubehelix_palette(start=2.2, rot=.75, dark=0.25, light=0.8, hue=0.6, cmap=True)
states_cmap_pars = "ch:s=2.2,r=0.75,h=0.6,l=0.8,d=0.25"
# models_cmap = sns.cubehelix_palette(start=0.9, rot=-.75, dark=0.3, light=0.8, hue=0.6, cmap=True)
models_cmap_pars = "ch:s=0.9,r=-0.75,h=0.6,l=0.8,d=0.3"
heatmap_cmap = sns.cubehelix_palette(as_cmap=True, light=0.95, dark=0, reverse=True, rot=0.4,start=-.2, hue=0.6)

plot_rc_pars = {"axes.labelsize":7, "font.size":5, "legend.fontsize":7, "xtick.labelsize":6, 
                                          "ytick.labelsize":6, "axes.titlesize":6, "legend.title_fontsize":5,
                                          "lines.markersize": 3, "axes.linewidth": 0.5,
                                            "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.minor.width": 0.5,
                                            "ytick.minor.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2,
                                            "xtick.minor.size": 1, "ytick.minor.size": 1, "legend.labelspacing": 0.2}


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
    df_k_pars["Parameter"] = pd.Categorical(df_k_pars["Parameter"], categories=[r"$k_{I_1}$", r"$k_{I_2}$", r"$k_N$"], ordered=True)

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

def make_predictions_data_frame(ifnb_predicted, beta, conditions):
    df_ifnb_predicted = pd.DataFrame(ifnb_predicted, columns=conditions)
    df_ifnb_predicted["par_set"] = np.arange(len(df_ifnb_predicted))
    df_ifnb_predicted = df_ifnb_predicted.melt(var_name="Data point", value_name=r"IFN$\beta$", id_vars="par_set")

    df_ifnb_predicted_data = pd.DataFrame({"Data point":conditions, r"IFN$\beta$":beta, "par_set":"Data"})
    df_ifnb_predicted = pd.concat([df_ifnb_predicted, df_ifnb_predicted_data], ignore_index=True)
    df_ifnb_predicted["Stimulus"] = df_ifnb_predicted["Data point"].str.split("_", expand=True)[0]
    df_ifnb_predicted["Stimulus"] = df_ifnb_predicted["Stimulus"].replace("polyIC", "PolyIC")
    df_ifnb_predicted["Genotype"] = df_ifnb_predicted["Data point"].str.split("_", expand=True)[1]
    df_ifnb_predicted["Genotype"] = df_ifnb_predicted["Genotype"].replace("relacrelKO", r"NFκBko")
    df_ifnb_predicted["Genotype"] = df_ifnb_predicted["Genotype"].replace("irf3irf7KO", "IRF3/7ko")
    df_ifnb_predicted["Genotype"] = df_ifnb_predicted["Genotype"].replace("irf3irf5irf7KO", "IRF3/5/7ko")
    df_ifnb_predicted["Genotype"] = df_ifnb_predicted["Genotype"].replace("p50KO", "p50ko")
    df_ifnb_predicted["Data point"] = df_ifnb_predicted["Stimulus"] + " " + df_ifnb_predicted["Genotype"]    
    stimuli_levels = ["basal", "CpG", "LPS", "polyIC"]
    # genotypes_levels = ["WT", "irf3irf7KO", "irf3irf5irf7KO", "relacrelKO"]
    genotypes_levels = ["WT", "IRF3/7ko", "IRF3/5/7ko", r"NFκBko","p50ko"]
    df_ifnb_predicted["Stimulus"] = pd.Categorical(df_ifnb_predicted["Stimulus"], categories=stimuli_levels, ordered=True)
    df_ifnb_predicted["Genotype"] = pd.Categorical(df_ifnb_predicted["Genotype"], categories=genotypes_levels, ordered=True)
    df_ifnb_predicted = df_ifnb_predicted.sort_values(["Stimulus", "Genotype"])
    return df_ifnb_predicted


def plot_predictions_one_plot(ifnb_predicted_1, h1, ifnb_predicted_2, h2, ifnb_predicted_3, h3, ifnb_predicted_4, h4, beta, conditions, name, figures_dir):
    # Plot predictions for all conditions in one plot. Average of best 20 models for each hill combination with error bars.
    df_ifnb_predicted_1 = make_predictions_data_frame(ifnb_predicted_1, beta, conditions)
    df_ifnb_predicted_2 = make_predictions_data_frame(ifnb_predicted_2, beta, conditions)
    df_ifnb_predicted_3 = make_predictions_data_frame(ifnb_predicted_3, beta, conditions)
    df_ifnb_predicted_4 = make_predictions_data_frame(ifnb_predicted_4, beta, conditions)

    data_df = df_ifnb_predicted_1.loc[df_ifnb_predicted_1["par_set"] == "Data"].copy()

    df_all = pd.concat([df_ifnb_predicted_1, df_ifnb_predicted_2, df_ifnb_predicted_3, df_ifnb_predicted_4], ignore_index=True)
    # df_all[r"H_{I_2}"] = np.concatenate([np.repeat("1", len(df_ifnb_predicted_1)), np.repeat("1", len(df_ifnb_predicted_2)),
    #                                     np.repeat("3", len(df_ifnb_predicted_3)), np.repeat("3", len(df_ifnb_predicted_4))])
    # df_all[r"H_{I_1}"] = np.concatenate([np.repeat("1", len(df_ifnb_predicted_1)), np.repeat("3", len(df_ifnb_predicted_2)),
    #                                     np.repeat("1", len(df_ifnb_predicted_3)), np.repeat("3", len(df_ifnb_predicted_4))])
    # df_all["Hill"] = "Model Fit\n" + r"($h_{I_1}$=" + df_all[r"H_{I_1}"] + r", $h_{I_2}$=" + df_all[r"H_{I_2}"] + r")
    df_all[r"H_I"] = np.concatenate([np.repeat(h1, len(df_ifnb_predicted_1)), np.repeat(h2, len(df_ifnb_predicted_2)),
                                        np.repeat(h3, len(df_ifnb_predicted_3)), np.repeat(h4, len(df_ifnb_predicted_4))])
    df_all["Hill"] = "Model Fit\n" + r"($h_{I}$=" + df_all[r"H_I"] + r")"
    
    # data_df[r"H_{I_2}"] = np.repeat("Data", len(data_df))
    # data_df[r"H_{I_1}"] = np.repeat("", len(data_df))
    data_df[r"H_I"] = np.repeat("Data", len(data_df))
    data_df["Hill"] = "Experimental"

    df_all = df_all.loc[df_all["par_set"] != "Data"] # contains duplicate data points
    df_all = pd.concat([df_all, data_df], ignore_index=True)
    

    with sns.plotting_context("paper",rc=plot_rc_pars):
        colors = sns.color_palette(models_cmap_pars, n_colors=4)
        col = data_color
        fig, ax = plt.subplots(figsize=(2.2,2))
        # sns.lineplot(data=df_all.loc[df_all["par_set"] != "Data"], x="Data point", y=r"IFN$\beta$", hue="Hill", palette=colors, 
        #              ax=ax, err_style="band", errorbar=("pi",50), zorder = 0)
        # sns.scatterplot(data=df_all.loc[df_all["par_set"] != "Data"], x="Data point", y=r"IFN$\beta$", hue="Hill", palette=colors, marker="o", ax=ax, 
        #                 legend=False, linewidth=0,  zorder = 1)
        
        df_sub = df_all.loc[df_all["par_set"] != "Data"]
        unique_hills = np.unique(df_sub["Hill"])

        for i, hill in enumerate(unique_hills):
            # Filter data for the current Hill
            df_hill = df_all[df_all["Hill"] == hill]

            # Create lineplot and scatterplot for the current Hill
            sns.lineplot(data=df_hill.loc[df_hill["par_set"] != "Data"], x="Data point", y=r"IFN$\beta$", color=colors[i], 
                        ax=ax, err_style="band", errorbar=("pi",50), zorder = i, label=hill)
            sns.scatterplot(data=df_hill.loc[df_hill["par_set"] != "Data"], x="Data point", y=r"IFN$\beta$", color=colors[i],
                        marker="o", ax=ax, linewidth=0,  zorder = i+0.5)


        sns.lineplot(data=df_all.loc[df_all["par_set"] == "Data"], x="Data point", y=r"IFN$\beta$", color=col, ax=ax, label="Experimental", zorder = 10)
        sns.scatterplot(data=df_all.loc[df_all["par_set"] == "Data"], x="Data point", y=r"IFN$\beta$", color=col, marker="o", ax=ax, legend=False, linewidth=0,
                         zorder = 11)
        xticks = ax.get_xticks()
        # for testing
        # labels = [item.get_text().replace(" ", "\n") for item in ax.get_xticklabels()]
        
        # ax.set_xticklabels(labels)

        labels_genotype_only = [item.get_text().split(" ")[1] for item in ax.get_xticklabels()]
        # ax.set_xticklabels(labels_genotype_only) # for testing
        labels_stimulus_only = [item.get_text().split(" ")[0] for item in ax.get_xticklabels()]
        unique_stimuli = np.unique(labels_stimulus_only)
        stimuli_locations = {stimulus: np.where(np.array(labels_stimulus_only) == stimulus)[0] for stimulus in unique_stimuli}
        stimuli_mean_locs = [np.mean(locations) for stimulus, locations in stimuli_locations.items()]
        stimuli_mean_locs = [loc + 10**-5 for loc in stimuli_mean_locs]
        xticks = xticks + stimuli_mean_locs
        unique_stimuli = ["\n\n\n\n\n%s" % stimulus for stimulus in unique_stimuli]
        labels = labels_genotype_only + unique_stimuli
        ax.set_xticks(xticks)
        ax.set_xticklabels(labels)

        for label in ax.get_xticklabels():
            if label.get_text() in labels_genotype_only:
                label.set_rotation(90)

        # Get all xticks
        xticks = ax.xaxis.get_major_ticks()

        # Remove the tick lines for the last three xticks
        for tick in xticks[len(labels_genotype_only):]:
            tick.tick1line.set_visible(False)
            tick.tick2line.set_visible(False)

        sns.despine()
        plt.tight_layout()
        sns.move_legend(ax, bbox_to_anchor=(0.5,1), title=None, frameon=False, loc="lower center", ncol=2)
        plt.savefig("%s/%s.png" % (figures_dir, name), bbox_inches="tight")
        plt.close()

def plot_predictions_one_plot_with_data(ifnb_predicted_1, h1, ifnb_predicted_2, h2, ifnb_predicted_3, h3, ifnb_predicted_4, h4, 
                                        training_data, name, figures_dir):
    # Plot predictions for all conditions in one plot. Average of best 20 models for each hill combination with error bars.
    beta = training_data["IFNb"]
    conditions = training_data["Stimulus"] + "_" + training_data["Genotype"]
    training_data["Condition"] = pd.Categorical(conditions, categories=conditions.unique(), ordered=True)
    df_ifnb_predicted_1 = make_predictions_data_frame(ifnb_predicted_1, beta, conditions)
    df_ifnb_predicted_2 = make_predictions_data_frame(ifnb_predicted_2, beta, conditions)
    df_ifnb_predicted_3 = make_predictions_data_frame(ifnb_predicted_3, beta, conditions)
    df_ifnb_predicted_4 = make_predictions_data_frame(ifnb_predicted_4, beta, conditions)

    data_df = df_ifnb_predicted_1.loc[df_ifnb_predicted_1["par_set"] == "Data"].copy()

    df_all = pd.concat([df_ifnb_predicted_1, df_ifnb_predicted_2, df_ifnb_predicted_3, df_ifnb_predicted_4], ignore_index=True)
    # df_all[r"H_{I_2}"] = np.concatenate([np.repeat("1", len(df_ifnb_predicted_1)), np.repeat("1", len(df_ifnb_predicted_2)),
    #                                     np.repeat("3", len(df_ifnb_predicted_3)), np.repeat("3", len(df_ifnb_predicted_4))])
    # df_all[r"H_{I_1}"] = np.concatenate([np.repeat("1", len(df_ifnb_predicted_1)), np.repeat("3", len(df_ifnb_predicted_2)),
    #                                     np.repeat("1", len(df_ifnb_predicted_3)), np.repeat("3", len(df_ifnb_predicted_4))])
    # df_all["Hill"] = "Model Fit\n" + r"($h_{I_1}$=" + df_all[r"H_{I_1}"] + r", $h_{I_2}$=" + df_all[r"H_{I_2}"] + r")
    df_all[r"H_I"] = np.concatenate([np.repeat(h1, len(df_ifnb_predicted_1)), np.repeat(h2, len(df_ifnb_predicted_2)),
                                        np.repeat(h3, len(df_ifnb_predicted_3)), np.repeat(h4, len(df_ifnb_predicted_4))])
    df_all["Hill"] ="Model Fit\n" r"$h_{I}$=" + df_all[r"H_I"]
    
    # data_df[r"H_{I_2}"] = np.repeat("Data", len(data_df))
    # data_df[r"H_{I_1}"] = np.repeat("", len(data_df))
    data_df[r"H_I"] = np.repeat("Data", len(data_df))
    data_df["Hill"] = "Experimental"

    df_all = df_all.loc[df_all["par_set"] != "Data"] # contains duplicate data points
    df_all = pd.concat([df_all, data_df], ignore_index=True)
    
    new_rc_pars = plot_rc_pars.copy()
    rc_dict = {"legend.fontsize":6.5,"legend.labelspacing":0.1}
    new_rc_pars.update(rc_dict)
    with sns.plotting_context("paper",rc=new_rc_pars):
        colors = sns.color_palette(models_cmap_pars, n_colors=4)
        col = data_color
        fig, ax = plt.subplots(2, 1, figsize=(2.2,2.2), gridspec_kw={"height_ratios": [4, 2]})
        # sns.lineplot(data=df_all.loc[df_all["par_set"] != "Data"], x="Data point", y=r"IFN$\beta$", hue="Hill", palette=colors, 
        #              ax=ax, err_style="band", errorbar=("pi",50), zorder = 0)
        # sns.scatterplot(data=df_all.loc[df_all["par_set"] != "Data"], x="Data point", y=r"IFN$\beta$", hue="Hill", palette=colors, marker="o", ax=ax, 
        #                 legend=False, linewidth=0,  zorder = 1)
        
        df_sub = df_all.loc[df_all["par_set"] != "Data"]
        unique_hills = np.unique(df_sub["Hill"])

        # Plot predictions
        for i, hill in enumerate(unique_hills):
            # Filter data for the current Hill
            df_hill = df_all[df_all["Hill"] == hill]

            # Create lineplot and scatterplot for the current Hill
            sns.lineplot(data=df_hill.loc[df_hill["par_set"] != "Data"], x="Data point", y=r"IFN$\beta$", color=colors[i], 
                        ax=ax[0], err_style="band", errorbar=("pi",50), zorder = i, label=hill)
            sns.scatterplot(data=df_hill.loc[df_hill["par_set"] != "Data"], x="Data point", y=r"IFN$\beta$", color=colors[i],
                        marker="o", ax=ax[0], linewidth=0,  zorder = i+0.5)


        sns.lineplot(data=df_all.loc[df_all["par_set"] == "Data"], x="Data point", y=r"IFN$\beta$", color=col, ax=ax[0], 
                     label="Experimental", zorder = 10)
        sns.scatterplot(data=df_all.loc[df_all["par_set"] == "Data"], x="Data point", y=r"IFN$\beta$", color=col, marker="o", 
                        ax=ax[0], legend=False, linewidth=0, zorder = 11)
        
        # Plot training data
        # Pivot so that IRF and NFKB column names go to "protein" and the values go to "concentration"
        training_data_pivot = training_data.loc[:, ["Condition", "IRF", "NFkB"]]
        training_data_pivot = training_data_pivot.melt(id_vars="Condition", var_name="Protein", value_name="Concentration")
        training_data_pivot["Protein"] = training_data_pivot["Protein"].replace({"IRF": r"$IRF$", "NFkB": r"$NF\kappa B$"})
        # Make wide so that protein is index and condition is columns
        training_data_pivot = training_data_pivot.pivot(index="Protein", columns="Condition", values="Concentration")

        sns.heatmap(training_data_pivot, vmin=0, vmax=1, cmap=heatmap_cmap, cbar_kws={"label": "Concentration"}, annot=True, 
                    fmt=".2f", annot_kws={"size": 5}, ax=ax[1])

        # Labels
        xticks = ax[0].get_xticks()
        labels_genotype_only = [item.get_text().split(" ")[1] for item in ax[0].get_xticklabels()]
        # ax.set_xticklabels(labels_genotype_only) # for testing
        labels_stimulus_only = [item.get_text().split(" ")[0] for item in ax[0].get_xticklabels()]
        unique_stimuli = np.unique(labels_stimulus_only)
        stimuli_locations = {stimulus: np.where(np.array(labels_stimulus_only) == stimulus)[0] for stimulus in unique_stimuli}
        stimuli_mean_locs = [np.mean(locations) for stimulus, locations in stimuli_locations.items()]
        stimuli_mean_locs = [loc + 10**-5 for loc in stimuli_mean_locs]
        xticks = xticks + stimuli_mean_locs
        unique_stimuli = ["\n\n\n\n\n\n%s" % stimulus for stimulus in unique_stimuli]
        labels = labels_genotype_only + unique_stimuli

        ax[0].set_xticklabels("")
        ax[0].set_xticks([])

        xticks = [x + 0.5 for x in xticks]
        ax[1].set_xticks(xticks)
        ax[1].set_xticklabels(labels)
        ax[1].set_ylabel("Input")
        ax[1].set_xlabel("")
        cbar = ax[1].collections[0].colorbar
        cbar.remove()
        ax[0].set_xlabel("")

        for label in ax[1].get_xticklabels():
            if label.get_text() in labels_genotype_only:
                label.set_rotation(90)
            else:
                label.set_rotation(0)

        # Get all xticks
        xticks = ax[1].xaxis.get_major_ticks()

        # Remove the tick lines for the last three xticks
        for tick in xticks[len(labels_genotype_only):]:
            tick.tick1line.set_visible(False)
            tick.tick2line.set_visible(False)

        sns.despine()
        plt.tight_layout()
        sns.move_legend(ax[0], bbox_to_anchor=(0.5,1), title=None, frameon=False, loc="lower center", ncol=3)
        plt.savefig("%s/%s.png" % (figures_dir, name), bbox_inches="tight")
        plt.close()

def make_parameters_data_frame(pars):
    df_pars = pars.drop(columns=["h1", "h2", "h3", "rmsd"], errors="ignore")

    df_pars["par_set"] = np.arange(len(df_pars))
    df_pars = df_pars.melt(var_name="Parameter", value_name="Value", id_vars="par_set")
    # df_t_pars = df_pars[df_pars["Parameter"].str.startswith("t")]
    df_t_pars = df_pars.loc[df_pars["Parameter"].str.startswith("t")].copy()
    num_t_pars = len(df_t_pars["Parameter"].unique())
    new_t_par_names = [r"$t_I$", r"$t_N$", "error","error", "error"]
    # Rename t parameters
    df_t_pars["Parameter"] = df_t_pars["Parameter"].replace(["t1", "t2", "t3", "t4", "t5"], new_t_par_names)
    df_t_pars["Parameter"] = df_t_pars["Parameter"].replace(["t_1", "t_2", "t_3", "t_4", "t_5"], new_t_par_names)
    new_t_par_order = [r"$t_I$", r"$t_N$", r"$t_{I_1I_2}$", r"$t_{I_1N}$"]
    df_t_pars["Parameter"] = pd.Categorical(df_t_pars["Parameter"], categories=new_t_par_order, ordered=True)

    
    # df_k_pars = df_pars[df_pars["Parameter"].str.startswith("k")]
    df_k_pars = df_pars.loc[df_pars["Parameter"].str.startswith("k")].copy()
    num_k_pars = len(df_k_pars["Parameter"].unique())
    # df_k_pars["Parameter"] = df_k_pars["Parameter"].str.replace("k3", r"$k_N$")
    # df_k_pars["Parameter"] = df_k_pars["Parameter"].str.replace("k2", r"$k_2$")
    # df_k_pars["Parameter"] = df_k_pars["Parameter"].str.replace("k1", r"$k_1$")
    # df_k_pars["Parameter"] = df_k_pars["Parameter"].str.replace("kn", r"$k_N$")
    df_k_pars.loc[df_k_pars["Parameter"] == "k1", "Parameter"] = r"$k_I$" # Rename
    df_k_pars.loc[df_k_pars["Parameter"] == "k2", "Parameter"] = r"$k_N$" # Rename
    df_k_pars["Parameter"] = pd.Categorical(df_k_pars["Parameter"], categories=[r"$k_I$", r"$k_N$"], ordered=True)
    return df_t_pars, df_k_pars, num_t_pars, num_k_pars

# Plot parameters one plot
def plot_parameters_one_plot(pars_1, hi_1, pars_2, hi_2, pars_3, hi_3, pars_4, hi_4, name, figures_dir):
    df_t_pars_1, df_k_pars_1, _, _ = make_parameters_data_frame(pars_1)
    df_t_pars_2, df_k_pars_2, _, _ = make_parameters_data_frame(pars_2)
    df_t_pars_3, df_k_pars_3, _, _ = make_parameters_data_frame(pars_3)
    df_t_pars_4, df_k_pars_4, num_t_pars, num_k_pars = make_parameters_data_frame(pars_4)

    df_all_t_pars = pd.concat([df_t_pars_1, df_t_pars_2, df_t_pars_3, df_t_pars_4], ignore_index=True)
    df_all_k_pars = pd.concat([df_k_pars_1, df_k_pars_2, df_k_pars_3, df_k_pars_4], ignore_index=True)

    # df_all_t_pars[r"H_{I_2}"] = np.concatenate([np.repeat("1", len(df_t_pars_1)), np.repeat("1", len(df_t_pars_2)),
    #                                     np.repeat("3", len(df_t_pars_3)), np.repeat("3", len(df_t_pars_4))])
    # df_all_t_pars[r"H_{I_1}"] = np.concatenate([np.repeat("1", len(df_t_pars_1)), np.repeat("3", len(df_t_pars_2)),
    #                                     np.repeat("1", len(df_t_pars_3)), np.repeat("3", len(df_t_pars_4))])
    # df_all_t_pars["Model"] = r"$h_{I_1}$=" + df_all_t_pars[r"H_{I_1}"] + r", $h_{I_2}$=" + df_all_t_pars[r"H_{I_2}"]

    # df_all_k_pars[r"H_{I_2}"] = np.concatenate([np.repeat("1", len(df_k_pars_1)), np.repeat("1", len(df_k_pars_2)),
    #                                     np.repeat("3", len(df_k_pars_3)), np.repeat("3", len(df_k_pars_4))])
    # df_all_k_pars[r"H_{I_1}"] = np.concatenate([np.repeat("1", len(df_k_pars_1)), np.repeat("3", len(df_k_pars_2)),
    #                                     np.repeat("1", len(df_k_pars_3)), np.repeat("3", len(df_k_pars_4))])
    # df_all_k_pars["Model"] = r"$h_{I_1}$=" + df_all_k_pars[r"H_{I_1}"] + r", $h_{I_2}$=" + df_all_k_pars[r"H_{I_2}"]

    df_all_t_pars[r"H_I"] = np.concatenate([np.repeat(hi_1, len(df_t_pars_1)), np.repeat(hi_2, len(df_t_pars_2)),
                                        np.repeat(hi_3, len(df_t_pars_3)), np.repeat(hi_4, len(df_t_pars_4))])
    df_all_k_pars[r"H_I"] = np.concatenate([np.repeat(hi_1, len(df_k_pars_1)), np.repeat(hi_2, len(df_k_pars_2)),
                                        np.repeat(hi_3, len(df_k_pars_3)), np.repeat(hi_4, len(df_k_pars_4))])
    df_all_t_pars["Model"] = r"$h_{I}$=" + df_all_t_pars[r"H_I"]
    df_all_k_pars["Model"] = r"$h_{I}$=" + df_all_k_pars[r"H_I"]

    colors = sns.color_palette(models_cmap_pars, n_colors=4)
    new_rc_pars = plot_rc_pars.copy()
    pars_rc = {"axes.labelsize":7, "font.size":7, "legend.fontsize":7, "xtick.labelsize":7, 
                                          "ytick.labelsize":7, "legend.title_fontsize":5}
    new_rc_pars.update(pars_rc)
    with sns.plotting_context("paper",rc=new_rc_pars):
        fig, ax = plt.subplots(1,2, figsize=(3,1.5), gridspec_kw={"width_ratios":[num_t_pars, num_k_pars]})
        # sns.lineplot(data=df_all_t_pars, x="Parameter", y="Value", hue="Model", ax=ax[0], palette=colors, zorder = 0, errorbar=None)
        # sns.scatterplot(data=df_all_t_pars, x="Parameter", y="Value", hue="Model", ax=ax[0], palette=colors, legend=False, zorder = 1, linewidth=0)
        # sns.lineplot(data=df_all_k_pars, x="Parameter", y="Value", hue="Model", ax=ax[1], palette=colors, legend=False, errorbar=None,
        #                 zorder = 0)
        # sns.scatterplot(data=df_all_k_pars, x="Parameter", y="Value", hue="Model", ax=ax[1], palette=colors, legend=False, zorder = 1, linewidth=0)
        
        unique_models = np.unique(df_all_t_pars["Model"])
        legend_handles = []
        for i, model in enumerate(unique_models):
            # Filter data for the current model
            df_model = df_all_t_pars[df_all_t_pars["Model"] == model]
            l = sns.lineplot(data=df_model, x="Parameter", y="Value", color=colors[i], ax=ax[0], zorder = i, errorbar=None, estimator=None, alpha=0.2, units="par_set")
            sns.scatterplot(data=df_model, x="Parameter", y="Value", color=colors[i], ax=ax[0], legend=False, zorder = i+0.5, linewidth=0)

            legend_handles.append(l.lines[0])

            df_model = df_all_k_pars[df_all_k_pars["Model"] == model]
            sns.lineplot(data=df_model, x="Parameter", y="Value", color=colors[i], ax=ax[1], zorder = i, errorbar=None, estimator=None, alpha=0.2, units="par_set")
            sns.scatterplot(data=df_model, x="Parameter", y="Value", color=colors[i], ax=ax[1], legend=False, zorder = i+0.5, linewidth=0, alpha=0.2)
        
        ax[1].set_yscale("log")
        ax[1].set_ylabel("")
        sns.despine()
        plt.tight_layout()
        # handles, labels = ax[0].get_legend_handles_labels()

        leg = fig.legend(legend_handles, unique_models, loc='lower center', bbox_to_anchor=(0.5, 1), ncol=4, frameon=False)

        for i in range(4):
            leg.legend_handles[i].set_alpha(1)
            leg.legend_handles[i].set_color(colors[i])


        # fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 1), ncol=2, frameon=False)
        # ax[0].get_legend().remove()
        plt.savefig("%s/%s.png" % (figures_dir, name), bbox_inches="tight")
        plt.close()

def calculate_ifnb(pars, data):
    num_t_pars = 2
    num_k_pars = 2
    num_h_pars = 2
    t_pars, k_pars, h_pars = pars[:num_t_pars], pars[num_t_pars:num_t_pars+num_k_pars], pars[num_t_pars+num_k_pars:num_t_pars+num_k_pars+num_h_pars]
    N, I = data["NFkB"], data["IRF"]
    ifnb = [get_f(n,i, model=None, t=t_pars, k=k_pars, h=h_pars, C=1) for n,i in zip(N,I)]
    ifnb = np.array(ifnb)
    return ifnb

def make_param_scan_plots():
    figures_dir = "two_site_final_figures"
    os.makedirs(figures_dir, exist_ok=True)
    model = "two_site"
    training_data = pd.read_csv("../data/training_data.csv")
    beta = training_data["IFNb"]
    conditions = training_data["Stimulus"] + "_" + training_data["Genotype"]
    h_values = np.meshgrid([1,3], [1,3], [1,3])
    h_values = np.array(h_values).T.reshape(-1,3)

    results_dir = "param_scan_2site/results/seed_0/"

    # Load best parameters
    print("Plotting best-fit parameters for all hill combinations on one plot", flush=True)
    best_20_pars_df_1_1 = pd.read_csv("%s/%s_hill_best_20_pars_hi_1_hn_1.csv" % (results_dir, model))
    best_20_pars_df_2_1 = pd.read_csv("%s/%s_hill_best_20_pars_hi_2_hn_1.csv" % (results_dir, model))
    best_20_pars_df_3_1 = pd.read_csv("%s/%s_hill_best_20_pars_hi_3_hn_1.csv" % (results_dir, model))
    best_20_pars_df_4_1 = pd.read_csv("%s/%s_hill_best_20_pars_hi_4_hn_1.csv" % (results_dir, model))
    plot_parameters_one_plot(best_20_pars_df_1_1, "1", best_20_pars_df_2_1, "2", best_20_pars_df_3_1, "3", best_20_pars_df_4_1, "4", "best_20_pars", figures_dir)

    # Calculate ifnb predictions
    print("Plotting best-fit predictions for all hill combinations on one plot", flush=True)
    with Pool(40) as p:
        predictions_1_1 = p.starmap(calculate_ifnb, [(pars, training_data) for pars in best_20_pars_df_1_1.values])
        predictions_2_1 = p.starmap(calculate_ifnb, [(pars, training_data) for pars in best_20_pars_df_2_1.values])
        predictions_3_1 = p.starmap(calculate_ifnb, [(pars, training_data) for pars in best_20_pars_df_3_1.values])
        predictions_4_1 = p.starmap(calculate_ifnb, [(pars, training_data) for pars in best_20_pars_df_4_1.values])

    del best_20_pars_df_1_1, best_20_pars_df_2_1, best_20_pars_df_3_1, best_20_pars_df_4_1

    plot_predictions_one_plot(predictions_1_1, "1", predictions_2_1, "2", predictions_3_1, "3", predictions_4_1, "4", beta, conditions, 
                              "best_20_predictions", figures_dir)
    plot_predictions_one_plot_with_data(predictions_1_1, "1", predictions_2_1, "2", predictions_3_1, "3", predictions_4_1, "4",
                                        training_data, "best_20_predictions_with_data", figures_dir)
    del predictions_1_1, predictions_2_1, predictions_3_1, predictions_4_1   

   
    print("Finished making param scan plots")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c","--contributions", action="store_true")
    parser.add_argument("-p","--param_scan", action="store_true")
    parser.add_argument("-s","--state_probs", action="store_true")
    args = parser.parse_args()

    t = time.time()
    if args.contributions:
        # make_contribution_plots()
        raise NotImplementedError("Contribution plots not implemented yet")

    if args.param_scan:
        make_param_scan_plots()

    if args.state_probs:
        # make_state_probabilities_plots()
        raise NotImplementedError("State probabilities plots not implemented yet")


    print("Finished making all plots, took %.2f seconds" % (time.time() - t))

if __name__ == "__main__":
    main()
