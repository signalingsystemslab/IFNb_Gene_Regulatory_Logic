# Plot data as bar graphs
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import os
import time
from multiprocessing import Pool
import argparse
import seaborn as sns
import scipy.stats.qmc as qmc
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors

# Plot settings
mpl.rcParams["figure.dpi"] = 600
mpl.rcParams["font.sans-serif"] = "Arial"


# data_color = sns.color_palette("ch:s=-1.5,r=-0.8,h=0.7,d=0.25,l=0.9,g=1_r",n_colors=5)[0]
# data_color = "#6B5A82"
# irf_color = sns.desaturate(sns.saturate("#659BAA"),0.8)
# nfkb_color = sns.desaturate(sns.saturate("#BC5869"),0.8)
irf_color = "#5D9FB5"
nfkb_color = "#BA4961"
data_color = "#6F5987"

states_cmap_pars = "ch:s=2.2,r=0.75,h=0.6,l=0.8,d=0.25"
# models_cmap_pars = "ch:s=0.9,r=-0.75,h=0.6,l=0.8,d=0.3"
# sns.cubehelix_palette(n_colors=4, start=0.2,gamma=1,rot=0.4,hue=0.8,dark=0.3,light=0.8,reverse=True)
# models_cmap_pars = "ch:s=-1.5,r=-0.8,h=0.7,d=0.25,l=0.9,g=1_r"
models_cmap_pars = "ch:s=-0.0,r=0.6,h=0.8,d=0.3,l=0.8,g=1_r"
heatmap_cmap = sns.cubehelix_palette(as_cmap=True, start=0.9, rot=-.75, dark=0.2, light=0.95, hue=0.6)

plot_rc_pars = {"axes.labelsize":7, "font.size":6, "legend.fontsize":6, "xtick.labelsize":6, 
                                        "ytick.labelsize":6, "axes.titlesize":7, "legend.title_fontsize":7,
                                        "lines.markersize": 3, "axes.linewidth": 0.5,
                                        "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.minor.width": 0.5,
                                        "ytick.minor.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2,
                                        "xtick.minor.size": 1, "ytick.minor.size": 1, "legend.labelspacing": 0.2}
rc_pars={"xtick.major.pad": 1, "ytick.major.pad": 1, "legend.labelspacing": 0.2}
mpl.rcParams.update(rc_pars)


experimental_data = pd.read_csv("../data/p50_training_data.csv")
stimuli_levels = ["basal", "CpG", "LPS", "PolyIC"]
genotypes_levels = ["WT", "IRF3/7ko", "IRF3/5/7ko", r"NFκBko","p50ko"]

# assign categories to data
experimental_data["Category"] = "Stimulus specific"
experimental_data.loc[experimental_data["Genotype"].str.contains("rela"), "Category"] = "NFκB dependence"
experimental_data.loc[experimental_data["Genotype"].str.contains("irf"), "Category"] = "IRF dependence"
experimental_data.loc[experimental_data["Genotype"].str.contains("p50"), "Category"] = "p50 dependence"


# Rename data
experimental_data["Stimulus"] = experimental_data["Stimulus"].replace("polyIC", "PolyIC")
experimental_data["Genotype"] = experimental_data["Genotype"].replace("relacrelKO", r"NFκBko")
experimental_data["Genotype"] = experimental_data["Genotype"].replace("irf3irf7KO", "IRF3/7ko")
experimental_data["Genotype"] = experimental_data["Genotype"].replace("irf3irf5irf7KO", "IRF3/5/7ko")
experimental_data["Genotype"] = experimental_data["Genotype"].replace("p50KO", "p50ko")
experimental_data["Data point"] = experimental_data["Stimulus"] + " " + experimental_data["Genotype"]  

# Sort data
experimental_data["Stimulus"] = pd.Categorical(experimental_data["Stimulus"], categories=stimuli_levels, ordered=True)
experimental_data["Genotype"] = pd.Categorical(experimental_data["Genotype"], categories=genotypes_levels, ordered=True)
experimental_data["Category"] = pd.Categorical(experimental_data["Category"], categories=["Stimulus specific","NFκB dependence", "IRF dependence","p50 dependence"], ordered=True)
experimental_data.sort_values(["Category","Stimulus", "Genotype"], inplace=True)
print(experimental_data)

def fix_ax_labels(ax, is_heatmap=False):
    labels_genotype_only = [item.get_text().split(" ")[1] for item in ax.get_xticklabels()]
    # ax.set_xticklabels(labels_genotype_only)
    labels_stimulus_only = [item.get_text().split(" ")[0] for item in ax.get_xticklabels()]
    unique_gens= np.unique(labels_genotype_only)

    gens_locations = {gen: np.where(np.array(labels_genotype_only) == gen)[0] for gen in unique_gens}
    gens_mean_locs = [np.mean(locations) for gen, locations in gens_locations.items()]
    gens_mean_locs = [loc + 10**-5 for loc in gens_mean_locs]
    xticks = ax.get_xticks()
    # xticks = xticks + gens_mean_locs
    if is_heatmap:
        gens_mean_locs = [loc + 0.5 for loc in gens_mean_locs]
    xticks = np.concatenate((xticks, gens_mean_locs))
    # print(xticks)
    unique_gens = ["\n\n%s" % gen for gen in unique_gens]
    labels = labels_stimulus_only + unique_gens
    ax.set_xticks(xticks)
    ax.set_xticklabels(labels)
    ax.set_xlabel("")

    for label in ax.get_xticklabels():
        label.set_rotation(0)

    # Get all xticks
    xticks = ax.xaxis.get_major_ticks()

    # Remove the tick lines for the last three xticks
    for tick in xticks[len(labels_genotype_only):]:
        tick.tick1line.set_visible(False)
        tick.tick2line.set_visible(False)

    return ax, labels_genotype_only

def plot_data_bar(df, name="IFNb"):
    with sns.plotting_context("paper", rc=plot_rc_pars):
        num_bars = len(df["Data point"].unique())
        add_vals = {"Stimulus specific": 0.1, "NFκB dependence": 0.026, "IRF dependence": 0.1, "p50 dependence": 0.1}
        add_val = add_vals[name]
        heights = {"Stimulus specific": 0.9, "NFκB dependence": 0.9, "IRF dependence": 0.9, "p50 dependence": 0.85}
        fig, ax = plt.subplots(figsize=(num_bars/3.8+add_val, heights[name]))
        p = sns.barplot(data=df, x="Data point", y="IFNb", color=data_color, ax=ax, width=0.8, saturation=.9)

        ax.set_xlabel("")
        ax.set_ylabel(r"")
        # remove ytick labels
        if name != "Stimulus specific":
            ax.set_yticklabels([])

        # ax.set_title(name)
        sns.despine()
        ax.set_ylim(0,1)

        ax, _ = fix_ax_labels(ax)
        # ax.set_xticklabels([])
        # ax.set_xticks([])

        plt.tight_layout(pad=0.05)
        filename = name.replace("dependence", "dependent")
        plt.savefig("%s_data_barplot.png" % filename)
        plt.close()

# Plot training data
        # # Pivot so that IRF and NFKB column names go to "protein" and the values go to "concentration"
        # training_data_pivot = training_data.loc[:, ["Condition", "IRF", "NFkB"]]
        # training_data_pivot = training_data_pivot.melt(id_vars="Condition", var_name="Protein", value_name="Concentration")
        # training_data_pivot["Protein"] = training_data_pivot["Protein"].replace({"IRF": r"$IRF$", "NFkB": r"$NF\kappa B$"})
        # # Make wide so that protein is index and condition is columns
        # training_data_pivot = training_data_pivot.pivot(index="Protein", columns="Condition", values="Concentration")

        # sns.heatmap(training_data_pivot, vmin=0, vmax=1, cmap=heatmap_cmap, cbar_kws={"label": "Concentration"}, annot=True, 
        #             fmt=".2f", annot_kws={"size": 5}, ax=ax[1])

        # sns.despine()
        # plt.tight_layout()
        # sns.move_legend(ax[0], bbox_to_anchor=(0.5,1), title=None, frameon=False, loc="lower center", ncol=3)
        # plt.savefig("%s/%s.png" % (figures_dir, name), bbox_inches="tight")
        # plt.close()

def plot_data_heatmap(df, name="IFNb"):
    data_point_order = df["Data point"].unique()
    training_data_pivot = df.loc[:, ["Data point", "IRF", "NFkB"]]
    training_data_pivot = training_data_pivot.melt(id_vars="Data point", var_name="Protein", value_name="Concentration")
    training_data_pivot["Protein"] = training_data_pivot["Protein"].replace({"IRF": r"$IRF$", "NFkB": r"$NF\kappa B$"})
    training_data_pivot = training_data_pivot.pivot(index="Protein", columns="Data point", values="Concentration")
    training_data_pivot = training_data_pivot[data_point_order]

    with sns.plotting_context("paper", rc=plot_rc_pars):
        num_pts=len(df["Data point"].unique())
        add_amt = 0.75 if num_pts == 3 else 1.4
        right_adj = 0.68 if num_pts == 3 else 0.64
        right_adj-=0.04
        # fig, ax = plt.subplots(figsize=(num_pts/3 + add_amt/3, 0.7))
        fig, ax = plt.subplots(figsize=(num_pts/3.8+0.05, 0.6))
        # cbar_ax = fig.add_axes([.8, .3, .03, .4])
        sns.heatmap(training_data_pivot, vmin=0, vmax=1, cmap=heatmap_cmap, annot=True, 
                     fmt=".2f", ax=ax, linewidths=0.5, linecolor="black", cbar=False)
        # ax.set_title(name)
        ax.set_xlabel("")
        ax.set_ylabel("")
        if name != "Stimulus specific":
            ax.set_yticklabels([])
        ax.tick_params(axis="both", which="both", length=0)
        # plt.subplots_adjust(top=0.9, right=right_adj, hspace=0.5, wspace = 0.2, bottom=0.25)
        # cbar_ax.set_title("Max-Normalized\n Activity", fontsize=11)
        # ax = fix_ax_labels(ax, is_heatmap=True)
        ax.set_xticklabels([])
        ax.set_xticks([])

        sns.despine()
        plt.tight_layout(pad=0.05)
        # relace "dependence" with "dependent" for filename
        filename = name.replace("dependence", "dependent")
        plt.savefig("%s_data_heatmap.png" % filename)
        plt.close()

def plot_irf_nfkb_bar(df, name="irf_nfkb"):
    # pivot df so that IRF and nfkb values go to "Protein" and the values go to "Activity"
    df = df.loc[:, ["Data point", "IRF", "NFkB"]]
    df = df.melt(id_vars="Data point", var_name="Protein", value_name="Activity")
    df["Protein"] = df["Protein"].replace({"NFkB": r"$NF\kappa B$"})

    # plot bar graph of IRF and NFkB values
    with sns.plotting_context("paper", rc=plot_rc_pars):
        # num_pts=len(df["Data point"].unique())
        # add_amt = 0.75 if num_pts == 3 else 1.4
        # right_adj = 0.68 if num_pts == 3 else 0.64
        # right_adj-=0.04

        num_bars = len(df["Data point"].unique())
        # add_vals = {"Stimulus specific": 0.1, "NFκB dependence": 0.05, "IRF dependence": 0.15, "p50 dependence": 0.1}
        add_vals = {"Stimulus specific": 0.05, "NFκB dependence": 0.0, "IRF dependence": 0.025, "p50 dependence": 0.1}
        widths = {"Stimulus specific": 0.8, "NFκB dependence": 0.75, "IRF dependence": 0.7, "p50 dependence": 0.8}
        add_val = add_vals[name]
        heights = {"Stimulus specific": 0.62, "NFκB dependence": 0.595, "IRF dependence": 0.595, "p50 dependence": 0.45}

        fig, ax = plt.subplots(figsize=(num_bars/3.8+add_val, heights[name]))
        # cbar_ax = fig.add_axes([.8, .3, .03, .4])
        colors = {"IRF": irf_color, r"$NF\kappa B$": nfkb_color}
        sns.barplot(data=df, x="Data point", y="Activity", hue="Protein", palette=colors, ax=ax, width=widths[name], 
                    legend=False, saturation=.9)
        ax.set_xlabel("")
        ax.set_ylabel("")
        if name != "Stimulus specific":
            ax.set_yticklabels([])
        else:
            ax.set_yticks([0,0.5,1.0])
        ax.set_xticklabels([])
        # ax.set_xticks([])
        # ax.tick_params(axis="both", which="both", pad=1)
        # ax.bar_label(ax.containers[0])

        sns.despine()
        plt.tight_layout(pad=0.05)
        # relace "dependence" with "dependent" for filename
        filename = name.replace("dependence", "dependent")
        plt.savefig("%s_irf_nfkb_barplot.png" % filename)
        plt.close()

        # Make another plot including legend
        if "NF" in filename:
            print("Making plot with legend")
            fig, ax = plt.subplots(figsize=(num_bars/3.8+0.1, 0.6))
            # cbar_ax = fig.add_axes([.8, .3, .03, .4])
            colors = {"IRF": irf_color, r"$NF\kappa B$": nfkb_color}
            sns.barplot(data=df, x="Data point", y="Activity", hue="Protein", palette=colors, ax=ax, width=0.8,
                saturation=.9)
            ax.set_xlabel("")
            ax.set_ylabel("")
            if name != "Stimulus specific":
                ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.set_xticks([])

            sns.despine()
            plt.tight_layout()
            sns.move_legend(ax, bbox_to_anchor=(1,1), title=None, frameon=False, loc="lower right", ncol=2)

            filename = name.replace("dependence", "dependent")
            plt.savefig("%s_irf_nfkb_legend.png" % filename, bbox_inches="tight")
            plt.close()

# Plot bar graph of IFNb values vs data points
for category in experimental_data["Category"].unique():
    plot_data_bar(experimental_data[experimental_data["Category"]==category], name=category)

# Plot heatmap of NFkB and IRF values
for category in experimental_data["Category"].unique():
    data_heatmap = experimental_data.copy()
    data_heatmap = data_heatmap[data_heatmap["Category"]==category]
    # data_heatmap["Data point"] = data_heatmap["Data point"].str.replace(" ", "\n")
    plot_data_heatmap(data_heatmap, name=category)
    plot_irf_nfkb_bar(data_heatmap, name=category)


# # Testing plotting barplot for all predictions
# def make_predictions_data_frame(ifnb_predicted, beta, conditions):
#     df_ifnb_predicted = pd.DataFrame(ifnb_predicted, columns=conditions)
#     df_ifnb_predicted["par_set"] = np.arange(len(df_ifnb_predicted))
#     df_ifnb_predicted = df_ifnb_predicted.melt(var_name="Data point", value_name=r"IFN$\beta$", id_vars="par_set")

#     df_ifnb_predicted_data = pd.DataFrame({"Data point":conditions, r"IFN$\beta$":beta, "par_set":"Data"})
#     df_ifnb_predicted = pd.concat([df_ifnb_predicted, df_ifnb_predicted_data], ignore_index=True)

#     df_ifnb_predicted["Stimulus"] = df_ifnb_predicted["Data point"].str.split("_", expand=True)[0]
#     df_ifnb_predicted["Stimulus"] = df_ifnb_predicted["Stimulus"].replace("polyIC", "PolyIC")

#     df_ifnb_predicted["Genotype"] = df_ifnb_predicted["Data point"].str.split("_", expand=True)[1]
#     df_ifnb_predicted["Category"] = "Stimulus-specific"
#     df_ifnb_predicted.loc[df_ifnb_predicted["Genotype"].str.contains("rela"), "Category"] = "NFκB-dependence"
#     df_ifnb_predicted.loc[df_ifnb_predicted["Genotype"].str.contains("irf"), "Category"] = "IRF-dependence"
#     df_ifnb_predicted.loc[df_ifnb_predicted["Genotype"].str.contains("p50"), "Category"] = "p50-dependence"

#     df_ifnb_predicted["Genotype"] = df_ifnb_predicted["Genotype"].replace("relacrelKO", r"NFκBko")
#     df_ifnb_predicted["Genotype"] = df_ifnb_predicted["Genotype"].replace("irf3irf7KO", "IRF3/7ko")
#     df_ifnb_predicted["Genotype"] = df_ifnb_predicted["Genotype"].replace("irf3irf5irf7KO", "IRF3/5/7ko")
#     df_ifnb_predicted["Genotype"] = df_ifnb_predicted["Genotype"].replace("p50KO", "p50ko")
#     df_ifnb_predicted["Data point"] = df_ifnb_predicted["Stimulus"] + " " + df_ifnb_predicted["Genotype"]    
#     stimuli_levels = ["basal", "CpG", "LPS", "PolyIC"]
#     # genotypes_levels = ["WT", "irf3irf7KO", "irf3irf5irf7KO", "relacrelKO"]
#     genotypes_levels = ["WT","p50ko", "IRF3/7ko", "IRF3/5/7ko", r"NFκBko"]
#     df_ifnb_predicted["Stimulus"] = pd.Categorical(df_ifnb_predicted["Stimulus"], categories=stimuli_levels, ordered=True)
#     df_ifnb_predicted["Genotype"] = pd.Categorical(df_ifnb_predicted["Genotype"], categories=genotypes_levels, ordered=True)
#     df_ifnb_predicted = df_ifnb_predicted.sort_values(["Stimulus", "Genotype"])
#     # print(df_ifnb_predicted)
#     return df_ifnb_predicted

# def plot_predictions_one_plot(ifnb_predicted_1_1, ifnb_predicted_1_3, ifnb_predicted_3_1, ifnb_predicted_3_3, beta, conditions, name, figures_dir):
#     # Plot predictions for all conditions in one plot. Average of best 20 models for each hill combination with error bars.
#     df_ifnb_predicted_1_1 = make_predictions_data_frame(ifnb_predicted_1_1, beta, conditions)
#     df_ifnb_predicted_1_3 = make_predictions_data_frame(ifnb_predicted_1_3, beta, conditions)
#     df_ifnb_predicted_3_1 = make_predictions_data_frame(ifnb_predicted_3_1, beta, conditions)
#     df_ifnb_predicted_3_3 = make_predictions_data_frame(ifnb_predicted_3_3, beta, conditions)

#     data_df = df_ifnb_predicted_1_1.loc[df_ifnb_predicted_1_1["par_set"] == "Data"].copy()

#     df_sym = pd.concat([df_ifnb_predicted_1_1, 
#                         df_ifnb_predicted_3_1,
#                         df_ifnb_predicted_1_3,
#                         df_ifnb_predicted_3_3], ignore_index=True)
#     df_sym[r"H_{I_2}"] = np.concatenate([np.repeat("1", len(df_ifnb_predicted_1_1)), np.repeat("3", len(df_ifnb_predicted_1_3)),
#                                         np.repeat("1", len(df_ifnb_predicted_3_1)), np.repeat("3", len(df_ifnb_predicted_3_3))])
#     df_sym[r"H_{I_1}"] = np.concatenate([np.repeat("1", len(df_ifnb_predicted_1_1)), np.repeat("1", len(df_ifnb_predicted_1_3)),
#                                         np.repeat("3", len(df_ifnb_predicted_3_1)), np.repeat("3", len(df_ifnb_predicted_3_3))])
#     df_sym["Hill"] = "Model Fit\n" + r"($h_{I_1}$=" + df_sym[r"H_{I_1}"] + r", $h_{I_2}$=" + df_sym[r"H_{I_2}"] + r")"
    
#     data_df[r"H_{I_2}"] = np.repeat("Data", len(data_df))
#     data_df[r"H_{I_1}"] = np.repeat("", len(data_df))
#     data_df["Hill"] = "Experimental"

#     df_sym = df_sym.loc[df_sym["par_set"] != "Data"] # contains duplicate data points
#     df_all = pd.concat([df_sym, data_df], ignore_index=True)

#     hill_categories = np.concatenate([data_df["Hill"].unique(), df_sym["Hill"].unique()])

#     df_all["Hill"] = pd.Categorical(df_all["Hill"], categories=hill_categories, ordered=True)

#     for category in df_all["Category"].unique():
#         new_rc_pars = plot_rc_pars.copy()
#         new_rc_pars.update({"axes.labelsize":12, "xtick.labelsize":12, "legend.fontsize":12, "legend.title_fontsize":12,
#                                         "ytick.labelsize":12, "axes.titlesize":12})
#         with sns.plotting_context("paper", rc=new_rc_pars):
#             num_bars = len(df_all[df_all["Category"]==category]["Data point"].unique())
#             fig, ax = plt.subplots(figsize=(num_bars*1.5 + 1, 3))
#             cols =[data_color] + sns.color_palette(models_cmap_pars, n_colors=4)
#             sns.barplot(data=df_all[df_all["Category"]==category], x="Data point", y=r"IFN$\beta$", hue="Hill", 
#                         palette=cols, ax=ax, width=0.8, errorbar=None)
#             ax.set_xlabel("")
#             ax.set_ylabel(r"IFN$\beta$")
#             ax.set_title(category)
#             sns.despine()
#             ax, _ = fix_ax_labels(ax)
#             plt.tight_layout()
#             plt.ylim(0,1)
#             sns.move_legend(ax, bbox_to_anchor=(1,1), title=None, frameon=False, loc="upper left", ncol=1)
#             plt.savefig("%s/%s_%s.png" % (figures_dir, name, category), bbox_inches="tight")
#             plt.close()

    

# figures_dir = "p50_final_figures/"
# os.makedirs(figures_dir, exist_ok=True)
# training_data = pd.read_csv("../data/p50_training_data.csv")
# beta = training_data["IFNb"]
# conditions = training_data["Stimulus"] + "_" + training_data["Genotype"]
# force_t_dir = "../p50_model/parameter_scan_force_t/"
# model_t = "p50_force_t"

# # Plot predictions on one plot
# print("Plotting predictions for all hill combinations on one plot", flush=True)
# predictions_force_t_1_1 = np.loadtxt("%s/%s_best_fits_ifnb_predicted.csv" % ("%s/results_h_1_1_1/" % force_t_dir, model_t), delimiter=",")
# predictions_force_t_1_3 = np.loadtxt("%s/%s_best_fits_ifnb_predicted.csv" % ("%s/results_h_1_3_1/" % force_t_dir, model_t), delimiter=",")
# predictions_force_t_3_1 = np.loadtxt("%s/%s_best_fits_ifnb_predicted.csv" % ("%s/results/" % force_t_dir, model_t), delimiter=",")
# predictions_force_t_3_3 = np.loadtxt("%s/%s_best_fits_ifnb_predicted.csv" % ("%s/results_h_3_3_1/" % force_t_dir, model_t), delimiter=",")
# plot_predictions_one_plot(predictions_force_t_1_1, predictions_force_t_1_3, predictions_force_t_3_1, predictions_force_t_3_3, beta, conditions, "best_20_ifnb_force_t_lines_all", figures_dir)
