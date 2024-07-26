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

mpl.rcParams["figure.dpi"] = 600
mpl.rcParams["font.sans-serif"] = "Arial"
# mpl.rcParams["font.sans-serif"] = "Liberation Sans"

# print(mpl.get_cachedir())

# data_color = sns.color_palette("ch:s=-1.5,r=-0.8,h=0.7,d=0.25,l=0.9,g=1_r",n_colors=5)[0]
irf_color = "#5D9FB5"
nfkb_color = "#BA4961"
data_color = "#6F5987"
# data_color = "#7C5897"

states_cmap_pars = "ch:s=2.2,r=0.75,h=0.6,l=0.8,d=0.25"
# models_cmap_pars = "ch:s=0.9,r=-0.75,h=0.6,l=0.8,d=0.3"
# sns.cubehelix_palette(n_colors=4, start=0.2,gamma=1,rot=0.4,hue=0.8,dark=0.3,light=0.8,reverse=True)
# models_cmap_pars = "ch:s=-1.5,r=-0.8,h=0.7,d=0.25,l=0.9,g=1_r"
models_cmap_pars = "ch:s=-0.0,r=0.6,h=1,d=0.3,l=0.8,g=1_r"

# data_color = "#FA4B5C"
# data_color = sns.cubehelix_palette(n_colors=5, start=-0.5,rot=0.5,hue=0.7,dark=0.15,light=0.8,reverse=True)[0]
# states_cmap_pars = "ch:s=2.2,r=0.75,h=0.6,l=0.8,d=0.25"
# # models_cmap_pars = "ch:s=0.9,r=-0.75,h=0.6,l=0.8,d=0.3"
# # sns.cubehelix_palette(n_colors=4, start=0.2,gamma=1,rot=0.4,hue=0.8,dark=0.3,light=0.8,reverse=True)
# models_cmap_pars = "ch:s=-0.4,r=0.7,h=0.7,d=0.3,l=0.9,g=1_r"
heatmap_cmap = sns.cubehelix_palette(as_cmap=True, light=0.95, dark=0, reverse=True, rot=0.4,start=-.2, hue=0.6)

plot_rc_pars = {"axes.labelsize":7, "font.size":6, "legend.fontsize":6, "xtick.labelsize":6, 
                                          "ytick.labelsize":6, "axes.titlesize":7, "legend.title_fontsize":7,
                                          "lines.markersize": 3, "axes.linewidth": 0.5,
                                            "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.minor.width": 0.5,
                                            "ytick.minor.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2,
                                            "xtick.minor.size": 1, "ytick.minor.size": 1, "legend.labelspacing": 0.2}
rc_pars={"xtick.major.pad": 1, "ytick.major.pad": 1, "legend.labelspacing": 0.2}
mpl.rcParams.update(rc_pars)

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
    sns.move_legend(ax, bbox_to_anchor=(1.05, 0.5), title=None, frameon=False, loc="center left")
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
    df_ifnb_predicted["Category"] = "Stimulus specific"
    df_ifnb_predicted.loc[df_ifnb_predicted["Genotype"].str.contains("rela"), "Category"] = "NFκB dependence"
    df_ifnb_predicted.loc[df_ifnb_predicted["Genotype"].str.contains("irf"), "Category"] = "IRF dependence"
    df_ifnb_predicted.loc[df_ifnb_predicted["Genotype"].str.contains("p50"), "Category"] = "p50 dependence"

    df_ifnb_predicted["Genotype"] = df_ifnb_predicted["Genotype"].replace("relacrelKO", r"NFκBko")
    df_ifnb_predicted["Genotype"] = df_ifnb_predicted["Genotype"].replace("irf3irf7KO", "IRF3/7ko")
    df_ifnb_predicted["Genotype"] = df_ifnb_predicted["Genotype"].replace("irf3irf5irf7KO", "IRF3/5/7ko")
    df_ifnb_predicted["Genotype"] = df_ifnb_predicted["Genotype"].replace("p50KO", "p50ko")
    df_ifnb_predicted["Data point"] = df_ifnb_predicted["Stimulus"] + " " + df_ifnb_predicted["Genotype"]    
    stimuli_levels = ["basal", "CpG", "LPS", "PolyIC"]
    # genotypes_levels = ["WT", "irf3irf7KO", "irf3irf5irf7KO", "relacrelKO"]
    genotypes_levels = ["WT","p50ko", "IRF3/7ko", "IRF3/5/7ko", r"NFκBko"]
    df_ifnb_predicted["Stimulus"] = pd.Categorical(df_ifnb_predicted["Stimulus"], categories=stimuli_levels, ordered=True)
    df_ifnb_predicted["Genotype"] = pd.Categorical(df_ifnb_predicted["Genotype"], categories=genotypes_levels, ordered=True)
    df_ifnb_predicted = df_ifnb_predicted.sort_values(["Stimulus", "Genotype"])
    # print(df_ifnb_predicted)
    return df_ifnb_predicted

def fix_ax_labels(ax, is_heatmap=False):
    # print([item.get_text().split(" ")[1] for item in ax.get_xticklabels()])
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

def make_predictions_plot(df_all, name, figures_dir):
    # Plot separately
    for category in df_all["Category"].unique():
        # new_rc_pars = plot_rc_pars.copy()
        # new_rc_pars.update({"axes.labelsize":12, "xtick.labelsize":12, "legend.fontsize":12, "legend.title_fontsize":12,
        #                                 "ytick.labelsize":12, "axes.titlesize":12})
        with sns.plotting_context("paper", rc=plot_rc_pars):
            num_bars = len(df_all[df_all["Category"]==category]["Data point"].unique())
            width  = 3.1*num_bars/3/2.1
            height = 1.3/1.7
            fig, ax = plt.subplots(figsize=(width, height))
            cols = [data_color] + sns.color_palette(models_cmap_pars, n_colors=4)
            sns.barplot(data=df_all[df_all["Category"]==category], x="Data point", y=r"IFN$\beta$", hue="Hill", 
                        palette=cols, ax=ax, width=0.8, errorbar="sd", legend=False, saturation=.9, err_kws={'linewidth': 0.75})
            ax.set_xlabel("")
            ax.set_ylabel(r"IFNβ $f$")
            # ax.set_title(category)
            sns.despine()
            ax, _ = fix_ax_labels(ax)
            plt.tight_layout(pad=0)
            plt.ylim(0,1)
            category_nospace = category.replace(" ", "-")
            plt.savefig("%s/%s_%s.png" % (figures_dir, name, category_nospace), bbox_inches="tight")
            plt.close()

    # Make one plot with legend
    with sns.plotting_context("paper", rc=plot_rc_pars):
            category = "NFκB dependence"
            num_bars = len(df_all[df_all["Category"]==category]["Data point"].unique())
            width  = 3.1*num_bars/3/2.1 + 0.5
            height = 1.3/1.7
            fig, ax = plt.subplots(figsize=(width, height))
            cols = [data_color] + sns.color_palette(models_cmap_pars, n_colors=4)
            sns.barplot(data=df_all[df_all["Category"]==category], x="Data point", y=r"IFN$\beta$", hue="Hill", 
                        palette=cols, ax=ax, width=0.8, errorbar="sd", saturation=.9, err_kws={'linewidth': 0.75})
            ax.set_xlabel("")
            ax.set_ylabel(r"IFN$\beta$")
            # ax.set_title(category)
            sns.despine()
            ax, _ = fix_ax_labels(ax)
            plt.tight_layout(pad=0)
            plt.ylim(0,1)
            sns.move_legend(ax, bbox_to_anchor=(1,1), title=None, frameon=False, loc="upper left", ncol=1)
            plt.savefig("%s/%s_legend.png" % (figures_dir, name), bbox_inches="tight")
            plt.close()

def plot_predictions_one_plot(ifnb_predicted_1, h1, ifnb_predicted_2, h2, ifnb_predicted_3, h3, ifnb_predicted_4, h4, beta, conditions, name, figures_dir):
    # Plot predictions for all conditions in one plot. Average of best 20 models for each hill combination with error bars.
    df_ifnb_predicted_1 = make_predictions_data_frame(ifnb_predicted_1, beta, conditions)
    df_ifnb_predicted_2 = make_predictions_data_frame(ifnb_predicted_2, beta, conditions)
    df_ifnb_predicted_3 = make_predictions_data_frame(ifnb_predicted_3, beta, conditions)
    df_ifnb_predicted_4 = make_predictions_data_frame(ifnb_predicted_4, beta, conditions)

    data_df = df_ifnb_predicted_1.loc[df_ifnb_predicted_1["par_set"] == "Data"].copy()

    df_sym = pd.concat([df_ifnb_predicted_1, 
                        df_ifnb_predicted_2,
                        df_ifnb_predicted_3,
                        df_ifnb_predicted_4], ignore_index=True)
    df_sym = pd.concat([df_ifnb_predicted_1, df_ifnb_predicted_2, df_ifnb_predicted_3, df_ifnb_predicted_4], ignore_index=True)
    df_sym[r"H_I"] = np.concatenate([np.repeat(h1, len(df_ifnb_predicted_1)), np.repeat(h2, len(df_ifnb_predicted_2)),
                                        np.repeat(h3, len(df_ifnb_predicted_3)), np.repeat(h4, len(df_ifnb_predicted_4))])
    df_sym["Hill"] = r"$h_{I}$=" + df_sym[r"H_I"]
    
    # data_df[r"H_{I_2}"] = np.repeat("Data", len(data_df))
    # data_df[r"H_{I_1}"] = np.repeat("", len(data_df))
    data_df[r"H_I"] = np.repeat("Data", len(data_df))
    data_df["Hill"] = "Exp."

    df_sym = df_sym.loc[df_sym["par_set"] != "Data"] # contains duplicate data points
    df_all = pd.concat([df_sym, data_df], ignore_index=True)

    hill_categories = np.concatenate([data_df["Hill"].unique(), df_sym["Hill"].unique()])

    df_all["Hill"] = pd.Categorical(df_all["Hill"], categories=hill_categories, ordered=True)
    make_predictions_plot(df_all, name, figures_dir)

    

# def plot_predictions_barplots_onefig(ifnb_predicted_1, h1, ifnb_predicted_2, h2, ifnb_predicted_3, h3, ifnb_predicted_4, h4, beta, conditions, name, figures_dir):
#     df_ifnb_predicted_1 = make_predictions_data_frame(ifnb_predicted_1, beta, conditions)
#     df_ifnb_predicted_2 = make_predictions_data_frame(ifnb_predicted_2, beta, conditions)
#     df_ifnb_predicted_3 = make_predictions_data_frame(ifnb_predicted_3, beta, conditions)
#     df_ifnb_predicted_4 = make_predictions_data_frame(ifnb_predicted_4, beta, conditions)

#     data_df = df_ifnb_predicted_1.loc[df_ifnb_predicted_1["par_set"] == "Data"].copy()

#     df_sym = pd.concat([df_ifnb_predicted_1, 
#                         df_ifnb_predicted_2,
#                         df_ifnb_predicted_3,
#                         df_ifnb_predicted_4], ignore_index=True)
#     df_sym = pd.concat([df_ifnb_predicted_1, df_ifnb_predicted_2, df_ifnb_predicted_3, df_ifnb_predicted_4], ignore_index=True)
#     df_sym[r"H_I"] = np.concatenate([np.repeat(h1, len(df_ifnb_predicted_1)), np.repeat(h2, len(df_ifnb_predicted_2)),
#                                         np.repeat(h3, len(df_ifnb_predicted_3)), np.repeat(h4, len(df_ifnb_predicted_4))])
#     df_sym["Hill"] = r"$h_{I}$=" + df_sym[r"H_I"]
    
#     # data_df[r"H_{I_2}"] = np.repeat("Data", len(data_df))
#     # data_df[r"H_{I_1}"] = np.repeat("", len(data_df))
#     data_df[r"H_I"] = np.repeat("Data", len(data_df))
#     data_df["Hill"] = "Exp."

#     df_sym = df_sym.loc[df_sym["par_set"] != "Data"] # contains duplicate data points
#     df_all = pd.concat([df_sym, data_df], ignore_index=True)

#     hill_categories = np.concatenate([data_df["Hill"].unique(), df_sym["Hill"].unique()])

#     df_all["Hill"] = pd.Categorical(df_all["Hill"], categories=hill_categories, ordered=True)

#     # print(df_all)
#     # return 1

#     data = data_df.copy()
#     barwidth = 0.2 # inch per bar
#     spacing = 3    # spacing between subplots in units of barwidth
#     figx = 5       # figure width in inch
#     left = 4       # left margin in units of bar width
#     right=2      # right margin in units of bar width

#     tc = len(data["Category"].unique())  # total number of categories
#     max_values = []  # holds the maximum number of bars to create
#     for category in data["Category"].unique():
#         num_bars = len(data[data["Category"]==category]["Data point"].unique())
#         max_values.append(num_bars)
#     max_values = np.array(max_values)

#     # print(max_values)
#     # return 1

#     # total figure height:
#     figy = ((np.sum(max_values)+tc) + (tc+1)*spacing)*barwidth #inch

#     fig = plt.figure(figsize=(figx,figy))
#     ax = None
#     for index, category in enumerate(data["Category"].unique()):
#         # print(category)
#         entries = []
#         values = []
#         for entry in data[data["Category"]==category]["Data point"].unique():
#             entries.append(entry)
#             values.append(data[(data["Category"]==category) & (data["Data point"]==entry)][r"IFN$\beta$"].values[0])
#         if not entries:
#             continue  # do not create empty charts

#         # print(entries)
#         # print(values)
#         y_ticks = range(1, len(entries) + 1)
#         # coordinates of new axes [left, bottom, width, height]
#         coord = [left*barwidth/figx, 
#                  1-barwidth*((index+1)*spacing+np.sum(max_values[:index+1])+index+1)/figy,  
#                  1-(left+right)*barwidth/figx,  
#                  (max_values[index]+1)*barwidth/figy ] 

#         ax = fig.add_axes(coord, sharex=ax)
#         ax.barh(y_ticks, values)
#         ax.set_ylim(0, max_values[index] + 1)  # limit the y axis for fixed height
#         ax.set_yticks(y_ticks)
#         ax.set_yticklabels(entries)
#         ax.invert_yaxis()
#         ax.set_title(category)
#         plt.savefig("%s/%s.png" % (figures_dir, name), bbox_inches="tight")
    

# def plot_predictions_one_plot(ifnb_predicted_1, h1, ifnb_predicted_2, h2, ifnb_predicted_3, h3, ifnb_predicted_4, h4, beta, conditions, name, figures_dir):
#     # Plot predictions for all conditions in one plot. Average of best 20 models for each hill combination with error bars.
#     df_ifnb_predicted_1 = make_predictions_data_frame(ifnb_predicted_1, beta, conditions)
#     df_ifnb_predicted_2 = make_predictions_data_frame(ifnb_predicted_2, beta, conditions)
#     df_ifnb_predicted_3 = make_predictions_data_frame(ifnb_predicted_3, beta, conditions)
#     df_ifnb_predicted_4 = make_predictions_data_frame(ifnb_predicted_4, beta, conditions)

#     data_df = df_ifnb_predicted_1.loc[df_ifnb_predicted_1["par_set"] == "Data"].copy()

#     df_all = pd.concat([df_ifnb_predicted_1, df_ifnb_predicted_2, df_ifnb_predicted_3, df_ifnb_predicted_4], ignore_index=True)
#     # df_all[r"H_{I_2}"] = np.concatenate([np.repeat("1", len(df_ifnb_predicted_1)), np.repeat("1", len(df_ifnb_predicted_2)),
#     #                                     np.repeat("3", len(df_ifnb_predicted_3)), np.repeat("3", len(df_ifnb_predicted_4))])
#     # df_all[r"H_{I_1}"] = np.concatenate([np.repeat("1", len(df_ifnb_predicted_1)), np.repeat("3", len(df_ifnb_predicted_2)),
#     #                                     np.repeat("1", len(df_ifnb_predicted_3)), np.repeat("3", len(df_ifnb_predicted_4))])
#     # df_all["Hill"] = "Model Fit\n" + r"($h_{I_1}$=" + df_all[r"H_{I_1}"] + r", $h_{I_2}$=" + df_all[r"H_{I_2}"] + r")
#     df_all[r"H_I"] = np.concatenate([np.repeat(h1, len(df_ifnb_predicted_1)), np.repeat(h2, len(df_ifnb_predicted_2)),
#                                         np.repeat(h3, len(df_ifnb_predicted_3)), np.repeat(h4, len(df_ifnb_predicted_4))])
#     df_all["Hill"] = "Model Fit\n" + r"($h_{I}$=" + df_all[r"H_I"] + r")"
    
#     # data_df[r"H_{I_2}"] = np.repeat("Data", len(data_df))
#     # data_df[r"H_{I_1}"] = np.repeat("", len(data_df))
#     data_df[r"H_I"] = np.repeat("Data", len(data_df))
#     data_df["Hill"] = "Experimental"

#     df_all = df_all.loc[df_all["par_set"] != "Data"] # contains duplicate data points
#     df_all = pd.concat([df_all, data_df], ignore_index=True)
    

#     with sns.plotting_context("paper",rc=plot_rc_pars):
#         colors = sns.color_palette(models_cmap_pars, n_colors=4)
#         col = data_color
#         fig, ax = plt.subplots(figsize=(2.2,2))
#         # sns.lineplot(data=df_all.loc[df_all["par_set"] != "Data"], x="Data point", y=r"IFN$\beta$", hue="Hill", palette=colors, 
#         #              ax=ax, err_style="band", errorbar=("pi",50), zorder = 0)
#         # sns.scatterplot(data=df_all.loc[df_all["par_set"] != "Data"], x="Data point", y=r"IFN$\beta$", hue="Hill", palette=colors, marker="o", ax=ax, 
#         #                 legend=False, linewidth=0,  zorder = 1)
        
#         df_sub = df_all.loc[df_all["par_set"] != "Data"]
#         unique_hills = np.unique(df_sub["Hill"])

#         for i, hill in enumerate(unique_hills):
#             # Filter data for the current Hill
#             df_hill = df_all[df_all["Hill"] == hill]

#             # Create lineplot and scatterplot for the current Hill
#             sns.lineplot(data=df_hill.loc[df_hill["par_set"] != "Data"], x="Data point", y=r"IFN$\beta$", color=colors[i], 
#                         ax=ax, err_style="band", errorbar=("pi",50), zorder = i, label=hill)
#             sns.scatterplot(data=df_hill.loc[df_hill["par_set"] != "Data"], x="Data point", y=r"IFN$\beta$", color=colors[i],
#                         marker="o", ax=ax, linewidth=0,  zorder = i+0.5)


#         sns.lineplot(data=df_all.loc[df_all["par_set"] == "Data"], x="Data point", y=r"IFN$\beta$", color=col, ax=ax, label="Experimental", zorder = 10)
#         sns.scatterplot(data=df_all.loc[df_all["par_set"] == "Data"], x="Data point", y=r"IFN$\beta$", color=col, marker="o", ax=ax, legend=False, linewidth=0,
#                          zorder = 11)
#         xticks = ax.get_xticks()
#         # for testing
#         # labels = [item.get_text().replace(" ", "\n") for item in ax.get_xticklabels()]
        
#         # ax.set_xticklabels(labels)

#         labels_genotype_only = [item.get_text().split(" ")[1] for item in ax.get_xticklabels()]
#         # ax.set_xticklabels(labels_genotype_only) # for testing
#         labels_stimulus_only = [item.get_text().split(" ")[0] for item in ax.get_xticklabels()]
#         unique_stimuli = np.unique(labels_stimulus_only)
#         stimuli_locations = {stimulus: np.where(np.array(labels_stimulus_only) == stimulus)[0] for stimulus in unique_stimuli}
#         stimuli_mean_locs = [np.mean(locations) for stimulus, locations in stimuli_locations.items()]
#         stimuli_mean_locs = [loc + 10**-5 for loc in stimuli_mean_locs]
#         xticks = xticks + stimuli_mean_locs
#         unique_stimuli = ["\n\n\n\n\n\n%s" % stimulus for stimulus in unique_stimuli]
#         labels = labels_genotype_only + unique_stimuli
#         ax.set_xticks(xticks)
#         ax.set_xticklabels(labels)

#         for label in ax.get_xticklabels():
#             if label.get_text() in labels_genotype_only:
#                 label.set_rotation(90)

#         # Get all xticks
#         xticks = ax.xaxis.get_major_ticks()

#         # Remove the tick lines for the last three xticks
#         for tick in xticks[len(labels_genotype_only):]:
#             tick.tick1line.set_visible(False)
#             tick.tick2line.set_visible(False)

#         sns.despine()
#         plt.tight_layout()
#         sns.move_legend(ax, bbox_to_anchor=(0.5,1), title=None, frameon=False, loc="lower center", ncol=3)
#         plt.savefig("%s/%s.png" % (figures_dir, name), bbox_inches="tight")
#         plt.close()

def plot_predictions_one_plot_with_data(ifnb_predicted_1, h1, ifnb_predicted_2, h2, ifnb_predicted_3, h3, ifnb_predicted_4, h4, 
                                        training_data, name, figures_dir, data_only=False):
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
    df_all["Hill"] =r"$h_{I}$=" + df_all[r"H_I"]
    
    # data_df[r"H_{I_2}"] = np.repeat("Data", len(data_df))
    # data_df[r"H_{I_1}"] = np.repeat("", len(data_df))
    data_df[r"H_I"] = np.repeat("Data", len(data_df))
    data_df["Hill"] = "Experimental"

    df_all = df_all.loc[df_all["par_set"] != "Data"] # contains duplicate data points
    df_all = pd.concat([df_all, data_df], ignore_index=True)
    
    new_rc_pars = plot_rc_pars.copy()
    rc_dict = {"legend.fontsize":5,"legend.labelspacing":0.1}
    new_rc_pars.update(rc_dict)
    with sns.plotting_context("paper",rc=new_rc_pars):
        colors = sns.color_palette(models_cmap_pars, n_colors=4)
        col = data_color
        fig, ax = plt.subplots(2, 1, figsize=(2.3,2.7), gridspec_kw={"height_ratios": [4, 2]})
        # sns.lineplot(data=df_all.loc[df_all["par_set"] != "Data"], x="Data point", y=r"IFN$\beta$", hue="Hill", palette=colors, 
        #              ax=ax, err_style="band", errorbar=("pi",50), zorder = 0)
        # sns.scatterplot(data=df_all.loc[df_all["par_set"] != "Data"], x="Data point", y=r"IFN$\beta$", hue="Hill", palette=colors, marker="o", ax=ax, 
        #                 legend=False, linewidth=0,  zorder = 1)
        
        df_sub = df_all.loc[df_all["par_set"] != "Data"]
        unique_hills = np.unique(df_sub["Hill"])

        if not data_only:
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
    # Make columns t_0=0 and t_IN=1
    df_pars["t_0"] = 0
    df_pars["t_IN"] = 1
    df_pars = df_pars.melt(var_name="Parameter", value_name="Value", id_vars="par_set")
    # df_t_pars = df_pars[df_pars["Parameter"].str.startswith("t")]
    df_t_pars = df_pars.loc[df_pars["Parameter"].str.startswith("t")].copy()
    num_t_pars = len(df_t_pars["Parameter"].unique())
    new_t_par_names = [r"$t_I$", r"$t_N$", "error","error", "error"]
    # Rename t parameters
    df_t_pars["Parameter"] = df_t_pars["Parameter"].replace(["t1", "t2", "t3", "t4", "t5"], new_t_par_names)
    df_t_pars["Parameter"] = df_t_pars["Parameter"].replace(["t_1", "t_2", "t_3", "t_4", "t_5"], new_t_par_names)
    df_t_pars["Parameter"] = df_t_pars["Parameter"].replace(["t_0", "t_IN"], [r"$t_0$", r"$t_{IN}$"])
    new_t_par_order = [r"$t_0$",r"$t_I$", r"$t_N$", r"$t_{IN}$"]
    df_t_pars["Parameter"] = pd.Categorical(df_t_pars["Parameter"], categories=new_t_par_order, ordered=True)

    
    # df_k_pars = df_pars[df_pars["Parameter"].str.startswith("k")]
    df_k_pars = df_pars.loc[df_pars["Parameter"].str.startswith("k") | df_pars["Parameter"].str.startswith("c")].copy()
    num_k_pars = len(df_k_pars["Parameter"].unique())
    # df_k_pars["Parameter"] = df_k_pars["Parameter"].str.replace("k3", r"$k_N$")
    # df_k_pars["Parameter"] = df_k_pars["Parameter"].str.replace("k2", r"$k_2$")
    # df_k_pars["Parameter"] = df_k_pars["Parameter"].str.replace("k1", r"$k_1$")
    # df_k_pars["Parameter"] = df_k_pars["Parameter"].str.replace("kn", r"$k_N$")
    df_k_pars.loc[df_k_pars["Parameter"] == "k1", "Parameter"] = r"$k_I$" # Rename
    df_k_pars.loc[df_k_pars["Parameter"] == "k2", "Parameter"] = r"$k_N$" # Rename
    df_k_pars.loc[df_k_pars["Parameter"] == "c", "Parameter"] = r"$C$"
    df_k_pars["Parameter"] = pd.Categorical(df_k_pars["Parameter"], categories=[r"$k_I$", r"$k_N$", r"$C$"], ordered=True)
    return df_t_pars, df_k_pars, num_t_pars, num_k_pars

def make_pars_plots(num_t_pars, num_k_pars, df_all_t_pars, df_all_k_pars, name, figures_dir):
    with sns.plotting_context("paper",rc=plot_rc_pars):
        colors = sns.color_palette(models_cmap_pars, n_colors=4)
        width = 2.1
        height = 1
        fig, ax = plt.subplots(1,2, figsize=(width, height), 
                               gridspec_kw={"width_ratios":[num_t_pars, num_k_pars]})
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

        ax[0].set_ylabel("Parameter Value")
        ax[0].set_xlabel("")
        ax[1].set_xlabel("")

        sns.despine()
        plt.tight_layout()
        # handles, labels = ax[0].get_legend_handles_labels()

        # leg = fig.legend(legend_handles, unique_models, loc='center left', bbox_to_anchor=(1, 0.5), title="Model", frameon=False)
        leg = fig.legend(legend_handles, unique_models, loc='center left', bbox_to_anchor=(1, 0.5), frameon=False,
                        columnspacing=1, handletextpad=0.5, handlelength=1.5)

        for i in range(len(unique_models)):
            leg.legend_handles[i].set_alpha(1)
            leg.legend_handles[i].set_color(colors[i])


        # fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 1), ncol=2, frameon=False)
        # ax[0].get_legend().remove()
        plt.savefig("%s/%s.png" % (figures_dir, name), bbox_inches="tight")
        plt.close()

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

    make_pars_plots(num_t_pars, num_k_pars, df_all_t_pars, df_all_k_pars, name, figures_dir)
    

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

    results_dir = "parameter_scan/"

    # Load best parameters
    print("Plotting best-fit parameters for all hill combinations on one plot", flush=True)
    best_20_pars_df_1_1 = pd.read_csv("%s/results_h_1_1/%s_best_fits_pars.csv" % (results_dir, model))
    best_20_pars_df_2_1 = pd.read_csv("%s/results_h_2_1/%s_best_fits_pars.csv" % (results_dir, model))
    best_20_pars_df_3_1 = pd.read_csv("%s/results_h_3_1/%s_best_fits_pars.csv" % (results_dir, model))
    best_20_pars_df_4_1 = pd.read_csv("%s/results_h_4_1/%s_best_fits_pars.csv" % (results_dir, model))
    plot_parameters_one_plot(best_20_pars_df_1_1, "1", best_20_pars_df_2_1, "2", best_20_pars_df_3_1, "3", best_20_pars_df_4_1, "4", "best_20_pars", figures_dir)
    del best_20_pars_df_1_1, best_20_pars_df_2_1, best_20_pars_df_3_1, best_20_pars_df_4_1
    
    # Calculate ifnb predictions
    print("Plotting best-fit predictions for all hill combinations on one plot", flush=True)
    # with Pool(40) as p:
    #     predictions_1_1 = p.starmap(calculate_ifnb, [(pars, training_data) for pars in best_20_pars_df_1_1.values])
    #     predictions_2_1 = p.starmap(calculate_ifnb, [(pars, training_data) for pars in best_20_pars_df_2_1.values])
    #     predictions_3_1 = p.starmap(calculate_ifnb, [(pars, training_data) for pars in best_20_pars_df_3_1.values])
    #     predictions_4_1 = p.starmap(calculate_ifnb, [(pars, training_data) for pars in best_20_pars_df_4_1.values])

    predictions_1_1 = np.loadtxt("%s/results_h_1_1/%s_best_fits_ifnb_predicted.csv" % (results_dir, model), delimiter=",")
    predictions_2_1 = np.loadtxt("%s/results_h_2_1/%s_best_fits_ifnb_predicted.csv" % (results_dir, model), delimiter=",")
    predictions_3_1 = np.loadtxt("%s/results_h_3_1/%s_best_fits_ifnb_predicted.csv" % (results_dir, model), delimiter=",")
    predictions_4_1 = np.loadtxt("%s/results_h_4_1/%s_best_fits_ifnb_predicted.csv" % (results_dir, model), delimiter=",")

    plot_predictions_one_plot(predictions_1_1, "1", predictions_2_1, "2", predictions_3_1, "3", predictions_4_1, "4", beta, conditions, 
                              "best_20_predictions", figures_dir)
    # plot_predictions_barplots_onefig(predictions_1_1, "1", predictions_2_1, "2", predictions_3_1, "3", predictions_4_1, "4", beta, conditions, 
    #                           "best_predictions_barplot", figures_dir)
    # plot_predictions_one_plot_with_data(predictions_1_1, "1", predictions_2_1, "2", predictions_3_1, "3", predictions_4_1, "4",
    #                                     training_data, "best_20_predictions_with_data", figures_dir)
    # # Plot data only
    # plot_predictions_one_plot_with_data(predictions_1_1, "1", predictions_2_1, "2", predictions_3_1, "3", predictions_4_1, "4",
    #                                     training_data, "scatter_heatmap_data_only", figures_dir, data_only=True)
    del predictions_1_1, predictions_2_1, predictions_3_1, predictions_4_1   

   
    print("Finished making param scan plots")

def make_supplemental_plots():
    figures_dir = "two_site_final_figures"
    os.makedirs(figures_dir, exist_ok=True)
    model = "two_site"
    training_data = pd.read_csv("../data/training_data.csv")
    beta = training_data["IFNb"]
    conditions = training_data["Stimulus"] + "_" + training_data["Genotype"]
    h_values = np.meshgrid([1,3], [1,3], [1,3])
    h_values = np.array(h_values).T.reshape(-1,3)

    results_dir = "parameter_scan/"

    # Load best parameters with c-scan
    best_20_pars_df_1_1 = pd.read_csv("%s/results_h_1_1_c_scan/%s_best_fits_pars.csv" % (results_dir, model))
    best_20_pars_df_2_1 = pd.read_csv("%s/results_h_2_1_c_scan/%s_best_fits_pars.csv" % (results_dir, model))
    best_20_pars_df_3_1 = pd.read_csv("%s/results_h_3_1_c_scan/%s_best_fits_pars.csv" % (results_dir, model))
    best_20_pars_df_4_1 = pd.read_csv("%s/results_h_4_1_c_scan/%s_best_fits_pars.csv" % (results_dir, model))
    plot_parameters_one_plot(best_20_pars_df_1_1, "1", best_20_pars_df_2_1, "2", best_20_pars_df_3_1, "3", best_20_pars_df_4_1, "4", "best_20_pars_c_scan", figures_dir)

    # Calculate ifnb predictions with c-scan
    predictions_1_1 = np.loadtxt("%s/results_h_1_1_c_scan/%s_best_fits_ifnb_predicted.csv" % (results_dir, model), delimiter=",")
    predictions_2_1 = np.loadtxt("%s/results_h_2_1_c_scan/%s_best_fits_ifnb_predicted.csv" % (results_dir, model), delimiter=",")
    predictions_3_1 = np.loadtxt("%s/results_h_3_1_c_scan/%s_best_fits_ifnb_predicted.csv" % (results_dir, model), delimiter=",")
    predictions_4_1 = np.loadtxt("%s/results_h_4_1_c_scan/%s_best_fits_ifnb_predicted.csv" % (results_dir, model), delimiter=",")
    plot_predictions_one_plot(predictions_1_1, "1", predictions_2_1, "2", predictions_3_1, "3", predictions_4_1, "4", beta, conditions, 
                              "best_20_predictions_c_scan", figures_dir)
    
    # Load best parameters with NFkB scan
    best_20_pars_df_1_1 = pd.read_csv("%s/results_h_1_1/%s_best_fits_pars.csv" % (results_dir, model))
    best_20_pars_df_1_3 = pd.read_csv("%s/results_h_1_3/%s_best_fits_pars.csv" % (results_dir, model))
    best_20_pars_df_3_1 = pd.read_csv("%s/results_h_3_1/%s_best_fits_pars.csv" % (results_dir, model))
    best_20_pars_df_3_3 = pd.read_csv("%s/results_h_3_3/%s_best_fits_pars.csv" % (results_dir, model))
    df_t_pars_1_1, df_k_pars_1_1, _, _ = make_parameters_data_frame(best_20_pars_df_1_1)
    df_t_pars_1_3, df_k_pars_1_3, _, _ = make_parameters_data_frame(best_20_pars_df_1_3)
    df_t_pars_3_1, df_k_pars_3_1, _, _ = make_parameters_data_frame(best_20_pars_df_3_1)
    df_t_pars_3_3, df_k_pars_3_3, num_t_pars, num_k_pars = make_parameters_data_frame(best_20_pars_df_3_3)
    df_all_t_pars = pd.concat([df_t_pars_1_1, df_t_pars_1_3, df_t_pars_3_1, df_t_pars_3_3], ignore_index=True)
    df_all_k_pars = pd.concat([df_k_pars_1_1, df_k_pars_1_3, df_k_pars_3_1, df_k_pars_3_3], ignore_index=True)
    df_all_t_pars[r"H_N"] = np.concatenate([np.repeat("1", len(df_t_pars_1_1)), np.repeat("3", len(df_t_pars_1_3)),
                                        np.repeat("1", len(df_t_pars_3_1)), np.repeat("3", len(df_t_pars_3_3))])
    df_all_t_pars[r"H_I"] = np.concatenate([np.repeat("1", len(df_t_pars_1_1)+len(df_t_pars_1_3)),
                                        np.repeat("3", len(df_t_pars_3_1)+len(df_t_pars_3_3))])
    df_all_k_pars[r"H_N"] = np.concatenate([np.repeat("1", len(df_k_pars_1_1)), np.repeat("3", len(df_k_pars_1_3)),
                                        np.repeat("1", len(df_k_pars_3_1)), np.repeat("3", len(df_k_pars_3_3))])
    df_all_k_pars[r"H_I"] = np.concatenate([np.repeat("1", len(df_k_pars_1_1)+len(df_k_pars_1_3)),
                                        np.repeat("3", len(df_k_pars_3_1)+len(df_k_pars_3_3))])
    df_all_t_pars["Model"] = r"$h_I=$" + df_all_t_pars[r"H_I"] + r", $h_N=$" + df_all_t_pars[r"H_N"]
    df_all_k_pars["Model"] = r"$h_I=$" + df_all_k_pars[r"H_I"] + r", $h_N=$" + df_all_k_pars[r"H_N"]
    make_pars_plots(num_t_pars, num_k_pars, df_all_t_pars, df_all_k_pars, "best_20_pars_NFkB_scan", figures_dir)

    # Calculate ifnb predictions with NFkB scan
    predictions_1_1 = np.loadtxt("%s/results_h_1_1/%s_best_fits_ifnb_predicted.csv" % (results_dir, model), delimiter=",")
    predictions_1_3 = np.loadtxt("%s/results_h_1_3/%s_best_fits_ifnb_predicted.csv" % (results_dir, model), delimiter=",")
    predictions_3_1 = np.loadtxt("%s/results_h_3_1/%s_best_fits_ifnb_predicted.csv" % (results_dir, model), delimiter=",")
    predictions_3_3 = np.loadtxt("%s/results_h_3_3/%s_best_fits_ifnb_predicted.csv" % (results_dir, model), delimiter=",")
    df_ifnb_predicted_1_1 = make_predictions_data_frame(predictions_1_1, beta, conditions)
    df_ifnb_predicted_1_3 = make_predictions_data_frame(predictions_1_3, beta, conditions)
    df_ifnb_predicted_3_1 = make_predictions_data_frame(predictions_3_1, beta, conditions)
    df_ifnb_predicted_3_3 = make_predictions_data_frame(predictions_3_3, beta, conditions)
    
    data_df = df_ifnb_predicted_1_3.loc[df_ifnb_predicted_1_3["par_set"] == "Data"].copy()

    df_sym = pd.concat([df_ifnb_predicted_1_1, df_ifnb_predicted_1_3, df_ifnb_predicted_3_1, df_ifnb_predicted_3_3], ignore_index=True)
    df_sym[r"H_I"] = np.concatenate([np.repeat("1", len(df_ifnb_predicted_1_1) + len(df_ifnb_predicted_1_3)),
                                        np.repeat("3", len(df_ifnb_predicted_3_1) + len(df_ifnb_predicted_3_3))])
    df_sym[r"H_N"] = np.concatenate([np.repeat("1", len(df_ifnb_predicted_1_1)), np.repeat("3", len(df_ifnb_predicted_1_3)),
                                        np.repeat("1", len(df_ifnb_predicted_3_1)), np.repeat("3", len(df_ifnb_predicted_3_3))])
    df_sym["Hill"] = r"$h_I=$" + df_sym[r"H_I"] + r", $h_N=$" + df_sym[r"H_N"]

    data_df[r"H_I"] = np.repeat("Data", len(data_df))
    data_df["Hill"] = "Exp."

    df_sym = df_sym.loc[df_sym["par_set"] != "Data"] # contains duplicate data points
    df_all = pd.concat([df_sym, data_df], ignore_index=True)

    hill_categories = np.concatenate([data_df["Hill"].unique(), df_sym["Hill"].unique()])

    df_all["Hill"] = pd.Categorical(df_all["Hill"], categories=hill_categories, ordered=True)
    make_predictions_plot(df_all, "best_20_predictions_NFkB_scan", figures_dir)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c","--contributions", action="store_true")
    parser.add_argument("-p","--param_scan", action="store_true")
    parser.add_argument("-s","--state_probs", action="store_true")
    parser.add_argument("-x","--supplemental", action="store_true")
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
    
    if args.supplemental:
        make_supplemental_plots()


    print("Finished making all plots, took %.2f seconds" % (time.time() - t))

if __name__ == "__main__":
    main()
