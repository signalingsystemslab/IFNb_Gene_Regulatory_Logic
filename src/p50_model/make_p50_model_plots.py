# Make nice version of the plots for the three site model
from p50_model_force_t import get_f
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

mpl.rcParams["figure.dpi"] = 600
mpl.rcParams["font.sans-serif"] = "Arial"
# irf_color = "#5D9FB5"
# nfkb_color = "#BA4961"
data_color = "#6F5987"

states_cmap_pars = "ch:s=0.9,r=-0.8,h=0.6,l=0.9,d=0.2"
models_cmap_pars = "ch:s=-0.0,r=0.6,h=1,d=0.3,l=0.8,g=1_r"

# # states_cmap = sns.cubehelix_palette(start=2.2, rot=.75, dark=0.25, light=0.8, hue=0.6, cmap=True)
# states_cmap_pars = "ch:s=2.2,r=0.75,h=0.6,l=0.8,d=0.25"
# # models_cmap = sns.cubehelix_palette(start=0.9, rot=-.75, dark=0.3, light=0.8, hue=0.6, cmap=True)
# models_cmap_pars = "ch:s=0.9,r=-0.75,h=0.6,l=0.8,d=0.3"
heatmap_cmap = sns.cubehelix_palette(as_cmap=True, light=0.95, dark=0, reverse=True, rot=0.4,start=-.2, hue=0.6)

plot_rc_pars = {"axes.labelsize":7, "font.size":6, "legend.fontsize":6, "xtick.labelsize":6, 
                                          "ytick.labelsize":6, "axes.titlesize":7, "legend.title_fontsize":7,
                                          "lines.markersize": 3, "axes.linewidth": 0.5,
                                            "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.minor.width": 0.5,
                                            "ytick.minor.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2,
                                            "xtick.minor.size": 1, "ytick.minor.size": 1, "legend.labelspacing": 0.2,
                                            "legend.columnspacing": 0.5, "legend.handletextpad": 0.5, "legend.handlelength": 1.5}
rc_pars={"xtick.major.pad": 1, "ytick.major.pad": 1, "legend.labelspacing": 0.2}
mpl.rcParams.update(rc_pars)

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
    # heatmap_pars = {"axes.titlesize":7}
    # new_rc_pars = plot_rc_pars.copy()
    # new_rc_pars.update(heatmap_pars)

    with sns.plotting_context("paper", rc=plot_rc_pars):
        num_states = len(contrib_df["state"].unique())
        if num_states > 4:
            ncols=4
            p = sns.FacetGrid(contrib_df, col="state", col_wrap=ncols, sharex=True, sharey=True, height=1.1)
            # cbar_ax = p.figure.add_axes([.90, .2, .03, .6])
            p.map_dataframe(helper_contrib_heatmap, r"NF$\kappa$B", "IRF", "contribution", data=contrib_df, cbar=None, vmin=0, vmax=1, 
                            cmap=cmap, square=True)
            p.set_titles("{col_name}")
            plt.subplots_adjust(top=0.93, hspace=0.5, wspace = 0.2)

            # Remove axes labels on all plots except for the plot on the lower left: last row, first column
            num_axes = len(p.axes)
            first_col = [ind % ncols == 0 for ind in range(num_axes)]
            last_row = np.max(np.arange(num_axes)[first_col])
            for i, ax in enumerate(p.axes):
                if i != last_row:
                    ax.set_xlabel("")
                    ax.set_ylabel("")
        else:
            ncols = 4
            p = sns.FacetGrid(contrib_df, col="state", col_wrap=ncols, sharex=True, sharey=True, height=1.3)
            # cbar_ax = p.figure.add_axes([.90, .2, .03, .6])
            p.map_dataframe(helper_contrib_heatmap, r"NF$\kappa$B", "IRF", "contribution", data=contrib_df, cbar=None, vmin=0, vmax=1, 
                            cmap=cmap, square=True)
            p.set_titles("{col_name}")
            plt.subplots_adjust(top=0.8, hspace=0.5, wspace = 0.05)

        # Label color bar
        # cbar_ax.set_title("Max-Normalized\n Transcription")

        

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
    state_names["state"] = state_names["state_new"].astype(str)
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
    cmap = heatmap_cmap

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

    # Filter for three states: IRF1IRF2, IRF1NFkB, IRF1IRF2NFkB
    contrib_df_three = contrib_df.loc[contrib_df["state"].isin([r"$IRF_1\cdot IRF_2$", r"$IRF_1\cdot NF\kappa B$", r"$IRF_1\cdot IRF_2\cdot NF\kappa B$"])].copy()
    contrib_df_three["state"] = contrib_df_three["state"].cat.remove_unused_categories()

    contrib_df_other = contrib_df.loc[~contrib_df["state"].isin([r"$IRF_1\cdot IRF_2$", r"$IRF_1\cdot NF\kappa B$", r"$IRF_1\cdot IRF_2\cdot NF\kappa B$"])].copy()
    contrib_df_other = contrib_df_other.groupby([r"NF$\kappa$B", "IRF"])["contribution"].sum().reset_index()
    contrib_df_other["state"] = "Other"
    contrib_df_other["state"] = pd.Categorical(contrib_df_other["state"], categories=["Other"], ordered=True)
    contrib_df_three = pd.concat([contrib_df_three, contrib_df_other], ignore_index=True)
    print(contrib_df_three)
    make_heatmap(contrib_df_three, cmap, model, "contrib_sweep_WT_three_states", figures_dir)

    # Make heatmap for all states, KO p50
    contrib_df_KO = pd.read_csv("%s/%s_best_params_contributions_sweep_p0.csv" % (results_dir, model))

    # Rename the columns in contrib_df
    contrib_df_KO.rename(columns=state_name_dict, inplace=True)

    contrib_df_KO = pd.melt(contrib_df_KO, id_vars=[r"NF$\kappa$B", "IRF", "par_set"], value_vars=state_names, var_name="state", value_name="contribution")
    contrib_df_KO["state"] = pd.Categorical(contrib_df_KO["state"], categories=state_names, ordered=True)
    contrib_df_KO = contrib_df_KO.groupby([r"NF$\kappa$B", "IRF", "state"])["contribution"].mean().reset_index()

    make_heatmap(contrib_df_KO, cmap, model, "contrib_sweep_KO", figures_dir)

    # make horizontal colorbar for heatmap cmap
    with sns.plotting_context("paper", rc=plot_rc_pars):
        fig, ax = plt.subplots(figsize=(1.3,0.4))
        norm = mcolors.Normalize(vmin=0, vmax=1)
        cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation="horizontal")
        cb.set_label("Max-Normalized Transcription")
        # move label to top
        ax.xaxis.set_label_position("top")
        plt.subplots_adjust(left=0.1, right=0.9, top=0.65, bottom=0.3)
        plt.savefig("%s/heatmap_colorbar.png" % (figures_dir))
        plt.close()



    ## Make stacked bar plots for LPS/pIC states ##
    contrib_df = pd.read_csv("%s/%s_specific_conds_contributions.csv" % (results_dir, model))
    contrib_df.rename(columns=state_name_dict, inplace=True)
    contrib_df = pd.melt(contrib_df, id_vars=["stimulus", "genotype", "par_set"], value_vars=state_names, var_name="state", value_name="contribution")
    contrib_df["state"] = pd.Categorical(contrib_df["state"], categories=state_names, ordered=True)
    contrib_df = contrib_df.groupby(["stimulus", "genotype", "state"])["contribution"].mean().reset_index()
    # Remove NaN values
    contrib_df = contrib_df.dropna()
    contrib_df["stimulus"] = contrib_df["stimulus"].replace("polyIC", "PolyIC")
    contrib_df["genotype"] = contrib_df["genotype"].replace("relacrelKO", r"NFκBko")
    contrib_df["genotype"] = contrib_df["genotype"].replace("p50KO", "p50ko")
    contrib_df["Condition"] = contrib_df["stimulus"] + " " + contrib_df["genotype"]
    stimulus_levels = ["basal", "CpG", "LPS", "PolyIC"]
    genotype_levels = ["WT", "p50ko", "IRF3/7ko", "IRF3/5/7ko", r"NFκBko"]
    contrib_df["stimulus"] = pd.Categorical(contrib_df["stimulus"], categories=stimulus_levels, ordered=True)
    contrib_df["genotype"] = pd.Categorical(contrib_df["genotype"], categories=genotype_levels, ordered=True)
    contrib_df = contrib_df.sort_values(["stimulus", "genotype"])
    condition_levels = contrib_df["Condition"].unique()
    contrib_df["Condition"] = pd.Categorical(contrib_df["Condition"], categories=condition_levels, ordered=True)

    # Contributing states
    contrib_states = [r"$IRF_1\cdot NF\kappa B$", r"$IRF_1\cdot IRF_2$", r"$IRF_1\cdot IRF_2\cdot NF\kappa B$"]
    # Sum all non-contributing states into "Other"
    contrib_df["state"] = contrib_df["state"].apply(lambda x: x if x in contrib_states else "Other")
    contrib_df["state"] = pd.Categorical(contrib_df["state"], categories=contrib_states + ["Other"], ordered=True)
    contrib_df = contrib_df.groupby(["Condition", "state"])["contribution"].sum().reset_index()

    # new_rc_pars = plot_rc_pars.copy()
    # stack_rc_pars = {"xtick.labelsize":5}
    # new_rc_pars.update(stack_rc_pars)
    with sns.plotting_context("paper", rc=plot_rc_pars):
        states_colors = sns.color_palette(states_cmap_pars, n_colors=len(contrib_states)) + ["#444D59"]
        fig, ax = plt.subplots(figsize=(2.9,1.7))
        ax = sns.histplot(data=contrib_df, x="Condition", hue="state", weights="contribution", multiple="stack", shrink=0.8,
                          palette=states_colors, ax=ax, linewidth=0.5)
        ax.set_ylabel("Transcription")
        # labels = [item.get_text().replace(" ", "\n") for item in ax.get_xticklabels()]
        # ax.set_xticklabels(labels)
        # ax.set_xticks(ax.get_xticks())
        ax.set_ylim(0,1)
        sns.despine()

        labels_genotype_only = [item.get_text().split(" ")[1] for item in ax.get_xticklabels()]
        # ax.set_xticklabels(labels_genotype_only)
        labels_stimulus_only = [item.get_text().split(" ")[0] for item in ax.get_xticklabels()]
        unique_stimuli = np.unique(labels_stimulus_only)
        stimuli_locations = {stimulus: np.where(np.array(labels_stimulus_only) == stimulus)[0] for stimulus in unique_stimuli}
        stimuli_mean_locs = [np.mean(locations) for stimulus, locations in stimuli_locations.items()]
        stimuli_mean_locs = [loc + 10**-5 for loc in stimuli_mean_locs]
        xticks = ax.get_xticks()
        xticks = xticks + stimuli_mean_locs
        unique_stimuli = ["\n\n%s" % stimulus for stimulus in unique_stimuli]
        labels = labels_genotype_only + unique_stimuli
        ax.set_xticks(xticks)
        ax.set_xticklabels(labels)
        ax.set_xlabel("")

        # for label in ax.get_xticklabels():
        #     if label.get_text() in labels_genotype_only:
        #         label.set_rotation(90)

        # Get all xticks
        xticks = ax.xaxis.get_major_ticks()

        # Remove the tick lines for the last three xticks
        for tick in xticks[len(labels_genotype_only):]:
            tick.tick1line.set_visible(False)
            tick.tick2line.set_visible(False)

        # sns.move_legend(ax, bbox_to_anchor=(1, 0.5), title=None, frameon=False, loc="center left")
        sns.move_legend(ax, bbox_to_anchor=(0.5, 1), title=None, frameon=False, loc="lower center", ncol=2)
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

def plot_predictions_one_plot(ifnb_predicted_1_1, ifnb_predicted_1_3, ifnb_predicted_3_1, ifnb_predicted_3_3, beta, conditions, name, figures_dir, hn = []):
    df_ifnb_predicted_1_1 = make_predictions_data_frame(ifnb_predicted_1_1, beta, conditions)
    df_ifnb_predicted_1_3 = make_predictions_data_frame(ifnb_predicted_1_3, beta, conditions)
    df_ifnb_predicted_3_1 = make_predictions_data_frame(ifnb_predicted_3_1, beta, conditions)
    df_ifnb_predicted_3_3 = make_predictions_data_frame(ifnb_predicted_3_3, beta, conditions)

    data_df = df_ifnb_predicted_1_1.loc[df_ifnb_predicted_1_1["par_set"] == "Data"].copy()

    df_sym = pd.concat([df_ifnb_predicted_1_1, 
                        df_ifnb_predicted_3_1,
                        df_ifnb_predicted_1_3,
                        df_ifnb_predicted_3_3], ignore_index=True)
    df_sym[r"H_{I_2}"] = np.concatenate([np.repeat("1", len(df_ifnb_predicted_1_1)), np.repeat("3", len(df_ifnb_predicted_1_3)),
                                        np.repeat("1", len(df_ifnb_predicted_3_1)), np.repeat("3", len(df_ifnb_predicted_3_3))])
    df_sym[r"H_{I_1}"] = np.concatenate([np.repeat("1", len(df_ifnb_predicted_1_1)), np.repeat("1", len(df_ifnb_predicted_1_3)),
                                        np.repeat("3", len(df_ifnb_predicted_3_1)), np.repeat("3", len(df_ifnb_predicted_3_3))])
    if len(hn) > 0:
        df_sym[r"H_N"] = np.concatenate([np.repeat(hn[0], len(df_ifnb_predicted_1_1)), np.repeat(hn[1], len(df_ifnb_predicted_1_3)),
                                        np.repeat(hn[2], len(df_ifnb_predicted_3_1)), np.repeat(hn[3], len(df_ifnb_predicted_3_3))])
        df_sym["Hill"] = r"$h_{I_1}$=" + df_sym[r"H_{I_1}"] + r", $h_{I_2}$=" + df_sym[r"H_{I_2}"] + r", $h_N$=" + df_sym[r"H_N"] + r""
    else:
        df_sym["Hill"] = r"$h_{I_1}$=" + df_sym[r"H_{I_1}"] + r", $h_{I_2}$=" + df_sym[r"H_{I_2}"] + r""
    
    data_df[r"H_{I_2}"] = np.repeat("Data", len(data_df))
    data_df[r"H_{I_1}"] = np.repeat("", len(data_df))
    data_df["Hill"] = "Exp."

    df_sym = df_sym.loc[df_sym["par_set"] != "Data"] # contains duplicate data points
    df_all = pd.concat([df_sym, data_df], ignore_index=True)

    hill_categories = np.concatenate([data_df["Hill"].unique(), df_sym["Hill"].unique()])

    df_all["Hill"] = pd.Categorical(df_all["Hill"], categories=hill_categories, ordered=True)
    # print(df_all)
    # print(df_all["Data point"].unique())

    # Plot separately
    for category in df_all["Category"].unique():
        print("Plotting %s" % category)
        with sns.plotting_context("paper", rc=plot_rc_pars):
            num_bars = len(df_all[df_all["Category"]==category]["Data point"].unique())
            width  = 3.1*num_bars/3/2.1
            height = 1.3/1.7
            fig, ax = plt.subplots(figsize=(width, height))
            cols = [data_color] + sns.color_palette(models_cmap_pars, n_colors=4)
            sns.barplot(data=df_all[df_all["Category"]==category], x="Data point", y=r"IFN$\beta$", hue="Hill", 
                        palette=cols, ax=ax, width=0.8, errorbar=None, legend=False, saturation=0.9)
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
        # print(df_all[df_all["Category"]==category])
        num_bars = len(df_all[df_all["Category"]==category]["Data point"].unique())
        width  = 3.1*num_bars/3/2.1 + 0.5
        height = 1.3/1.7
        fig, ax = plt.subplots(figsize=(width, height))
        cols = [data_color] + sns.color_palette(models_cmap_pars, n_colors=4)
        sns.barplot(data=df_all[df_all["Category"]==category], x="Data point", y=r"IFN$\beta$", hue="Hill", 
                    palette=cols, ax=ax, width=0.8, errorbar=None, saturation=0.9)
        ax.set_xlabel("")
        ax.set_ylabel(r"IFN$\beta$")
        # ax.set_title(category)
        sns.despine()
        print([item.get_text().split(" ") for item in ax.get_xticklabels()])
        ax, _ = fix_ax_labels(ax)
        plt.tight_layout(pad=0)
        plt.ylim(0,1)
        sns.move_legend(ax, bbox_to_anchor=(1,1), title=None, frameon=False, loc="upper left", ncol=1)
        plt.savefig("%s/%s_legend.png" % (figures_dir, name), bbox_inches="tight")
        plt.close()

def make_parameters_data_frame(pars):
    df_pars = pars.drop(columns=["h1", "h2", "h3", "rmsd"], errors="ignore")

    df_pars["par_set"] = np.arange(len(df_pars))
    df_pars["t_0"] = 0
    df_pars["t_I1I2N"] = 1
    df_pars = df_pars.melt(var_name="Parameter", value_name="Value", id_vars="par_set")
    # df_t_pars = df_pars[df_pars["Parameter"].str.startswith("t")]
    df_t_pars = df_pars.loc[df_pars["Parameter"].str.startswith("t")].copy()
    num_t_pars = len(df_t_pars["Parameter"].unique())
    new_t_par_names = [r"$t_{I}$", r"$t_{I}$", r"$t_N$", r"$t_{I_1I_2}$", r"$t_{I_1N}$"]
    # Rename t parameters
    df_t_pars["Parameter"] = df_t_pars["Parameter"].replace(["t1", "t2", "t3", "t4", "t5"], new_t_par_names)
    df_t_pars["Parameter"] = df_t_pars["Parameter"].replace(["t_1", "t_2", "t_3", "t_4", "t_5"], new_t_par_names)
    df_t_pars["Parameter"] = df_t_pars["Parameter"].replace(["t_0", "t_I1I2N"], [r"$t_0$", r"$t_{I_1I_2N}$"])
    new_t_par_order = [r"$t_0$",r"$t_{I}$", r"$t_N$", r"$t_{I_1I_2}$", r"$t_{I_1N}$", r"$t_{I_1I_2N}$"]
    df_t_pars["Parameter"] = pd.Categorical(df_t_pars["Parameter"], categories=new_t_par_order, ordered=True)

    df_k_pars = df_pars.loc[df_pars["Parameter"].str.startswith("k") | df_pars["Parameter"].str.startswith("c")].copy()
    num_k_pars = len(df_k_pars["Parameter"].unique())
    # df_k_pars["Parameter"] = df_k_pars["Parameter"].str.replace("k3", r"$k_N$")
    # df_k_pars["Parameter"] = df_k_pars["Parameter"].str.replace("k2", r"$k_2$")
    # df_k_pars["Parameter"] = df_k_pars["Parameter"].str.replace("k1", r"$k_1$")
    # df_k_pars["Parameter"] = df_k_pars["Parameter"].str.replace("kn", r"$k_N$")
    df_k_pars.loc[df_k_pars["Parameter"] == "k1", "Parameter"] = r"$k_{I_2}$" # Rename
    df_k_pars.loc[df_k_pars["Parameter"] == "k2", "Parameter"] = r"$k_{I_1}$" # Rename
    df_k_pars.loc[df_k_pars["Parameter"] == "kn", "Parameter"] = r"$K_N$"
    df_k_pars.loc[df_k_pars["Parameter"] == "k3", "Parameter"] = r"$K_N$"
    df_k_pars.loc[df_k_pars["Parameter"] == "kp", "Parameter"] = r"$K_P$"
    df_k_pars.loc[df_k_pars["Parameter"] == "k4", "Parameter"] = r"$K_P$"
    df_k_pars.loc[df_k_pars["Parameter"] == "c", "Parameter"] = r"$C$"
    df_k_pars["Parameter"] = pd.Categorical(df_k_pars["Parameter"], categories=[r"$k_{I_1}$", r"$k_{I_2}$", r"$K_N$", r"$K_P$",r"$C$"], ordered=True)
    return df_t_pars, df_k_pars, num_t_pars, num_k_pars

def combine_parameters_data_frame(pars_1_1, pars_1_3, pars_3_1, pars_3_3):
    df_t_pars_1_1, df_k_pars_1_1, _, _ = make_parameters_data_frame(pars_1_1)
    df_t_pars_1_3, df_k_pars_1_3, _, _ = make_parameters_data_frame(pars_1_3)
    df_t_pars_3_1, df_k_pars_3_1, _, _ = make_parameters_data_frame(pars_3_1)
    df_t_pars_3_3, df_k_pars_3_3, num_t_pars, num_k_pars = make_parameters_data_frame(pars_3_3)

    df_all_t_pars = pd.concat([df_t_pars_1_1, df_t_pars_1_3, df_t_pars_3_1, df_t_pars_3_3], ignore_index=True)
    df_all_k_pars = pd.concat([df_k_pars_1_1, df_k_pars_1_3, df_k_pars_3_1, df_k_pars_3_3], ignore_index=True)

    df_all_t_pars[r"H_{I_2}"] = np.concatenate([np.repeat("1", len(df_t_pars_1_1)), np.repeat("1", len(df_t_pars_1_3)),
                                        np.repeat("3", len(df_t_pars_3_1)), np.repeat("3", len(df_t_pars_3_3))])
    df_all_t_pars[r"H_{I_1}"] = np.concatenate([np.repeat("1", len(df_t_pars_1_1)), np.repeat("3", len(df_t_pars_1_3)),
                                        np.repeat("1", len(df_t_pars_3_1)), np.repeat("3", len(df_t_pars_3_3))])
    df_all_t_pars["Model"] = r"$h_{I_1}$=" + df_all_t_pars[r"H_{I_1}"] + r", $h_{I_2}$=" + df_all_t_pars[r"H_{I_2}"]
    # Filter for Hi1=1,Hi2=1 and Hi1=1,Hi2=3
    df_all_t_pars = df_all_t_pars.loc[(df_all_t_pars["H_{I_1}"] == "1") & (df_all_t_pars["H_{I_2}"] == "1") |
                                        (df_all_t_pars["H_{I_1}"] == "1") & (df_all_t_pars["H_{I_2}"] == "3")]

    df_all_k_pars[r"H_{I_2}"] = np.concatenate([np.repeat("1", len(df_k_pars_1_1)), np.repeat("1", len(df_k_pars_1_3)),
                                        np.repeat("3", len(df_k_pars_3_1)), np.repeat("3", len(df_k_pars_3_3))])
    df_all_k_pars[r"H_{I_1}"] = np.concatenate([np.repeat("1", len(df_k_pars_1_1)), np.repeat("3", len(df_k_pars_1_3)),
                                        np.repeat("1", len(df_k_pars_3_1)), np.repeat("3", len(df_k_pars_3_3))])
    df_all_k_pars["Model"] = r"$h_{I_1}$=" + df_all_k_pars[r"H_{I_1}"] + r", $h_{I_2}$=" + df_all_k_pars[r"H_{I_2}"]
    # Filter for Hi1=1,Hi2=1 and Hi1=1,Hi2=3
    df_all_k_pars = df_all_k_pars.loc[(df_all_k_pars["H_{I_1}"] == "1") & (df_all_k_pars["H_{I_2}"] == "1") |
                                        (df_all_k_pars["H_{I_1}"] == "1") & (df_all_k_pars["H_{I_2}"] == "3")]
    
    return df_all_t_pars, df_all_k_pars, num_t_pars, num_k_pars

def make_ki_plot(df_ki_pars, name, figures_dir):
    # Filter for parameter containing "I"
    df_ki_pars = df_ki_pars.loc[df_ki_pars["Parameter"].str.contains("I")]

    IRF_array = np.arange(0, 1.1, 0.05)
    # Duplicate the dataframe for each value of IRF
    df_ki_pars = pd.concat([df_ki_pars]*len(IRF_array), ignore_index=True)
    df_ki_pars["IRF"] = np.repeat(IRF_array, len(df_ki_pars)/len(IRF_array))

    df_ki_pars[r"$H_{I_1}$"] = df_ki_pars["H_{I_1}"].astype(int)
    df_ki_pars[r"$H_{I_2}$"] = df_ki_pars["H_{I_2}"].astype(int)
    # H_I is equal to value of HI1 when parameter is k_I1 and HI2 when parameter is k_I2
    df_ki_pars[r"$h_I$"] = np.where(df_ki_pars["Parameter"] == r"$k_{I_1}$", df_ki_pars[r"$H_{I_1}$"], 
                                    np.where(df_ki_pars["Parameter"] == r"$k_{I_2}$",
                                             df_ki_pars[r"$H_{I_2}$"], None))
    df_ki_pars[r"$K_I$"] = df_ki_pars["Value"]*df_ki_pars["IRF"]**(df_ki_pars[r"$h_I$"]-1)

    df_ki_pars["Parameter"] = df_ki_pars["Parameter"].cat.remove_unused_categories()

    colors = sns.color_palette(models_cmap_pars, n_colors=4)

    # plot with different colors for each value of h_I, x-axis is IRF, y-axis is K_I, facet by Parameter
    with sns.plotting_context("paper", rc=plot_rc_pars):
        g = sns.FacetGrid(df_ki_pars, col="Parameter", hue="Model", palette=colors, col_wrap=2, height=1.5, aspect=.2)
        g.map(sns.lineplot, "IRF", r"$K_I$", zorder = 0,  errorbar=None, estimator=None, alpha=0.2, units="par_set", data=df_ki_pars)
        g.set_axis_labels(r"$[IRF]$ (MNU)", r"$k [IRF]^{h_I-1}$ (MNU$^{-1}$)")
        g.set_titles(r"$k=$" +"{col_name}")
        g.add_legend(title="", ncol=4,loc="lower center", bbox_to_anchor=(0.5, 1), frameon=False)
        g.despine()
        leg = g._legend
        for line in leg.get_lines():
            line.set_alpha(1)
        plt.tight_layout()
        plt.savefig("%s/%s.png" % (figures_dir, name), bbox_inches="tight")
        plt.close()

# Plot parameters one plot
def plot_parameters_one_plot(pars_1_1, pars_1_3, pars_3_1, pars_3_3, name, figures_dir):
    df_t_pars_1_1, df_k_pars_1_1, _, _ = make_parameters_data_frame(pars_1_1)
    df_t_pars_1_3, df_k_pars_1_3, _, _ = make_parameters_data_frame(pars_1_3)
    df_t_pars_3_1, df_k_pars_3_1, _, _ = make_parameters_data_frame(pars_3_1)
    df_t_pars_3_3, df_k_pars_3_3, num_t_pars, num_k_pars = make_parameters_data_frame(pars_3_3)

    df_all_t_pars = pd.concat([df_t_pars_1_1, df_t_pars_1_3, df_t_pars_3_1, df_t_pars_3_3], ignore_index=True)
    df_all_k_pars = pd.concat([df_k_pars_1_1, df_k_pars_1_3, df_k_pars_3_1, df_k_pars_3_3], ignore_index=True)

    df_all_t_pars[r"H_{I_2}"] = np.concatenate([np.repeat("1", len(df_t_pars_1_1)), np.repeat("1", len(df_t_pars_1_3)),
                                        np.repeat("3", len(df_t_pars_3_1)), np.repeat("3", len(df_t_pars_3_3))])
    df_all_t_pars[r"H_{I_1}"] = np.concatenate([np.repeat("1", len(df_t_pars_1_1)), np.repeat("3", len(df_t_pars_1_3)),
                                        np.repeat("1", len(df_t_pars_3_1)), np.repeat("3", len(df_t_pars_3_3))])
    df_all_t_pars["Model"] = r"$h_{I_1}$=" + df_all_t_pars[r"H_{I_1}"] + r", $h_{I_2}$=" + df_all_t_pars[r"H_{I_2}"]

    df_all_k_pars[r"H_{I_2}"] = np.concatenate([np.repeat("1", len(df_k_pars_1_1)), np.repeat("1", len(df_k_pars_1_3)),
                                        np.repeat("3", len(df_k_pars_3_1)), np.repeat("3", len(df_k_pars_3_3))])
    df_all_k_pars[r"H_{I_1}"] = np.concatenate([np.repeat("1", len(df_k_pars_1_1)), np.repeat("3", len(df_k_pars_1_3)),
                                        np.repeat("1", len(df_k_pars_3_1)), np.repeat("3", len(df_k_pars_3_3))])
    df_all_k_pars["Model"] = r"$h_{I_1}$=" + df_all_k_pars[r"H_{I_1}"] + r", $h_{I_2}$=" + df_all_k_pars[r"H_{I_2}"]

    colors = sns.color_palette(models_cmap_pars, n_colors=4)

    k_parameters = [r"$K_N$",r"$K_P$"]

    with sns.plotting_context("paper",rc=plot_rc_pars):
        width = 2.8
        height = 1
        fig, ax = plt.subplots(1,2, figsize=(width, height), 
                               gridspec_kw={"width_ratios":[num_t_pars, 2.5]})
        
        unique_models = np.unique(df_all_t_pars["Model"])
        legend_handles = []

        s = sns.stripplot(data=df_all_t_pars, x="Parameter", y="Value", hue = "Model", palette=colors, ax=ax[0], zorder = 0, linewidth=0,
                            alpha=0.2, jitter=0, dodge=True, legend=False)
        
        legend_handles = s.collections
       

        df2 = df_all_k_pars[(df_all_k_pars["Parameter"].isin(k_parameters))]
        df2 = df2.copy()
        df2["Parameter"] = df2["Parameter"].cat.remove_unused_categories()
        s = sns.stripplot(data=df2, x="Parameter", y="Value", hue = "Model", palette=colors, ax=ax[1], zorder = 0, linewidth=0, 
                          alpha=0.2, jitter=0, dodge=True, legend=False)        
       
        ax[1].set_yscale("log")
        ax[1].set_ylabel(r"Value (MNU$^{-1}$)")
        
        ax1_xtick_labels = ax[1].get_xticklabels()
        # Replace ", " with "\n" in xtick labels
        new_xtick_labels = [label.get_text().replace(", $h_{I_2}$=", "\n") for label in ax1_xtick_labels]
        new_xtick_labels = [label.replace("$h_{I_1}$=", "") for label in new_xtick_labels]

        ax[1].set_xticklabels(new_xtick_labels)

        ax[0].set_ylabel("Parameter Value")
        
        for x in ax[0], ax[1]:
            x.set_xlabel("")

        sns.despine()
        plt.tight_layout()
        leg = fig.legend(legend_handles, unique_models, loc="lower center", bbox_to_anchor=(0.5, 1), frameon=False, 
                         ncol=4, columnspacing=1, handletextpad=0.5, handlelength=1.5)

        for i in range(4):
            leg.legend_handles[i].set_alpha(1)
            leg.legend_handles[i].set_color(colors[i])

        plt.savefig("%s/%s.png" % (figures_dir, name), bbox_inches="tight")
        plt.close()

        make_ki_plot(df_all_k_pars, name + "_k_i", figures_dir)

# def plot_parameters_one_plot(pars_1_1, pars_1_3, pars_3_1, pars_3_3, name, figures_dir):
#     df_t_pars_1_1, df_k_pars_1_1, _, _ = make_parameters_data_frame(pars_1_1)
#     df_t_pars_1_3, df_k_pars_1_3, _, _ = make_parameters_data_frame(pars_1_3)
#     df_t_pars_3_1, df_k_pars_3_1, _, _ = make_parameters_data_frame(pars_3_1)
#     df_t_pars_3_3, df_k_pars_3_3, num_t_pars, num_k_pars = make_parameters_data_frame(pars_3_3)

#     df_all_t_pars = pd.concat([df_t_pars_1_1, df_t_pars_1_3, df_t_pars_3_1, df_t_pars_3_3], ignore_index=True)
#     df_all_k_pars = pd.concat([df_k_pars_1_1, df_k_pars_1_3, df_k_pars_3_1, df_k_pars_3_3], ignore_index=True)

#     df_all_t_pars[r"H_{I_2}"] = np.concatenate([np.repeat("1", len(df_t_pars_1_1)), np.repeat("1", len(df_t_pars_1_3)),
#                                         np.repeat("3", len(df_t_pars_3_1)), np.repeat("3", len(df_t_pars_3_3))])
#     df_all_t_pars[r"H_{I_1}"] = np.concatenate([np.repeat("1", len(df_t_pars_1_1)), np.repeat("3", len(df_t_pars_1_3)),
#                                         np.repeat("1", len(df_t_pars_3_1)), np.repeat("3", len(df_t_pars_3_3))])
#     df_all_t_pars["Model"] = r"$h_{I_1}$=" + df_all_t_pars[r"H_{I_1}"] + r", $h_{I_2}$=" + df_all_t_pars[r"H_{I_2}"]

#     df_all_t_pars = df_all_t_pars.loc[df_all_t_pars[r"H_{I_1}"]  == "1"] # acceptable models only

#     df_all_k_pars[r"H_{I_2}"] = np.concatenate([np.repeat("1", len(df_k_pars_1_1)), np.repeat("1", len(df_k_pars_1_3)),
#                                         np.repeat("3", len(df_k_pars_3_1)), np.repeat("3", len(df_k_pars_3_3))])
#     df_all_k_pars[r"H_{I_1}"] = np.concatenate([np.repeat("1", len(df_k_pars_1_1)), np.repeat("3", len(df_k_pars_1_3)),
#                                         np.repeat("1", len(df_k_pars_3_1)), np.repeat("3", len(df_k_pars_3_3))])
#     df_all_k_pars["Model"] = r"$h_{I_1}$=" + df_all_k_pars[r"H_{I_1}"] + r", $h_{I_2}$=" + df_all_k_pars[r"H_{I_2}"]

#     df_all_k_pars = df_all_k_pars.loc[df_all_k_pars[r"H_{I_1}"]  == "1"] # acceptable models only

#     colors = sns.color_palette(models_cmap_pars, n_colors=4)

#     with sns.plotting_context("paper",rc=plot_rc_pars):
#         width = 2.8
#         height = 1
#         fig, ax = plt.subplots(1,2, figsize=(width, height), 
#                                gridspec_kw={"width_ratios":[num_t_pars, num_k_pars]})
#         # sns.lineplot(data=df_all_t_pars, x="Parameter", y="Value", hue="Model", ax=ax[0], palette=colors, zorder = 0, errorbar=None)
#         # sns.scatterplot(data=df_all_t_pars, x="Parameter", y="Value", hue="Model", ax=ax[0], palette=colors, legend=False, zorder = 1, linewidth=0)
#         # sns.lineplot(data=df_all_k_pars, x="Parameter", y="Value", hue="Model", ax=ax[1], palette=colors, legend=False, errorbar=None,
#         #                 zorder = 0)
#         # sns.scatterplot(data=df_all_k_pars, x="Parameter", y="Value", hue="Model", ax=ax[1], palette=colors, legend=False, zorder = 1, linewidth=0)
        
#         unique_models = np.unique(df_all_t_pars["Model"])
#         legend_handles = []
#         for i, model in enumerate(unique_models):
#             # Filter data for the current model
#             df_model = df_all_t_pars[df_all_t_pars["Model"] == model]
#             l = sns.lineplot(data=df_model, x="Parameter", y="Value", color=colors[i], ax=ax[0], zorder = i, errorbar=None, estimator=None, alpha=0.2, units="par_set")
#             sns.scatterplot(data=df_model, x="Parameter", y="Value", color=colors[i], ax=ax[0], legend=False, zorder = i+0.5, linewidth=0)

#             legend_handles.append(l.lines[0])

#             df_model = df_all_k_pars[df_all_k_pars["Model"] == model]
#             sns.lineplot(data=df_model, x="Parameter", y="Value", color=colors[i], ax=ax[1], zorder = i, errorbar=None, estimator=None, alpha=0.2, units="par_set")
#             sns.scatterplot(data=df_model, x="Parameter", y="Value", color=colors[i], ax=ax[1], legend=False, zorder = i+0.5, linewidth=0, alpha=0.2)
        
#         ax[1].set_yscale("log")
#         ax[1].set_ylabel("")

#         ax[0].set_ylabel("Parameter Value")
#         ax[0].set_xlabel("")
#         ax[1].set_xlabel("")

#         sns.despine()
#         plt.tight_layout()
#         # handles, labels = ax[0].get_legend_handles_labels()

#         # leg = fig.legend(legend_handles, unique_models, loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
#         leg = fig.legend(legend_handles, unique_models, loc="lower center", bbox_to_anchor=(0.5, 1), frameon=False, 
#                          ncol=4, columnspacing=1, handletextpad=0.5, handlelength=1.5)

#         for i in range(len(unique_models)):
#             leg.legend_handles[i].set_alpha(1)
#             leg.legend_handles[i].set_color(colors[i])


#         # fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 1), ncol=2, frameon=False)
#         # ax[0].get_legend().remove()
#         plt.savefig("%s/%s.png" % (figures_dir, name), bbox_inches="tight")
#         plt.close()
    
    
def plot_rmsd_boxplot(all_opt_rmsd, model, figures_dir):
    all_opt_rmsd["Hill"] = pd.Categorical(all_opt_rmsd["Hill"], ordered=True)
    all_opt_rmsd = all_opt_rmsd.sort_values("Hill")

    # remove rows where RMSD == "rmsd_initial"
    rmsd_df = all_opt_rmsd.copy()
    fig, ax = plt.subplots(figsize=(6,6))
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

    colors = sns.color_palette(models_cmap_pars, n_colors=4)
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
    training_data = pd.read_csv("../data/p50_training_data.csv")
    beta = training_data["IFNb"]
    conditions = training_data["Stimulus"] + "_" + training_data["Genotype"]
    force_t_dir = "parameter_scan_force_t/"
    model_t = "p50_force_t"

    # Plot predictions on one plot
    print("Plotting predictions for all hill combinations on one plot", flush=True)
    predictions_force_t_1_1 = np.loadtxt("%s/%s_best_fits_ifnb_predicted.csv" % ("%s/results_h_1_1_1/" % force_t_dir, model_t), delimiter=",")
    predictions_force_t_1_3 = np.loadtxt("%s/%s_best_fits_ifnb_predicted.csv" % ("%s/results_h_1_3_1/" % force_t_dir, model_t), delimiter=",")
    predictions_force_t_3_1 = np.loadtxt("%s/%s_best_fits_ifnb_predicted.csv" % ("%s/results/" % force_t_dir, model_t), delimiter=",")
    predictions_force_t_3_3 = np.loadtxt("%s/%s_best_fits_ifnb_predicted.csv" % ("%s/results_h_3_3_1/" % force_t_dir, model_t), delimiter=",")
    plot_predictions_one_plot(predictions_force_t_1_1, predictions_force_t_1_3, predictions_force_t_3_1, predictions_force_t_3_3, beta, conditions, "best_20_ifnb_force_t", figures_dir)
    del predictions_force_t_1_1, predictions_force_t_1_3, predictions_force_t_3_1, predictions_force_t_3_3        

    # Plot parameters on one plot
    print("Plotting best-fit parameters for all hill combinations on one plot", flush=True)
    best_20_pars_df_1_1 = pd.read_csv("%s/%s_best_fits_pars.csv" % ("%s/results_h_1_1_1/" % force_t_dir, model_t))
    best_20_pars_df_1_3 = pd.read_csv("%s/%s_best_fits_pars.csv" % ("%s/results_h_1_3_1/" % force_t_dir, model_t))
    best_20_pars_df_3_1 = pd.read_csv("%s/%s_best_fits_pars.csv" % ("%s/results/" % force_t_dir, model_t))
    best_20_pars_df_3_3 = pd.read_csv("%s/%s_best_fits_pars.csv" % ("%s/results_h_3_3_1/" % force_t_dir, model_t))
    plot_parameters_one_plot(best_20_pars_df_1_1, best_20_pars_df_1_3, best_20_pars_df_3_1, best_20_pars_df_3_3, "best_20_pars_force_t_all", figures_dir)
    del best_20_pars_df_1_1, best_20_pars_df_1_3, best_20_pars_df_3_1, best_20_pars_df_3_3

    # Add RMSD density plot of two models: h1=1,h2=1 and h1=1,h2=3
    print("Plotting RMSD density plot for two hill combinations", flush=True)
    rmsd_1_1 = pd.read_csv("%s/%s_rmsd_optimized.csv" % ("%s/results_h_1_1_1/" % force_t_dir, model_t))
    rmsd_3_1 = pd.read_csv("%s/%s_rmsd_optimized.csv" % ("%s/results/" % force_t_dir, model_t))
    rmsd = pd.concat([rmsd_1_1, rmsd_3_1], ignore_index=True)
    rmsd[r"H_{I_1}"] = np.concatenate([np.repeat("1", len(rmsd_1_1) + len(rmsd_3_1))])
    rmsd[r"H_{I_2}"] = np.concatenate([np.repeat("1", len(rmsd_1_1)), np.repeat("3", len(rmsd_3_1))])
    rmsd["Hill"] = r"$h_{I_1}$=" + rmsd[r"H_{I_1}"] + r", $h_{I_2}$=" + rmsd[r"H_{I_2}"]
    rmsd = rmsd.loc[rmsd["rmsd_type"] == "rmsd_final"]

    with sns.plotting_context("paper", rc=plot_rc_pars):
        fig, ax = plt.subplots(figsize=(1.6,1.1))
        p = sns.kdeplot(data=rmsd, x="RMSD", hue="Hill", fill=True, common_norm=False, palette=sns.color_palette(models_cmap_pars, n_colors=4), ax=ax)
        ax.set_xlabel("RMSD")
        ax.set_ylabel("Density")
        sns.despine()
        sns.move_legend(ax, bbox_to_anchor=(0,1), title=None, frameon=False, loc="upper left", ncol=1)
        plt.tight_layout()
        plt.savefig("%s/rmsd_good_models_density_plot.png" % figures_dir)


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

    cmap = heatmap_cmap

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
    state_probs_df["Genotype"] = state_probs_df["Genotype"].replace("relacrelKO", r"NFκBko")
    state_probs_df["Genotype"] = state_probs_df["Genotype"].replace("p50KO", "p50ko")
    state_probs_df["Genotype"] = state_probs_df["Genotype"].replace("irf3irf7KO", "IRF3/7ko")
    state_probs_df["Genotype"] = state_probs_df["Genotype"].replace("irf3irf5irf7KO", "IRF3/5/7ko")
    state_probs_df["Stimulus"] = state_probs_df["Stimulus"].replace("polyIC", "PolyIC")
    state_probs_df["Stimulus"] = state_probs_df["Stimulus"].replace("basal", "Basal")
    stimuli_levels = ["Basal", "CpG", "LPS", "PolyIC"]
    # stimuli_levels = ["PolyIC", "LPS", "CpG", "Basal"]
    genotypes_levels = ["WT","p50ko", "IRF3/7ko", "IRF3/5/7ko", r"NFκBko"]
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
    # categories = [lambda x: "NFκB dependence" if "rela" in x else "IRF dependence" if "irf" in x else "p50 dependence" if "p50" in x else "Stimulus Specific" for x in state_probs_df["Genotype"]]
    # state_probs_df["Category"] = state_probs_df["Genotype"].apply(lambda x: "NFκB dependence" if "NFκB" in x else "IRF dependence" if "IRF" in x else "p50 dependence" if "p50" in x else "Stimulus Specific")
    # category_dict = {state: category for state, category in zip(state_probs_df["state"].unique(), state_probs_df["Category"].unique())}

    # Plot state probabilities
    # Make dictionary of colors
    # genotype_colors = sns.cubehelix_palette(n_colors=len(genotypes_levels), start=-0.2, rot=0.65, dark=0.2, light=0.8, reverse=True)
    genotype_colors = sns.color_palette(states_cmap_pars, n_colors=len(genotypes_levels))
    genotype_colors = {genotype: color for genotype, color in zip(genotypes_levels, genotype_colors)}

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
    t_pars_df["Condition"] = ["Transcription \n" r"capability ($t$)"]
    t_pars_df = t_pars_df.set_index("Condition")

    # State probability
    state_probs_df["state"] = pd.Categorical(state_probs_df["state"], categories=state_probs_df["state"].unique(), ordered=True)
    state_probs_df = state_probs_df.pivot(index="Condition", columns="state", values="Probability")
    print(state_probs_df)

    # categories dict: NFκB dependence (contains NFκB or WT, LPS or pIC only), IRF dependence (contains IRF, LPS or pIC only),
    # p50 dependence (contains p50, LPS or CpG only), Stimulus Specific (contains WT )
    category_dict = {"NFκB dependence": [x for x in state_probs_df.index if ("NFκB" in x or "WT" in x) and ("LPS" in x or "PolyIC" in x)],
                    "IRF dependence": [x for x in state_probs_df.index if ("IRF" in x or "WT" in x) and ("LPS" in x or "PolyIC" in x)],
                    "p50 dependence": [x for x in state_probs_df.index if ("p50" in x or "WT" in x) and ("LPS" in x or "CpG" in x)],
                    "Stimulus Specific": [x for x in state_probs_df.index if "WT" in x]}
    # print(category_dict)

    # raise ValueError("Stop here")

    for category in category_dict:
        state_probs_df_category = state_probs_df.loc[category_dict[category]]
        # print(state_probs_df_category)

        n_columns = state_probs_df_category.shape[1]
        n_rows = state_probs_df_category.shape[0]

        # IFNb column
        ifnb_df = pd.read_csv("%s/%s_best_fits_ifnb_predicted.csv" % (results_dir, model), header=None, names=training_data["Stimulus"] + "_" + training_data["Genotype"]).mean()
        t_pars = best_fit_parameters.iloc[:4].values
        k_pars = best_fit_parameters.iloc[4:8].values
        h_pars = [3,1]
        ifnb_basal_wt = get_f(t_pars, k_pars, 0.01, 0.01, 1, h_pars=h_pars)
        ifnb_df.loc["basal_WT"] = ifnb_basal_wt
        ifnb_df = ifnb_df.rename(condition_renaming_dict)
        ifnb_df = ifnb_df.rename("")
        ifnb_df = ifnb_df.loc[state_probs_df_category.index]

        # Add a blank column
        blank_col = pd.DataFrame(np.nan, index=state_probs_df_category.index, columns=[" "])
        state_probs_df_category = pd.concat([state_probs_df_category, blank_col], axis=1)

        # Add the IFNb column
        state_probs_df_category = pd.concat([state_probs_df_category, ifnb_df], axis=1)

        # Add a blank row
        blank_row = pd.DataFrame(np.nan, index=[" "], columns=state_probs_df_category.columns)
        state_probs_df_category = pd.concat([state_probs_df_category, blank_row])

        # Add the t_pars row
        t_pars_df = t_pars_df.reindex(columns=state_probs_df_category.columns)
        state_probs_df_category = pd.concat([state_probs_df_category, t_pars_df])

        # heatmap_rc_pars = {"xtick.labelsize":6,"ytick.labelsize":6, "legend.fontsize":6}
        # new_rc_pars = plot_rc_pars.copy()
        # new_rc_pars.update(heatmap_rc_pars)
        
        with sns.plotting_context("paper", rc=plot_rc_pars):
            # Create the heatmap
            fig, ax = plt.subplots(figsize=(2, 2))
            # cbar_ax = fig.add_axes([.95, .3, .02, .5])
            sns.heatmap(data=state_probs_df_category, cbar=False, ax=ax, vmin=0, vmax=1, square=True, cmap=cmap)
            ax.set_ylabel("")
            ax.tick_params(axis="both", which="both", length=0)    
            plt.subplots_adjust(top=0.93, right=0.8, hspace=0.1, wspace=0.2)
            # Write r"IFN$\beta$ mRNA" to the right of the heatmap
            # ifnb_y_position = 1-(n_rows / (state_probs_df_category.shape[0] + 1) / 2)
            # ax.text(1.05, ifnb_y_position, r"IFN$\beta$ mRNA ($f$)", ha="center", va="center", rotation=270, fontsize=6, transform=ax.transAxes)
            # Title color bar
            # cbar_ax.set_title("Max-Normalized\n Transcription", fontsize=6)
            # Title heatmap
            title_x_position = n_columns / (state_probs_df_category.shape[1] + 1) / 2
            # ax.set_title("State Probabilities", fontsize=6, pad=4, x=title_x_position)

            # plt.tight_layout()
            cat = category.replace(" ", "_")
            plt.savefig("%s/state_probabilities_heatmap_%s.png" % (figures_dir, cat), bbox_inches="tight")
            plt.close()

            if category == "Stimulus Specific":
                # Create the heatmap
                fig, ax = plt.subplots(figsize=(2.2, 2))
                cbar_ax = fig.add_axes([.9, .3, .02, .5])
                sns.heatmap(data=state_probs_df_category, cbar_ax=cbar_ax, ax=ax, vmin=0, vmax=1, square=True, cmap=cmap)
                ax.set_ylabel("")
                ax.tick_params(axis="both", which="both", length=0)    
                plt.subplots_adjust(top=0.93, right=0.7, hspace=0.1, wspace=0.2)
                # Write r"IFN$\beta$ mRNA" to the right of the heatmap
                ifnb_y_position = 1-(n_rows / (state_probs_df_category.shape[0] + 1) / 2)
                ax.text(1.03, ifnb_y_position, r"IFN$\beta$ mRNA ($f$)", ha="center", va="center", rotation=270, fontsize=6, transform=ax.transAxes)
                # Title color bar
                cbar_ax.set_title("Max-Normalized\n Transcription", fontsize=6)
                # Title heatmap
                title_x_position = n_columns / (state_probs_df_category.shape[1] + 1) / 2
                ax.set_title("State Probabilities", fontsize=6, pad=4, x=title_x_position)

                # plt.tight_layout()
                cat = category.replace(" ", "_")
                plt.savefig("%s/state_probabilities_heatmap_cbar.png" % (figures_dir), bbox_inches="tight")
                plt.close()




### Supplemental plots ###
def plot_predictions(ifnb_predicted, beta, conditions, name, figures_dir, lines = True):
    df_ifnb_predicted = make_predictions_data_frame(ifnb_predicted, beta, conditions)
    col = data_color

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
    df_t_pars, df_k_pars, num_t_pars, num_k_pars = make_parameters_data_frame(pars)

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

def make_supplemental_plots():
    # Param scan
    figures_dir = "p50_final_figures/"
    os.makedirs(figures_dir, exist_ok=True)
    num_t_pars = 5
    num_k_pars = 3
    num_h_pars = 2

    model = "p50_force_t"
    training_data = pd.read_csv("../data/p50_training_data.csv")
    beta = training_data["IFNb"]
    conditions = training_data["Stimulus"] + "_" + training_data["Genotype"]
    num_threads = 4
    # h_scan_dir = "p50_param_scan_hill/results/seed_0/"
    h_values = ["3_1_1","1_1_1", "3_3_1", "1_3_1"]

    force_t_dir = "parameter_scan_force_t/"
    model_t = "p50_force_t"

    # best fit cooperativity model with different Hill combinations
    predictions_c_1_1 = np.loadtxt("%s/%s_best_fits_ifnb_predicted.csv" % ("%s/results_h_1_1_1_c_scan" % force_t_dir, model_t), delimiter=",")
    predictions_c_1_3 = np.loadtxt("%s/%s_best_fits_ifnb_predicted.csv" % ("%s/results_h_1_3_1_c_scan" % force_t_dir, model_t), delimiter=",")
    predictions_c_3_3 = np.loadtxt("%s/%s_best_fits_ifnb_predicted.csv" % ("%s/results_h_3_3_1_c_scan" % force_t_dir, model_t), delimiter=",")
    predictions_c_3_1 = np.loadtxt("%s/%s_best_fits_ifnb_predicted.csv" % ("%s/results_c_scan" % force_t_dir, model_t), delimiter=",")
    plot_predictions_one_plot(predictions_c_1_1, predictions_c_1_3, predictions_c_3_1, predictions_c_3_3, beta, conditions, 
                              "best_20_ifnb_c_scan", figures_dir)
    
    del predictions_c_1_1, predictions_c_1_3, predictions_c_3_1, predictions_c_3_3

    # best fit parameters from cooperativity model with different Hill combinations
    best_20_pars_df_c_1_1 = pd.read_csv("%s/%s_best_fits_pars.csv" % ("%s/results_h_1_1_1_c_scan" % force_t_dir, model_t))
    best_20_pars_df_c_1_3 = pd.read_csv("%s/%s_best_fits_pars.csv" % ("%s/results_h_1_3_1_c_scan" % force_t_dir, model_t))
    best_20_pars_df_c_3_1 = pd.read_csv("%s/%s_best_fits_pars.csv" % ("%s/results_c_scan" % force_t_dir, model_t))
    best_20_pars_df_c_3_3 = pd.read_csv("%s/%s_best_fits_pars.csv" % ("%s/results_h_3_3_1_c_scan" % force_t_dir, model_t))
    plot_parameters_one_plot(best_20_pars_df_c_1_1, best_20_pars_df_c_1_3, best_20_pars_df_c_3_1, best_20_pars_df_c_3_3, "best_20_pars_c_scan", figures_dir)

    del best_20_pars_df_c_1_1, best_20_pars_df_c_1_3, best_20_pars_df_c_3_1, best_20_pars_df_c_3_3
    
    # Pairwise plot of parameters
    pars_df_1_1 = pd.read_csv("%s/%s_best_fits_pars.csv" % ("%s/results_h_1_1_1/" % force_t_dir, model_t))
    pars_df_1_1["Model"] = r"$h_{I_1}$=1, $h_{I_2}$=1"
    pars_df_1_3 = pd.read_csv("%s/%s_best_fits_pars.csv" % ("%s/results_h_1_3_1/" % force_t_dir, model_t))
    pars_df_1_3["Model"] = r"$h_{I_1}$=3, $h_{I_2}$=1"
    pars_df_3_1 = pd.read_csv("%s/%s_best_fits_pars.csv" % ("%s/results/" % force_t_dir, model_t))
    pars_df_3_1["Model"] = r"$h_{I_1}$=1, $h_{I_2}$=3"
    pars_df_3_3 = pd.read_csv("%s/%s_best_fits_pars.csv" % ("%s/results_h_3_3_1/" % force_t_dir, model_t))
    pars_df_3_3["Model"] = r"$h_{I_1}$=3, $h_{I_2}$=3"
    df_pars = pd.concat([pars_df_1_1, pars_df_1_3, pars_df_3_1, pars_df_3_3], ignore_index=True)
    new_par_names = [r"$t_{I}$", r"$t_{I}$", r"$t_N$", r"$t_{I_1I_2}$", r"$t_{I_1N}$",r"$k_{I_2}$", r"$k_{I_1}$", r"$k_N$", r"$k_P$"]
    old_par_names = ["t_1", "t_2", "t_3", "t_4", "t_5", "k_1", "k_2", "k_n", "k_p"]
    old_par_names2 = ["t1", "t2", "t3", "t4", "t5", "k1", "k2", "kn", "kp"]
    rename_dict = {old:new for old, new in zip(old_par_names, new_par_names)}
    rename_dict2 = {old:new for old, new in zip(old_par_names2, new_par_names)}
    df_pars = df_pars.rename(columns=rename_dict)
    df_pars = df_pars.rename(columns=rename_dict2)
    df_pars = df_pars.drop(columns=["rmsd"])
    column_order = [r"$t_{I}$", r"$t_N$", r"$t_{I_1I_2}$", r"$t_{I_1N}$",r"$k_{I_1}$", r"$k_{I_2}$", r"$k_N$", r"$k_P$", "Model"]
    df_pars = df_pars[column_order]
    k_val_ranges = {}
    for k_val in [r"$k_{I_1}$", r"$k_{I_2}$", r"$k_N$", r"$k_P$"]:
        # print("k val: %s, min: %f, max: %f" % (k_val, df_pars[k_val].min(), df_pars[k_val].max()))
        # add k value ranges to the dict. round to nearest 10^x
        min = df_pars[k_val].min()
        max = df_pars[k_val].max()
        min = 10**np.floor(np.log10(min))
        max = 10**np.ceil(np.log10(max))
        if max > 10**4-1:
            max = 10**5
        k_val_ranges[k_val] = (min, max)
        print("k val: %s, min: %f, max: %f" % (k_val, min, max))

    rc = {"axes.labelsize":25,"xtick.labelsize":20,"ytick.labelsize":20, "legend.fontsize":20}
    with sns.plotting_context("talk", rc=rc):
        g = sns.pairplot(df_pars, hue="Model", palette=sns.color_palette(models_cmap_pars, n_colors=4), diag_kind="kde", height=2, aspect=1)
        for i, ax in enumerate(g.axes.flatten()):
            # Skip the diagonal axes
            if i % (g.axes.shape[0] + 1) == 0:
                continue
            for k_val, rgs in k_val_ranges.items():
                if k_val in ax.get_xlabel():
                    ax.set_xlim(rgs)
                    ax.set_xscale("log")
                if k_val in ax.get_ylabel():
                    ax.set_ylim(rgs)
                    ax.set_yscale("log")
            if "t" in ax.get_xlabel():
                ax.set_xlim(0-0.1,1+0.1)
            if "t" in ax.get_ylabel():
                ax.set_ylim(0-0.1,1+0.1)
        # g._legend.remove()
        sns.move_legend(g, bbox_to_anchor=(1,0.5), title=None, frameon=False, loc="upper left")
        plt.tight_layout()
        plt.savefig("%s/pairwise_parameters.png" % figures_dir)
        plt.close()


    # # RMSD distribution for all hill combinations (select top 20 for each)
    # print("Plotting RMSD distributions for all hill combinations", flush=True)
    # all_best_rmsd = pd.DataFrame(columns=["rmsd", r"$h_{I_1}$", r"$h_{I_2}$"])
    # for row in h_values:
    #     h_vals_str = row
    #     if row == "3_1_1":
    #         dir = "%s/results/" % force_t_dir
    #     else:
    #         dir = "%s/results_h_%s/" % (force_t_dir, h_vals_str)
    #     rmsd_df = pd.read_csv("%s/%s_rmsd.csv" % (dir, model))
    #     h1, h2, _ = row.split("_")
    #     rmsd_df[r"$h_{I_1}$"] = h2
    #     rmsd_df[r"$h_{I_2}$"] = h1
    #     all_best_rmsd = pd.concat([all_best_rmsd, rmsd_df], ignore_index=True)
    #     del rmsd_df

    # with sns.plotting_context("talk"):
    #     col = sns.color_palette("rocket", n_colors=4)[1]
    #     sns.displot(data=all_best_rmsd, x="rmsd", row=r"$h_{I_1}$", col=r"$h_{I_2}$", kind="kde", fill=True, alpha=0.5, color=col)
    #     sns.despine()
    #     plt.tight_layout()
    #     plt.xlabel("RMSD")
    #     plt.savefig("%s/%s_rmsd_distributions_top_par_scan.png" % (figures_dir, model))
    #     plt.close()
    
    #     # Plot box plot of rmsd
    #     all_best_rmsd["Hill"] = all_best_rmsd[r"$h_{I_1}$"] + "_" + all_best_rmsd[r"$h_{I_2}$"] 
    #     plot_rmsd_boxplot(all_best_rmsd, model, figures_dir)
    #     del all_best_rmsd

    #     # Best fit model for h=1_1_1, force t constraint
    
    #     dir = "%s/results_h_1_1_1/" % force_t_dir
    #     print("Plotting predictions for best fit model with h=1_1_1, force t constraint", flush=True)
    #     predictions_force_t = np.loadtxt("%s/%s_best_fits_ifnb_predicted.csv" % (dir, model_t), delimiter=",")
    #     plot_predictions(predictions_force_t, beta, conditions, "best_20_ifnb_force_t_lines_1_1", figures_dir, lines=True)
    #     del predictions_force_t

    #     # Plot best-fit parameters for h=1_1_1, force t constraint
    #     print("Plotting best-fit parameters for h=1_1_1, force t constraint", flush=True)
    #     best_20_pars_df = pd.read_csv("%s/%s_best_fits_pars.csv" % (dir, model_t))
    #     plot_parameters(best_20_pars_df, "best_20_pars_force_t_1_1", figures_dir)

    #     # Plot best-fit model for h=3_3_1, force t constraint
    #     dir = "%s/results_h_3_3_1/" % force_t_dir
    #     print("Plotting predictions for best fit model with h=3_3_1, force t constraint", flush=True)
    #     predictions_force_t = np.loadtxt("%s/%s_best_fits_ifnb_predicted.csv" % (dir, model_t), delimiter=",")
    #     plot_predictions(predictions_force_t, beta, conditions, "best_20_ifnb_force_t_lines_3_3", figures_dir, lines=True)
    #     del predictions_force_t

    #     # Best fit model for h=3_1_1, force t constraint
    #     dir = "%s/results/" % force_t_dir
    #     print("Plotting predictions for best fit model with h=3_1_1, force t constraint", flush=True)
    #     predictions_force_t = np.loadtxt("%s/%s_best_fits_ifnb_predicted.csv" % (dir, model_t), delimiter=",")
    #     plot_predictions(predictions_force_t, beta, conditions, "best_20_ifnb_force_t_lines_3_1", figures_dir, lines=True)
    #     del predictions_force_t

    #     # Plot best-fit parameters for h=3_1_1, force t constraint
    #     print("Plotting best-fit parameters for h=3_1_1, force t constraint", flush=True)
    #     best_20_pars_df = pd.read_csv("%s/%s_best_fits_pars.csv" % (dir, model_t))
    #     plot_parameters(best_20_pars_df, "best_20_pars_force_t_3_1", figures_dir)

    # # State probabilities
    # figures_dir = "p50_final_figures/"
    # os.makedirs(figures_dir, exist_ok=True)
    # model = "p50_force_t"
    # training_data = pd.read_csv("../data/p50_training_data.csv")
    # conditions = training_data["Stimulus"] + "_" + training_data["Genotype"]
    # conditions = pd.concat([conditions, pd.Series("basal_WT")], ignore_index=True)
    # num_threads = 4
    # results_dir = "parameter_scan_force_t/results/"
    # names_dir = "p50_contrib/results/"

    # cmap = heatmap_cmap

    # state_name_dict = get_renaming_dict(names_dir)

    # print("Making state probabilities plots", flush=True)
    # # Load all state probabilities for the avg. best 20 force t models
    # old_state_names = np.loadtxt("%s/p50_state_names.txt" % (names_dir), dtype=str, delimiter="\0")
    # # initialize df with state names as index
    # probabilities_df = pd.DataFrame(index=state_name_dict.values())
    # for condition in conditions:
    #     state_probs_df = pd.read_csv("%s/%s_state_probabilities_optimized_%s.csv" % (results_dir, model, condition), names=old_state_names)
    #     state_probs_df.rename(columns=state_name_dict, inplace=True)
    #     state_probs_df = state_probs_df.mean()
    #     probabilities_df = pd.concat([probabilities_df, state_probs_df.rename(condition)], axis=1)

    # probabilities_df["state"] = probabilities_df.index
    # state_probs_df = probabilities_df.melt(var_name="Condition", value_name="Probability", id_vars="state")
    # state_probs_df["Stimulus"] = state_probs_df["Condition"].str.split("_", expand=True)[0]
    # state_probs_df["Genotype"] = state_probs_df["Condition"].str.split("_", expand=True)[1]
    # state_probs_df["Genotype"] = state_probs_df["Genotype"].replace("relacrelKO", r"NFκBko")
    # state_probs_df["Genotype"] = state_probs_df["Genotype"].replace("p50KO", "p50ko")
    # state_probs_df["Genotype"] = state_probs_df["Genotype"].replace("irf3irf7KO", "IRF3/7ko")
    # state_probs_df["Genotype"] = state_probs_df["Genotype"].replace("irf3irf5irf7KO", "IRF3/5/7ko")
    # state_probs_df["Stimulus"] = state_probs_df["Stimulus"].replace("polyIC", "PolyIC")
    # state_probs_df["Stimulus"] = state_probs_df["Stimulus"].replace("basal", "Basal")
    # stimuli_levels = ["Basal", "CpG", "LPS", "PolyIC"]
    # # stimuli_levels = ["PolyIC", "LPS", "CpG", "Basal"]
    # genotypes_levels = ["WT","p50ko", "IRF3/7ko", "IRF3/5/7ko", r"NFκBko"]
    # state_probs_df["Condition_old_name"] = state_probs_df["Condition"]
    # state_probs_df["Condition"] = state_probs_df["Stimulus"] + " " + state_probs_df["Genotype"]

    # condition_renaming_dict = {old: new for old, new in zip(state_probs_df["Condition_old_name"].unique(), state_probs_df["Condition"].unique())}

    # state_probs_df["Stimulus"] = pd.Categorical(state_probs_df["Stimulus"], categories=stimuli_levels, ordered=True)
    # state_probs_df["Genotype"] = pd.Categorical(state_probs_df["Genotype"], categories=genotypes_levels, ordered=True)
    # state_probs_df = state_probs_df.sort_values(["Stimulus", "Genotype"])
    # conditions = state_probs_df["Condition"].unique()
    # state_probs_df["Condition"] = pd.Categorical(state_probs_df["Condition"], categories=conditions, ordered=True)
    # # replace " state" in all state names
    # state_probs_df["state"] = state_probs_df["state"].str.replace(" state", "")
    # # print(state_probs_df)

    # # Plot state probabilities
    # with sns.plotting_context("paper", rc=plot_rc_pars):
    #     # make separate plots for IRF/NFkB KOs and p50 KOs
    #     irf_kb_data = state_probs_df.loc[state_probs_df["Genotype"].isin(["WT","IRF3/7ko", r"NFκBko"]) & state_probs_df["Stimulus"].isin(["PolyIC", "LPS"])].copy()
    #     p50_data = state_probs_df.loc[state_probs_df["Genotype"].isin(["WT", "p50ko"])].copy()

    #     # Make dictionary of colors
    #     # genotype_colors = sns.cubehelix_palette(n_colors=len(genotypes_levels), start=-0.2, rot=0.65, dark=0.2, light=0.8, reverse=True)
    #     genotype_colors = sns.color_palette(states_cmap_pars, n_colors=len(genotypes_levels))
    #     genotype_colors = {genotype: color for genotype, color in zip(genotypes_levels, genotype_colors)}

    #     # First plot
    #     irf_kb_data["Genotype"] = pd.Categorical(irf_kb_data["Genotype"], categories=["WT", "IRF3/7ko", r"NFκBko"], ordered=True)
    #     irf_kb_data["Stimulus"] = pd.Categorical(irf_kb_data["Stimulus"], categories=["PolyIC", "LPS"], ordered=True)
    #     p = sns.catplot(data=irf_kb_data, x="state", y="Probability", row="Stimulus", col="Genotype", hue="Genotype", dodge=False, 
    #                     kind="bar", alpha=0.8, palette=genotype_colors, height=5, aspect=0.9, legend=False)
    #     p.set_titles("{row_name} {col_name}")
    #     labels = state_probs_df["state"].unique()
    #     for ax in p.axes.flat:
    #         ax.set_xticklabels(labels, rotation=90)
    #     sns.despine()
    #     plt.tight_layout()
    #     plt.savefig("%s/%s_state_probabilities_barplot_irf_kb.png" % (figures_dir, model), bbox_inches="tight")
    #     plt.close()

    #     # Second plot
    #     p50_data["Genotype"] = pd.Categorical(p50_data["Genotype"], categories=["WT", "p50ko"], ordered=True)
    #     p50_data["Stimulus"] = pd.Categorical(p50_data["Stimulus"], categories=["LPS", "CpG"], ordered=True)
    #     p = sns.catplot(data=p50_data, x="state", y="Probability", row="Stimulus", col="Genotype", hue="Genotype", dodge=False,
    #                     kind="bar", alpha=0.8, palette=genotype_colors, height=5, aspect=0.9, legend=False)
    #     p.set_titles("{row_name} {col_name}")
    #     labels = state_probs_df["state"].unique()
    #     for ax in p.axes.flat:
    #         ax.set_xticklabels(labels, rotation=90)
    #     sns.despine()
    #     plt.tight_layout()
    #     plt.savefig("%s/%s_state_probabilities_barplot_p50.png" % (figures_dir, model), bbox_inches="tight")
    #     plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c","--contributions", action="store_true")
    parser.add_argument("-p","--param_scan", action="store_true")
    parser.add_argument("-s","--state_probs", action="store_true")
    parser.add_argument("-x","--supplemental", action="store_true")
    args = parser.parse_args()

    t = time.time()
    if args.contributions:
        make_contribution_plots()

    if args.param_scan:
        make_param_scan_plots()

    if args.state_probs:
        make_state_probabilities_plots()

    if args.supplemental:
        make_supplemental_plots()

    print("Finished making all plots, took %.2f seconds" % (time.time() - t))

if __name__ == "__main__":
    main()
