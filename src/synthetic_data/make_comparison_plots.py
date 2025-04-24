# Make plots comparing results for different models

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors

mpl.rcParams["figure.dpi"] = 600
mpl.rcParams["font.sans-serif"] = "Arial"
data_color = "#6F5987"

states_cmap_pars = "ch:s=0.9,r=-0.8,h=0.6,l=0.9,d=0.2"
models_colors=["#83CCD2","#A7CDA8","#D6CE7E","#E69F63"]

# Background white version of color palettes:
heatmap_cmap = sns.blend_palette(["#17131C", "#3F324D","#997BBA", "#B38FD9","#D2A8FF","#F8F4F9"][::-1],as_cmap=True)
cmap_probs = sns.blend_palette(["white", "#77A5A4","#5A8A8A","#182828"], as_cmap=True)
cmap_t = sns.blend_palette(["white", "black"], as_cmap=True)

plot_rc_pars = {"axes.labelsize":7, "font.size":6, "legend.fontsize":6, "xtick.labelsize":6, 
                                          "ytick.labelsize":6, "axes.titlesize":7, "legend.title_fontsize":7,
                                          "lines.markersize": 3, "axes.linewidth": 0.5,
                                            "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.minor.width": 0.5,
                                            "ytick.minor.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2,
                                            "xtick.minor.size": 1, "ytick.minor.size": 1, "legend.labelspacing": 0.2,
                                            "legend.columnspacing": 0.5, "legend.handletextpad": 0.5, "legend.handlelength": 1.5}
rc_pars={"xtick.major.pad": 1, "ytick.major.pad": 1, "legend.labelspacing": 0.2}
mpl.rcParams.update(rc_pars)

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

def make_predictions_data_frame(ifnb_predicted, beta, conditions):
    # Melt simulated values
    df_ifnb_predicted = ifnb_predicted.melt(var_name="Data point", value_name=r"IFN$\beta$", id_vars=["Dataset"])
    df_ifnb_predicted["beta_type"] = "simulated"

    # Add data values
    df_ifnb_predicted_data = pd.DataFrame({"Data point":conditions, r"IFN$\beta$":beta, "beta_type":"Data", "Dataset":"original"})
    df_ifnb_predicted = pd.concat([df_ifnb_predicted, df_ifnb_predicted_data], ignore_index=True)
    
    # Extract stimulus and genotype
    df_ifnb_predicted["Stimulus"] = df_ifnb_predicted["Data point"].str.split("_", expand=True)[0]
    df_ifnb_predicted["Stimulus"] = df_ifnb_predicted["Stimulus"].replace("polyIC", "PolyIC")
    df_ifnb_predicted["Genotype"] = df_ifnb_predicted["Data point"].str.split("_", expand=True)[1]

    # Sort into category (each plot will be a different category)
    df_ifnb_predicted["Category"] = "Stimulus specific"
    df_ifnb_predicted.loc[df_ifnb_predicted["Genotype"].str.contains("rela"), "Category"] = "NFκB dependence"
    df_ifnb_predicted.loc[df_ifnb_predicted["Genotype"].str.contains("irf"), "Category"] = "IRF dependence"
    df_ifnb_predicted.loc[df_ifnb_predicted["Genotype"].str.contains("p50"), "Category"] = "p50 dependence"

    # Give nicer names
    df_ifnb_predicted["Genotype"] = df_ifnb_predicted["Genotype"].replace("relacrelKO", r"NFκBko")
    df_ifnb_predicted["Genotype"] = df_ifnb_predicted["Genotype"].replace("irf3irf7KO", "IRF3/7ko")
    df_ifnb_predicted["Genotype"] = df_ifnb_predicted["Genotype"].replace("irf3irf5irf7KO", "IRF3/5/7ko")
    df_ifnb_predicted["Genotype"] = df_ifnb_predicted["Genotype"].replace("p50KO", "p50ko")
    df_ifnb_predicted["Data point"] = df_ifnb_predicted["Stimulus"] + " " + df_ifnb_predicted["Genotype"]    

    # Specify order
    stimuli_levels = ["basal", "CpG", "LPS", "PolyIC"]
    genotypes_levels = ["WT","p50ko", "IRF3/7ko", "IRF3/5/7ko", r"NFκBko"]
    df_ifnb_predicted["Stimulus"] = pd.Categorical(df_ifnb_predicted["Stimulus"], categories=stimuli_levels, ordered=True)
    df_ifnb_predicted["Genotype"] = pd.Categorical(df_ifnb_predicted["Genotype"], categories=genotypes_levels, ordered=True)
    df_ifnb_predicted = df_ifnb_predicted.sort_values(["Stimulus", "Genotype"])
    df_ifnb_predicted["Data point"] = pd.Categorical(df_ifnb_predicted["Data point"], categories=df_ifnb_predicted["Data point"].unique(), ordered=True)
    return df_ifnb_predicted

def plot_predictions(df_all, name, figures_dir):
    os.makedirs(figures_dir, exist_ok=True)
    # Plot separately
    for category in df_all["Category"].unique():
        print("Plotting %s for %s" % (category, name), flush=True)
        df = df_all[df_all["Category"]==category].copy()
        # print(df)
        df["Data point"] = df["Data point"].cat.remove_unused_categories()

        exp_only_df = df[df["beta_type"] == "Data"].copy()
        sim_only_df = df[~(df["beta_type"] == "Data")].copy()

        with sns.plotting_context("paper", rc=plot_rc_pars):
            num_bars = len(df["Data point"].unique())

            width  = 3.1*num_bars/3/2.1
            height = 0.8
            fig, ax = plt.subplots(figsize=(width, height))
            cols = [data_color] + models_colors
            # Barplot of mean ifnb vs data point
            sns.barplot(data=df, x="Data point", y=r"IFN$\beta$", hue="Hill", 
                        palette=cols, ax=ax, width=0.8, errorbar=None, legend=False, saturation=0.9, 
                        linewidth=0.5, edgecolor="black")
            # Plot individual data points
            sns.stripplot(data=sim_only_df, x="Data point", y=r"IFN$\beta$", hue="Hill", alpha=0.5, 
                          ax=ax, size=0.5, jitter=True, dodge=True, palette="dark:black", legend=False)
            # # Plot real data points
            # sns.stripplot(data=exp_only_df, x="Data point", y=r"IFN$\beta$", hue="Hill", alpha=1,
            #                 ax=ax, size=2, jitter=False, dodge=True, palette="dark:black", legend=False)
            ax.set_xlabel("")
            ax.set_ylabel(r"IFNβ $f$")
            # ax.set_title(category)
            sns.despine()
            ax, _ = fix_ax_labels(ax)
            plt.tight_layout(pad=0)
            plt.ylim(0,1.05)
            category_nospace = category.replace(" ", "-")
            plt.savefig("%s/%s_%s.png" % (figures_dir, name, category_nospace), bbox_inches="tight")
            plt.close()

    # Make one plot with legend
    with sns.plotting_context("paper", rc=plot_rc_pars):
        category = "NFκB dependence"
        # print(df)
        num_bars = len(df["Data point"].unique())
        
        width  = 3.1*num_bars/3/2.1 + 0.5
        height = 0.8
        fig, ax = plt.subplots(figsize=(width, height))
        cols = [data_color] + models_colors
        sns.barplot(data=df, x="Data point", y=r"IFN$\beta$", hue="Hill", 
                    palette=cols, ax=ax, width=0.8, errorbar=None, saturation=0.9, linewidth=0.5, edgecolor="black")
        ax.set_xlabel("")
        ax.set_ylabel(r"IFN$\beta$")
        # ax.set_title(category)
        sns.despine()
        ax, _ = fix_ax_labels(ax)
        plt.tight_layout(pad=0)
        plt.ylim(0,1.05)
        sns.move_legend(ax, bbox_to_anchor=(1,1), title=None, frameon=False, loc="upper left", ncol=1)
        plt.savefig("%s/%s_legend.png" % (figures_dir, name), bbox_inches="tight")
        plt.close()

def make_predictions_comparisons_plots(error, model_list, beta, conditions, name="all_IRF_Hills"):
    df_ifnb = pd.DataFrame()

    for model in model_list:
        results_dir = "parameter_scan_dist_syn/no_restrict/%s/results_%.1f_combined/" % (model, error)
        df_ifnb_predicted = pd.read_csv("%s/best_optimized_predictions_combined.csv" % results_dir)
        df_ifnb_predicted = make_predictions_data_frame(df_ifnb_predicted, beta, conditions)
        df_ifnb_predicted["Hill"] = model
        df_ifnb_predicted.loc[df_ifnb_predicted["beta_type"] == "Data", "Hill"] = "Data"
        df_ifnb_predicted[r"$h_{I_2}$"] = df_ifnb_predicted["Hill"].str.split("_", expand=True)[1]
        df_ifnb_predicted[r"$h_{I_1}$"] = df_ifnb_predicted["Hill"].str.split("_", expand=True)[2]
        df_ifnb_predicted[r"$h_N$"] = df_ifnb_predicted["Hill"].str.split("_", expand=True)[3]
        n_models = df_ifnb_predicted[r"$h_N$"].unique()
        print(n_models, flush=True)
        if any(m not in ["1", None] for m in n_models):
            df_ifnb_predicted.loc[~(df_ifnb_predicted["beta_type"] == "Data"),"Hill"] = r"$h_{I_1}$=" + df_ifnb_predicted[r"$h_{I_1}$"] + r", $h_{I_2}$=" + df_ifnb_predicted[r"$h_{I_2}$"] + r", $h_N$=" + df_ifnb_predicted[r"$h_N$"]
        else:
            df_ifnb_predicted.loc[~(df_ifnb_predicted["beta_type"] == "Data"),"Hill"] = r"$h_{I_1}$=" + df_ifnb_predicted[r"$h_{I_1}$"] + r", $h_{I_2}$=" + df_ifnb_predicted[r"$h_{I_2}$"]
        df_ifnb = pd.concat([df_ifnb, df_ifnb_predicted], ignore_index=True)

    # remove any duplicate rows
    df_ifnb = df_ifnb.loc[~df_ifnb.duplicated(subset=["Data point", r"IFN$\beta$"], keep="first")]
    
    # Make models have correct order
    df_ifnb = df_ifnb.sort_values(["$h_{I_1}$", "$h_{I_2}$", "$h_N$", "Stimulus", "Genotype"])
    model_order = df_ifnb["Hill"].unique()
    model_order = np.insert(model_order[:-1], 0, model_order[-1])
    df_ifnb["Hill"] = pd.Categorical(df_ifnb["Hill"], categories=model_order, ordered=True)

    plot_predictions(df_ifnb, "%s_%.1f" % (name,error), "parameter_scan_dist_syn/no_restrict/comparison_figures/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--error", type=float, default=40, help="Percent error to use for the model")

    training_data = pd.read_csv("../data/p50_training_data.csv")
    beta = training_data["IFNb"]
    conditions = training_data["Stimulus"] + "_" + training_data["Genotype"]
    err = parser.parse_args().error

    model_list = ["h_1_1_1", "h_3_1_1", "h_1_3_1", "h_3_3_1"]

    make_predictions_comparisons_plots(err, model_list, beta, conditions, name="all_IRF_Hills")

    model_list = ["h_1_1_3", "h_3_1_3", "h_1_3_3", "h_3_3_3"]
    make_predictions_comparisons_plots(err, model_list, beta, conditions, name="all_NFkB_3_IRF_Hills")



if __name__ == "__main__":
    main()