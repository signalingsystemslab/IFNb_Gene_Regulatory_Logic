# Make nice version of the plots for the three site model
from p50_model_distal_synergy import get_f
from make_p50_model_plots import plot_predictions_one_plot, make_predictions_data_frame, fix_ax_labels, combine_parameters_data_frame, make_ki_plot, plot_parameters_one_plot
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
data_color = "#6F5987"

states_cmap_pars = "ch:s=0.9,r=-0.8,h=0.6,l=0.9,d=0.2"
models_cmap_pars = "ch:s=-0.0,r=0.6,h=1,d=0.3,l=0.8,g=1_r"

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

def make_parameters_data_frame(pars):
    df_pars = pars.drop(columns=["h1", "h2", "h3", "rmsd"], errors="ignore")

    df_pars["par_set"] = np.arange(len(df_pars))
    df_pars["t_0"] = 0
    df_pars["t_I1I2N"] = 1
    df_pars = df_pars.melt(var_name="Parameter", value_name="Value", id_vars="par_set")
    # df_t_pars = df_pars[df_pars["Parameter"].str.startswith("t")]
    df_t_pars = df_pars.loc[df_pars["Parameter"].str.startswith("t")].copy()
    num_t_pars = len(df_t_pars["Parameter"].unique())
    new_t_par_names = [r"$t_{I}$", r"$t_{I}$", r"$t_N$", r"$t_{I_1I_2}$", r"$t_{I_1N}$", r"$t_{I_2N}$"]
    # Rename t parameters
    df_t_pars["Parameter"] = df_t_pars["Parameter"].replace(["t1", "t2", "t3", "t4", "t5", "t6"], new_t_par_names)
    df_t_pars["Parameter"] = df_t_pars["Parameter"].replace(["t_1", "t_2", "t_3", "t_4", "t_5", "t_6"], new_t_par_names)
    df_t_pars["Parameter"] = df_t_pars["Parameter"].replace(["t_0", "t_I1I2N"], [r"$t_0$", r"$t_{I_1I_2N}$"])
    new_t_par_order = [r"$t_0$",r"$t_{I}$", r"$t_N$", r"$t_{I_1I_2}$", r"$t_{I_1N}$", r"$t_{I_2N}$", r"$t_{I_1I_2N}$"]
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

    df_all_t_pars = df_all_t_pars.loc[df_all_t_pars[r"H_{I_1}"]  == "1"] # acceptable models only


    df_all_k_pars[r"H_{I_2}"] = np.concatenate([np.repeat("1", len(df_k_pars_1_1)), np.repeat("1", len(df_k_pars_1_3)),
                                        np.repeat("3", len(df_k_pars_3_1)), np.repeat("3", len(df_k_pars_3_3))])
    df_all_k_pars[r"H_{I_1}"] = np.concatenate([np.repeat("1", len(df_k_pars_1_1)), np.repeat("3", len(df_k_pars_1_3)),
                                        np.repeat("1", len(df_k_pars_3_1)), np.repeat("3", len(df_k_pars_3_3))])
    df_all_k_pars["Model"] = r"$h_{I_1}$=" + df_all_k_pars[r"H_{I_1}"] + r", $h_{I_2}$=" + df_all_k_pars[r"H_{I_2}"]

    df_all_k_pars = df_all_k_pars.loc[df_all_k_pars[r"H_{I_1}"]  == "1"] # acceptable models only

    colors = sns.color_palette(models_cmap_pars, n_colors=4)

    k_parameters = [r"$k_{I_1}$",r"$K_N$",r"$K_P$"]

    all_parameters = df_all_k_pars["Parameter"].unique()

    if (r"$C$" in all_parameters) or ("C" in all_parameters):
        has_c=True
    else:
        has_c=False

    with sns.plotting_context("paper",rc=plot_rc_pars):
        width = 2.8
        height = 1
        if has_c:
            fig, ax = plt.subplots(1,3, figsize=(width+1, height), 
                               gridspec_kw={"width_ratios":[num_t_pars, 2.5, 1]})
        else:
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

        if has_c:
            df2 = df_all_k_pars[(df_all_k_pars["Parameter"].isin(["C",r"$C$"]))]
            df2 = df2.copy()
            df2["Parameter"] = df2["Parameter"].cat.remove_unused_categories()
            s = sns.stripplot(data=df2, x="Parameter", y="Value", hue = "Model", palette=colors, ax=ax[2], zorder = 0, linewidth=0, 
                          alpha=0.2, jitter=0, dodge=True, legend=False)
            ax[2].set_yscale("log")
            ax[2].set_ylabel(r"Value")
            ax[2].set_xlabel("")
            ax[2].set_ylim(ax[1].get_ylim())
        
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

        for i in range(len(leg.legend_handles)):
            leg.legend_handles[i].set_alpha(1)
            leg.legend_handles[i].set_color(colors[i])

        plt.savefig("%s/%s.png" % (figures_dir, name), bbox_inches="tight")
        plt.close()

        make_ki_plot(df_all_k_pars, name + "_k_i", figures_dir)

def make_param_scan_plots():
    figures_dir = "parameter_scan_dist_syn/nice_figures/"
    os.makedirs(figures_dir, exist_ok=True)
    training_data = pd.read_csv("../data/p50_training_data.csv")
    beta = training_data["IFNb"]
    conditions = training_data["Stimulus"] + "_" + training_data["Genotype"]
    scan_dir = "parameter_scan_dist_syn"
    model_t = "p50_dist_syn"

    # Plot predictions on one plot
    print("Plotting predictions for all hill combinations on one plot", flush=True)
    predictions_1_1 = np.loadtxt("%s/%s_best_fits_ifnb_predicted.csv" % ("%s/results_h_1_1_1/" % scan_dir, model_t), delimiter=",")
    predictions_1_3 = np.loadtxt("%s/%s_best_fits_ifnb_predicted.csv" % ("%s/results_h_1_3_1/" % scan_dir, model_t), delimiter=",")
    predictions_3_1 = np.loadtxt("%s/%s_best_fits_ifnb_predicted.csv" % ("%s/results/" % scan_dir, model_t), delimiter=",")
    predictions_3_3 = np.loadtxt("%s/%s_best_fits_ifnb_predicted.csv" % ("%s/results_h_3_3_1/" % scan_dir, model_t), delimiter=",")
    plot_predictions_one_plot(predictions_1_1, predictions_1_3, predictions_3_1, predictions_3_3, beta, conditions, "best_20_ifnb", figures_dir)
    del predictions_1_1, predictions_1_3, predictions_3_1, predictions_3_3        

    # Plot parameters on one plot
    print("Plotting best-fit parameters for all hill combinations on one plot", flush=True)
    best_20_pars_df_1_1 = pd.read_csv("%s/%s_best_fits_pars.csv" % ("%s/results_h_1_1_1/" % scan_dir, model_t))
    best_20_pars_df_1_3 = pd.read_csv("%s/%s_best_fits_pars.csv" % ("%s/results_h_1_3_1/" % scan_dir, model_t))
    best_20_pars_df_3_1 = pd.read_csv("%s/%s_best_fits_pars.csv" % ("%s/results/" % scan_dir, model_t))
    best_20_pars_df_3_3 = pd.read_csv("%s/%s_best_fits_pars.csv" % ("%s/results_h_3_3_1/" % scan_dir, model_t))
    plot_parameters_one_plot(best_20_pars_df_1_1, best_20_pars_df_1_3, best_20_pars_df_3_1, best_20_pars_df_3_3, "best_20_pars_all", figures_dir)
    del best_20_pars_df_1_1, best_20_pars_df_1_3, best_20_pars_df_3_1, best_20_pars_df_3_3

    # Add RMSD density plot of two models: h1=1,h2=1 and h1=1,h2=3
    print("Plotting RMSD density plot for two hill combinations", flush=True)
    rmsd_1_1 = pd.read_csv("%s/%s_rmsd_optimized.csv" % ("%s/results_h_1_1_1/" % scan_dir, model_t))
    rmsd_3_1 = pd.read_csv("%s/%s_rmsd_optimized.csv" % ("%s/results/" % scan_dir, model_t))
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

def main():
    make_param_scan_plots()

if __name__ == "__main__":
    main()