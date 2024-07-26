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
from make_p50_model_plots import plot_rc_pars, make_predictions_data_frame, make_parameters_data_frame, fix_ax_labels, models_cmap_pars, data_color
import warnings

def make_predictions_plot(df_all, name, figures_dir):
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
                        palette=cols, ax=ax, width=0.8, errorbar="sd", legend=False, saturation=0.9, err_kws={'linewidth': 1})
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
                    palette=cols, ax=ax, width=0.8, errorbar="sd", saturation=0.9, err_kws={'linewidth': 1})
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

def make_parameters_plot(df_all_t_pars, df_all_k_pars, df_all_h_pars, num_t_pars, num_k_pars, name, figures_dir):
    colors = sns.color_palette(models_cmap_pars, n_colors=4)
    num_h_pars=2

    with sns.plotting_context("paper",rc=plot_rc_pars):
        if df_all_h_pars.empty:
            width = 2.8
            height = 1
            fig, ax = plt.subplots(1,2, figsize=(width, height), 
                                gridspec_kw={"width_ratios":[num_t_pars, num_k_pars]})
        else:
            width = 3.2
            height = 1
            fig, ax = plt.subplots(1,3, figsize=(width, height), 
                                gridspec_kw={"width_ratios":[num_t_pars, num_k_pars, num_h_pars]})
    
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

            if df_all_h_pars.empty == False:
                df_model = df_all_h_pars[df_all_h_pars["Model"] == model]
                sns.lineplot(data=df_model, x="Parameter", y="Value", color=colors[i], ax=ax[2], zorder = i, errorbar=None, estimator=None, alpha=0.2, units="par_set")
                sns.scatterplot(data=df_model, x="Parameter", y="Value", color=colors[i], ax=ax[2], legend=False, zorder = i+0.5, linewidth=0, alpha=0.2)

        ax[1].set_yscale("log")
        ax[1].set_ylabel("")

        ax[0].set_ylabel("Parameter Value")
        # ax[0].set_xlabel("")
        # ax[1].set_xlabel("")
        for a in ax:
            a.set_xlabel("")

        sns.despine()
        plt.tight_layout()
        leg = fig.legend(legend_handles, unique_models, loc="lower center", bbox_to_anchor=(0.5, 1), frameon=False, 
                         ncol=4, columnspacing=1, handletextpad=0.5, handlelength=1.5)

        for i in range(len(unique_models)):
            leg.legend_handles[i].set_alpha(1)
            leg.legend_handles[i].set_color(colors[i])
        plt.savefig("%s/%s.png" % (figures_dir, name), bbox_inches="tight")
        plt.close()

def make_param_scan_plots():
    figures_dir = "p50_different_h_figures/"
    os.makedirs(figures_dir, exist_ok=True)
    training_data = pd.read_csv("../data/p50_training_data.csv")
    beta = training_data["IFNb"]
    conditions = training_data["Stimulus"] + "_" + training_data["Genotype"]
    force_t_dir = "parameter_scan_force_t/"
    model_t = "p50_force_t"

    # Scanning Hill coefficients
    dir = "parameter_scan_h/results/"
    print("Plotting predictions for scanning Hill coefficients", flush=True)
    predictions = np.loadtxt("%s/%s_best_fits_ifnb_predicted.csv" % (dir, model_t), delimiter=",")
    df_ifnb_predicted = make_predictions_data_frame(predictions, beta, conditions)
    df_ifnb_predicted["Hill"] = ["Data" if ps == "Data" else "Optimized" for ps in df_ifnb_predicted["par_set"]]
    # print(df_ifnb_predicted)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        make_predictions_plot(df_ifnb_predicted, "predictions_h_scan", figures_dir)

    print("Plotting best-fit parameters for scanning Hill", flush=True)
    parameters = pd.read_csv("%s/%s_best_fits_pars.csv" % (dir, model_t))
    df_t_pars, df_k_pars, num_t_pars, num_k_pars = make_parameters_data_frame(parameters)

    parameters["par_set"] = np.arange(len(parameters))
    parameters = parameters.melt(var_name="Parameter", value_name="Value", id_vars="par_set")
    df_h_pars = parameters.loc[parameters["Parameter"].str.startswith("h")].copy()
    new_h_par_names = [r"$h_{I_2}$", r"$h_{I_1}$"]
    df_h_pars["Parameter"] = df_h_pars["Parameter"].replace(["h1", "h2"], new_h_par_names)
    df_h_pars["Parameter"] = pd.Categorical(df_h_pars["Parameter"], categories=new_h_par_names[::-1], ordered=True)
    df_h_pars["Model"] = "Optimized"

    df_t_pars["Model"] = "Optimized"
    df_k_pars["Model"] = "Optimized"
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        make_parameters_plot(df_t_pars, df_k_pars, df_h_pars, num_t_pars, num_k_pars, "parameters_h_scan", figures_dir)

    # Plot predictions on one plot
    print("Plotting predictions for all hill combinations on one plot", flush=True)
    predictions_force_t_1_2 = np.loadtxt("%s/%s_best_fits_ifnb_predicted.csv" % ("%s/results_h_1_2_1/" % force_t_dir, model_t), delimiter=",")
    predictions_force_t_2_1 = np.loadtxt("%s/%s_best_fits_ifnb_predicted.csv" % ("%s/results_h_2_1_1/" % force_t_dir, model_t), delimiter=",")
    predictions_force_t_1_3 = np.loadtxt("%s/%s_best_fits_ifnb_predicted.csv" % ("%s/results_h_1_3_1/" % force_t_dir, model_t), delimiter=",")
    predictions_force_t_3_1 = np.loadtxt("%s/%s_best_fits_ifnb_predicted.csv" % ("%s/results/" % force_t_dir, model_t), delimiter=",")
    predictions_force_t_1_2_df = make_predictions_data_frame(predictions_force_t_1_2, beta, conditions)
    predictions_force_t_2_1_df = make_predictions_data_frame(predictions_force_t_2_1, beta, conditions)
    predictions_force_t_1_3_df = make_predictions_data_frame(predictions_force_t_1_3, beta, conditions)
    predictions_force_t_3_1_df = make_predictions_data_frame(predictions_force_t_3_1, beta, conditions)
    predictions_force_t_1_2_df["Hill"] = ["Data" if ps == "Data" else r"$h_{I_1}$=2, $h_{I_2}$=1" for ps in predictions_force_t_1_2_df["par_set"]]
    predictions_force_t_2_1_df["Hill"] = ["Data" if ps == "Data" else r"$h_{I_1}$=1, $h_{I_2}$=2" for ps in predictions_force_t_2_1_df["par_set"]]
    predictions_force_t_1_3_df["Hill"] = ["Data" if ps == "Data" else r"$h_{I_1}$=3, $h_{I_2}$=1" for ps in predictions_force_t_1_3_df["par_set"]]
    predictions_force_t_3_1_df["Hill"] = ["Data" if ps == "Data" else r"$h_{I_1}$=1, $h_{I_2}$=3" for ps in predictions_force_t_3_1_df["par_set"]]
    df_all = pd.concat([predictions_force_t_1_2_df, predictions_force_t_2_1_df, predictions_force_t_1_3_df, predictions_force_t_3_1_df], ignore_index=True)
    df_all["Hill"] = pd.Categorical(df_all["Hill"], categories=["Data", r"$h_{I_1}$=1, $h_{I_2}$=2", r"$h_{I_1}$=1, $h_{I_2}$=3",
                                                                 r"$h_{I_1}$=2, $h_{I_2}$=1", r"$h_{I_1}$=3, $h_{I_2}$=1"], ordered=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        make_predictions_plot(df_all, "predictions_h_2s_all", figures_dir)


    # Plot parameters on one plot
    print("Plotting best-fit parameters for all hill combinations on one plot", flush=True)
    best_20_pars_df_1_2 = pd.read_csv("%s/%s_best_fits_pars.csv" % ("%s/results_h_1_2_1/" % force_t_dir, model_t))
    best_20_pars_df_2_1 = pd.read_csv("%s/%s_best_fits_pars.csv" % ("%s/results_h_2_1_1/" % force_t_dir, model_t))
    df_t_pars_1_2, df_k_pars_1_2, _, _ = make_parameters_data_frame(best_20_pars_df_1_2)
    df_t_pars_2_1, df_k_pars_2_1, num_t_pars, num_k_pars = make_parameters_data_frame(best_20_pars_df_2_1)
    df_t_pars_1_2["Model"] = r"$h_{I_1}$=2, $h_{I_2}$=1"
    df_k_pars_1_2["Model"] = r"$h_{I_1}$=2, $h_{I_2}$=1"
    df_t_pars_2_1["Model"] = r"$h_{I_1}$=1, $h_{I_2}$=2"
    df_k_pars_2_1["Model"] = r"$h_{I_1}$=1, $h_{I_2}$=2"
    df_t_pars = pd.concat([df_t_pars_1_2, df_t_pars_2_1], ignore_index=True)
    df_k_pars = pd.concat([df_k_pars_1_2, df_k_pars_2_1], ignore_index=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        make_parameters_plot(df_t_pars, df_k_pars, pd.DataFrame(), num_t_pars, num_k_pars, "parameters_h_2s_all", figures_dir)

    # best_20_pars_df_1_1 = pd.read_csv("%s/%s_best_fits_pars.csv" % ("%s/results_h_1_1_1/" % force_t_dir, model_t))
    # best_20_pars_df_1_3 = pd.read_csv("%s/%s_best_fits_pars.csv" % ("%s/results_h_1_3_1/" % force_t_dir, model_t))
    # best_20_pars_df_3_1 = pd.read_csv("%s/%s_best_fits_pars.csv" % ("%s/results/" % force_t_dir, model_t))
    # best_20_pars_df_3_3 = pd.read_csv("%s/%s_best_fits_pars.csv" % ("%s/results_h_3_3_1/" % force_t_dir, model_t))
    # plot_parameters_one_plot(best_20_pars_df_1_1, best_20_pars_df_1_3, best_20_pars_df_3_1, best_20_pars_df_3_3, "best_20_pars_force_t_all", figures_dir)
    # del best_20_pars_df_1_1, best_20_pars_df_1_3, best_20_pars_df_3_1, best_20_pars_df_3_3

    # # Add RMSD density plot of two models: h1=1,h2=1 and h1=1,h2=3
    # print("Plotting RMSD density plot for two hill combinations", flush=True)
    # rmsd_1_1 = pd.read_csv("%s/%s_rmsd_optimized.csv" % ("%s/results_h_1_1_1/" % force_t_dir, model_t))
    # rmsd_3_1 = pd.read_csv("%s/%s_rmsd_optimized.csv" % ("%s/results/" % force_t_dir, model_t))
    # rmsd = pd.concat([rmsd_1_1, rmsd_3_1], ignore_index=True)
    # rmsd[r"H_{I_1}"] = np.concatenate([np.repeat("1", len(rmsd_1_1) + len(rmsd_3_1))])
    # rmsd[r"H_{I_2}"] = np.concatenate([np.repeat("1", len(rmsd_1_1)), np.repeat("3", len(rmsd_3_1))])
    # rmsd["Hill"] = r"$h_{I_1}$=" + rmsd[r"H_{I_1}"] + r", $h_{I_2}$=" + rmsd[r"H_{I_2}"]
    # rmsd = rmsd.loc[rmsd["rmsd_type"] == "rmsd_final"]

    # with sns.plotting_context("paper", rc=plot_rc_pars):
    #     fig, ax = plt.subplots(figsize=(1.6,1.1))
    #     p = sns.kdeplot(data=rmsd, x="RMSD", hue="Hill", fill=True, common_norm=False, palette=sns.color_palette(models_cmap_pars, n_colors=4), ax=ax)
    #     ax.set_xlabel("RMSD")
    #     ax.set_ylabel("Density")
    #     sns.despine()
    #     sns.move_legend(ax, bbox_to_anchor=(0,1), title=None, frameon=False, loc="upper left", ncol=1)
    #     plt.tight_layout()
    #     plt.savefig("%s/rmsd_good_models_density_plot.png" % figures_dir)


    print("Finished making param scan plots")


def main():
    t = time.time()
    make_param_scan_plots()
    print("Finished making all plots, took %.2f seconds" % (time.time() - t))

if __name__ == "__main__":
    main()