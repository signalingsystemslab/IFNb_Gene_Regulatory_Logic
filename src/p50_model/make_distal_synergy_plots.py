# Make nice version of the plots for the three site model
from p50_model_distal_synergy import get_f, get_contribution
from make_p50_model_plots import make_predictions_data_frame, get_renaming_dict, make_cbars, fix_ax_labels
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
# models_cmap_pars = "ch:s=-0.0,r=0.6,h=1,d=0.3,l=0.8,g=1_r"
# models_cmap_pars = "ch:s=0.1,r=0.7,h=1,d=0.3,l=0.8,g=1_r"
models_colors=["#83CCD2","#A7CDA8","#D6CE7E","#E69F63"]

# Background white version of color palettes:
heatmap_cmap = sns.blend_palette(["#17131C", "#3F324D","#997BBA", "#B38FD9","#D2A8FF","#F8F4F9"][::-1],as_cmap=True)
cmap_probs = sns.blend_palette(["white", "#77A5A4","#5A8A8A","#182828"], as_cmap=True)
cmap_t = sns.blend_palette(["white", "black"], as_cmap=True)

# # Background black version of color palettes:
# heatmap_cmap = sns.blend_palette(["#17131C","#997BBA","#D2A8FF","#E7D4FC","#F4EEFA"],as_cmap=True)
# cmap_probs = sns.cubehelix_palette(as_cmap=True, light=0.8, dark=0, reverse=True, rot=0.3,start=2, hue=0.6)
# cmap_t = sns.cubehelix_palette(as_cmap=True, light=0.8, dark=0, reverse=True, rot=0.25,start=0.9, hue=0.6)

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

    #     self.t[1] = self.parsT[0] # IRF - t1
    #     self.t[2] = self.parsT[0] # IRF_G - t1
    #     self.t[3] = self.parsT[0] # IRF + p50 - t1
    #     self.t[4] = self.parsT[1] # NFkB - t3
    #     self.t[5] = self.parsT[1] # NFkB + p50 - t3
    #     # 6 is zero
    #     self.t[7] = self.parsT[2] # IRF + IRF_G - t4
    #     self.t[8] = self.parsT[4] # IRF + NFkB - t6
    #     self.t[9] = self.parsT[3] # IRF_G + NFkB - t5
    #     self.t[10] = self.parsT[4] # IRF + NFkB + p50 - t6
    #     self.t[11] = 1

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

def plot_predictions(ifnb_predicted_0, ifnb_predicted_1, ifnb_predicted_2, ifnb_predicted_3, beta, conditions, name, figures_dir, hi2, hi1, hn = []):
    df_ifnb_predicted_0 = make_predictions_data_frame(ifnb_predicted_0, beta, conditions)
    df_ifnb_predicted_1 = make_predictions_data_frame(ifnb_predicted_1, beta, conditions)
    df_ifnb_predicted_2 = make_predictions_data_frame(ifnb_predicted_2, beta, conditions)
    df_ifnb_predicted_3 = make_predictions_data_frame(ifnb_predicted_3, beta, conditions)

    data_df = df_ifnb_predicted_0.loc[df_ifnb_predicted_0["par_set"] == "Data"].copy()

    df_sym = pd.concat([df_ifnb_predicted_0,
                        df_ifnb_predicted_2,
                        df_ifnb_predicted_1,
                        df_ifnb_predicted_3], ignore_index=True) 
    
    hi1 = [str(h) for h in hi1]
    hi2 = [str(h) for h in hi2]
    
    # hi1[0] -> df_ifnb_predicted_0, hi1[1] -> df_ifnb_predicted_1, hi1[2] -> df_ifnb_predicted_2, hi1[3] -> df_ifnb_predicted_3
    # current order: 0, 2, 1, 3
    df_sym[r"H_{I_1}"] = np.concatenate([np.repeat(hi1[0], len(df_ifnb_predicted_0)), np.repeat(hi1[2], len(df_ifnb_predicted_2)),
                                        np.repeat(hi1[1], len(df_ifnb_predicted_1)), np.repeat(hi1[3], len(df_ifnb_predicted_3))])
    df_sym[r"H_{I_2}"] = np.concatenate([np.repeat(hi2[0], len(df_ifnb_predicted_0)), np.repeat(hi2[2], len(df_ifnb_predicted_2)),
                                        np.repeat(hi2[1], len(df_ifnb_predicted_1)), np.repeat(hi2[3], len(df_ifnb_predicted_3))])
    if len(hn) > 0:
        hn = [str(h) for h in hn]
        df_sym[r"H_N"] = np.concatenate([np.repeat(hn[0], len(df_ifnb_predicted_0)), np.repeat(hn[2], len(df_ifnb_predicted_2)),
                                        np.repeat(hn[1], len(df_ifnb_predicted_1)), np.repeat(hn[3], len(df_ifnb_predicted_3))])
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
            cols = [data_color] + models_colors
            sns.barplot(data=df_all[df_all["Category"]==category], x="Data point", y=r"IFN$\beta$", hue="Hill", 
                        palette=cols, ax=ax, width=0.8, errorbar=None, legend=False, saturation=.9, 
                        linewidth=0.5, edgecolor="black", err_kws={'linewidth': 0.75, "color":"black"})
            sns.stripplot(data=df_all[(df_all["Category"]==category)&(~(df_all["par_set"] == "Data"))], x="Data point", y=r"IFN$\beta$", 
                          hue="Hill", alpha=0.5, ax=ax, size=1.5, jitter=True, dodge=True, palette="dark:black", legend=False)
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
        cols = [data_color] + models_colors
        sns.barplot(data=df_all[df_all["Category"]==category], x="Data point", y=r"IFN$\beta$", hue="Hill", 
                    palette=cols, ax=ax, width=0.8, errorbar=None, saturation=0.9, linewidth=0.5, edgecolor="black")
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

def make_ki_plot(df_ki_pars, name, figures_dir, colors = models_colors):
    # Filter for parameter containing "I"
    df_ki_pars = df_ki_pars.loc[df_ki_pars["Parameter"].str.contains("I_2")]

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

    # log-log scale
    with sns.plotting_context("paper", rc=plot_rc_pars):
        fig, ax = plt.subplots(figsize=(2,1.25))
        sns.lineplot(data=df_ki_pars, x="IRF", y=r"$K_I$", hue="Model", palette=colors, ax=ax, zorder = 0,  errorbar=None, estimator=None, alpha=0.2, units="par_set")
        # sns.scatterplot(data=df_ki_pars,x="IRF", y=r"$K_I$", hue=r"$h_I$", palette=colors, ax=ax, legend=False, zorder = 1, linewidth=0, alpha=0.2)
        ax.set_xlabel(r"$[IRF]$ (MNU)")
        ax.set_ylabel(r"$k_{I_2} [IRF]^{h_{I_2}-1}$ (MNU$^{-1}$)")
        ax.set_yscale("log")
        ax.set_xscale("log")
        sns.despine()
        sns.move_legend(ax, loc='center left', bbox_to_anchor=(1, 0.5), frameon=False,
                        columnspacing=1, handletextpad=0.5, handlelength=1.5)
        plt.tight_layout()

        # Change alpha of legend
        leg = ax.get_legend()
        for line in leg.get_lines():
            line.set_alpha(1)

        plt.savefig("%s/%s_log_log.png" % (figures_dir, name), bbox_inches="tight")
        plt.close()


def plot_parameters_one_plot(pars_1_1, pars_1_3, pars_3_1, pars_3_3, name, figures_dir, acceptable_models_only=True):
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

    if acceptable_models_only:
        df_all_t_pars = df_all_t_pars.loc[(df_all_t_pars[r"H_{I_1}"]  == "1") & (df_all_t_pars[r"H_{I_2}"]  == "3")]
        df_all_k_pars = df_all_k_pars.loc[(df_all_k_pars[r"H_{I_1}"]  == "1") & (df_all_k_pars[r"H_{I_2}"]  == "3")]
        colors = models_colors[1:]
    else:
        colors = models_colors

    k_parameters = [r"$k_{I_1}$",r"$K_N$",r"$K_P$"]

    all_parameters = df_all_k_pars["Parameter"].unique()

    if (r"$C$" in all_parameters) or ("C" in all_parameters):
        has_c=True
    else:
        has_c=False

    with sns.plotting_context("paper",rc=plot_rc_pars):
        width = 3
        if acceptable_models_only == False:
            width += 1
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

        make_ki_plot(df_all_k_pars, name + "_k_i", figures_dir, colors=colors)

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
    plot_predictions(predictions_1_1, predictions_1_3, predictions_3_1, predictions_3_3, beta, conditions, "best_20_ifnb", figures_dir,
                     hi2=[1,1,3,3], hi1=[1,3,1,3])
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
        p = sns.kdeplot(data=rmsd, x="RMSD", hue="Hill", fill=True, common_norm=False, palette=models_colors, ax=ax)
        ax.set_xlabel("RMSD")
        ax.set_ylabel("Density")
        sns.despine()
        sns.move_legend(ax, bbox_to_anchor=(0,1), title=None, frameon=False, loc="upper left", ncol=1)
        plt.tight_layout()
        plt.savefig("%s/rmsd_good_models_density_plot.png" % figures_dir)


    print("Finished making param scan plots")

def get_contribution_data(num_t_pars=5, num_k_pars=4, results_dir="parameter_scan_dist_syn/p50_contrib/", num_threads=40):
    h_pars = "3_1_1"
    best_fit_dir ="parameter_scan_dist_syn/results/"
    model = "p50_dist_syn"
    num_threads = 40

    specific_conds_file = "%s/%s_specific_conds_contributions.csv" % (results_dir, model)
    
    # check date of file
    if os.path.exists(specific_conds_file):
        file_time = os.path.getmtime(specific_conds_file)
        if (time.time() - file_time) < 60*60*24:
            print("File %s was modified %.2f hours ago. Not recalculating contributions" % (specific_conds_file, (time.time() - file_time)/3600))
            return 1

    # col_names = ["t%d" % i for i in range(1, num_t_pars+1)] + ["k%d" % i for i in range(1, num_k_pars+1)]
    best_20_pars_df = pd.read_csv("%s/%s_best_fits_pars.csv" % (best_fit_dir, model))
    best_20_pars_df["h1"] = int(h_pars.split("_")[0])
    best_20_pars_df["h2"] = int(h_pars.split("_")[1])
    best_20_pars_df["hn"] = int(h_pars.split("_")[2])
    best_tpars, best_kpars = best_20_pars_df.iloc[:, :num_t_pars].values, best_20_pars_df.iloc[:, num_t_pars:num_t_pars+num_k_pars].values
    best_hpars = best_20_pars_df.loc[:, ["h1", "h2"]].values
    # print("Best t-pars: ", best_tpars)
    # print("Best k-pars: ", best_kpars)
    # print("Best h-pars: ", best_hpars)
    # return 1

    # Calculate relative contributions of each state for values of N and I
    print("Calculating contributions", flush=True)
    N_vals = np.linspace(0, 1, 101)
    I_vals = np.linspace(0, 1, 101)
    N, I = np.meshgrid(N_vals, I_vals)
    N = N.flatten()
    I = I.flatten()
    P = [0,1]
    for p50 in P:
        inputs = [(tpars, kpars, n, i, p50, None, hpars) for tpars, kpars, hpars in zip(best_tpars, best_kpars, best_hpars) for n, i in zip(N, I)]
        par_set = np.repeat(range(len(best_tpars)), len(N_vals) * len(I_vals))

        start = time.time()
        with Pool(num_threads) as p:
            results = p.starmap(get_contribution, [i for i in inputs])
        f_contributions = np.array([r[0] for r in results])
        state_names = results[0][1]
        state_names = ["%s state" % s.replace("none", "Unbound") for s in state_names]
        np.savetxt("%s/p50_state_names.txt" % (results_dir), state_names, fmt="%s")

        contrib_df = pd.DataFrame(f_contributions, columns=state_names)
        contrib_df[r"NF$\kappa$B"] = [inputs[i][2] for i in range(len(inputs))]
        contrib_df["IRF"] = [inputs[i][3] for i in range(len(inputs))]
        contrib_df["par_set"] = par_set
        contrib_df.to_csv("%s/%s_best_params_contributions_sweep_p%d.csv" % (results_dir, model, p50), index=False)
        end = time.time()
        print("Took %.2f seconds to calculate contributions" % (end - start), flush=True)

        # save N and I values
        np.savetxt("%s/%s_N_vals.txt" % (results_dir, model), N_vals, fmt="%.2f")
        np.savetxt("%s/%s_I_vals.txt" % (results_dir, model), I_vals, fmt="%.2f")

        # return 1
        # Plots
        start = time.time()
        contrib_df = pd.read_csv("%s/%s_best_params_contributions_sweep_p%d.csv" % (results_dir, model, p50))

        # if p50 == 1:
        #     print("Total contribution from IRF2+NFkB+p50= %.3f" % np.sum(contrib_df["$IRF\cdot NF\kappa B\cdot p50$ state"]))
        #     print("Total contribution from IRF1+IRF2= %.3f" % np.sum(contrib_df["$IRF\cdot IRF_G$ state"]))
        #     print("Total contribution from IRF1+NFkB= %.3f" % np.sum(contrib_df["$IRF_G\cdot NF\kappa B$ state"]))
        #     print("Total contribution from IRF2+NFkB= %.3f" % np.sum(contrib_df["$IRF\cdot NF\kappa B$ state"]))
        #     print("Total contribution from full bound= %.3f" % np.sum(contrib_df["$IRF\cdot IRF_G\cdot NF\kappa B$ state"]))

        # pivot so that each state value goes into a column called "contribution" and the name of the state goes into a column called "state"
        contrib_df = pd.melt(contrib_df, id_vars=[r"NF$\kappa$B", "IRF", "par_set"], value_vars=state_names, var_name="state", value_name="contribution")
        # Make state a categorical variable so that the order of the states is preserved in the plots
        contrib_df["state"] = pd.Categorical(contrib_df["state"], categories=state_names, ordered=True)

    # print("\nTotal contribution from IRF2+NFkB+p50= %.3f" % np.sum(contrib_df.loc[contrib_df["state"] == "$IRF\cdot NF\kappa B\cdot p50$ state","contribution"]))
    # print("Total contribution from IRF1+IRF2= %.3f" % np.sum(contrib_df.loc[contrib_df["state"] == "$IRF\cdot IRF_G$ state","contribution"]))
    # print("Total contribution from IRF1+NFkB= %.3f" % np.sum(contrib_df.loc[contrib_df["state"] == "$IRF_G\cdot NF\kappa B$ state","contribution"]))
    # print("Total contribution from full bound=  %.3f" % np.sum(contrib_df.loc[contrib_df["state"] == "$IRF\cdot NF\kappa B$ state","contribution"]))
    # print("Total contribution from full bound=  %.3f" % np.sum(contrib_df.loc[contrib_df["state"] == "$IRF\cdot IRF_G\cdot NF\kappa B$ state","contribution"]))

    # Calculate contribution at LPS and polyIC WT and nfkb KO values
    print("Calculating contributions for specific conditions", flush=True)
    training_data = pd.read_csv("../data/p50_training_data.csv")
    stims = ["LPS", "polyIC", "CpG"]
    gen_vals = ["WT", "relacrelKO", "p50KO"]
    filtered_training_data = training_data[(training_data["Stimulus"].isin(stims)) & (training_data["Genotype"].isin(gen_vals))]
    
    inputs = [(tpars, kpars, n, i, p50, None, hpars) for tpars, kpars, hpars in zip(best_tpars, best_kpars, best_hpars) for n, i, p50 in zip(filtered_training_data["NFkB"].values, filtered_training_data["IRF"].values, filtered_training_data["p50"].values)]

    with Pool(num_threads) as p:
        results = p.starmap(get_contribution, [i for i in inputs])
    f_contributions = np.array([r[0] for r in results])
    

    contrib_df = pd.DataFrame(f_contributions, columns=state_names)
    contrib_df["stimulus"] = np.tile(filtered_training_data["Stimulus"].values, len(best_tpars))
    contrib_df["genotype"] = np.tile(filtered_training_data["Genotype"].values, len(best_tpars))
    contrib_df[r"NF$\kappa$B"] = [inputs[i][2] for i in range(len(inputs))]
    contrib_df["IRF"] = [inputs[i][3] for i in range(len(inputs))]
    contrib_df["par_set"] = np.repeat(range(len(best_tpars)), len(filtered_training_data))

    # print("Total contribution from IRF2+NFkB+p50= %.3f" % np.sum(contrib_df["$IRF\cdot NF\kappa B\cdot p50$ state"]))
    # print("Total contribution from IRF2+p50= %.3f" % np.sum(contrib_df["$IRF\cdot p50$ state"]))
    # print("Total contribution from IRF1+IRF2= %.3f" % np.sum(contrib_df["$IRF\cdot IRF_G$ state"]))
    # print("Total contribution from IRF1+NFkB= %.3f" % np.sum(contrib_df["$IRF_G\cdot NF\kappa B$ state"]))
    # print("Total contribution from IRF2+NFkB= %.3f" % np.sum(contrib_df["$IRF\cdot NF\kappa B$ state"]))
    # print("Total contribution from full bound= %.3f" % np.sum(contrib_df["$IRF\cdot IRF_G\cdot NF\kappa B$ state"]))

    # save 
    contrib_df.to_csv("%s/%s_specific_conds_contributions.csv" % (results_dir, model), index=False)

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
    ax.set_xticklabels(["%.1f" % i for i in x_ticks_labels], rotation=0)
    ax.set_yticklabels(["%.1f" % i for i in y_ticks_labels])

    if False:
        # set spines
        for spine in ['top', 'right', 'bottom', 'left']:
            ax.spines[spine].set_visible(True) 
            ax.spines[spine].set_linewidth(0.5)
            ax.spines[spine].set_color('black') 

def make_heatmap(contrib_df, cmap, model, name, figures_dir, facet_type="state"):
    # heatmap_pars = {"axes.titlesize":7}
    # new_rc_pars = plot_rc_pars.copy()
    # new_rc_pars.update(heatmap_pars)
    # print(contrib_df)

    with sns.plotting_context("paper", rc=plot_rc_pars):
        num_states = len(contrib_df[facet_type].unique())
        if num_states > 6:
            ncols=4
            p = sns.FacetGrid(contrib_df, col=facet_type, col_wrap=ncols, sharex=True, sharey=True, height=1.1)
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
            ncols = 3 if num_states > 4 else 2
            # p = sns.FacetGrid(contrib_df, col=facet_type, col_wrap=ncols, sharex=True, sharey=True, height=1)
            # # cbar_ax = p.figure.add_axes([.90, .2, .03, .6])
            # p.map_dataframe(helper_contrib_heatmap, r"NF$\kappa$B", "IRF", "contribution", data=contrib_df, vmin=0, vmax=1, 
            #                 cmap=cmap, square=True, norm=mcolors.LogNorm())
            # p.set_titles("{col_name}")
            # # plt.subplots_adjust(top=0.8, hspace=0.5, wspace = 0.05)
            # plt.tight_layout()
            # plt.savefig("%s/%s_%s_heatmap_log.png" % (figures_dir, model, name))

            p = sns.FacetGrid(contrib_df, col=facet_type, col_wrap=ncols, sharex=True, sharey=True, height=1)
            # cbar_ax = p.figure.add_axes([.90, .2, .03, .6])
            p.map_dataframe(helper_contrib_heatmap, r"NF$\kappa$B", "IRF", "contribution", data=contrib_df, cbar=None, vmin=0, vmax=1, 
                            cmap=cmap, square=True)
            p.set_titles("{col_name}")
            p.set_xlabels(r"[NF$\kappa$B]")
            p.set_ylabels("[IRF]")
            # plt.subplots_adjust(top=0.8, hspace=0.5, wspace = 0.05)
            plt.tight_layout()


        # Label color bar
        # cbar_ax.set_title("Max-Normalized\n Transcription")

        

        plt.savefig("%s/%s_%s_heatmap.png" % (figures_dir, model, name))
        plt.close()

def make_contribution_plots():
    figures_dir = "parameter_scan_dist_syn/nice_figures/"
    os.makedirs(figures_dir, exist_ok=True)
    training_data = pd.read_csv("../data/p50_training_data.csv")
    os.makedirs(figures_dir, exist_ok=True)
    results_dir = "parameter_scan_dist_syn/p50_contrib"
    model = "p50_dist_syn"
    # best_20_pars_df = pd.read_csv("%s/%s_all_best_20_pars_h_%s.csv" % (best_fit_dir, model, h))
    cmap = heatmap_cmap

    get_contribution_data()

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

    # print(state_names)
    # print("\nTotal contribution from IRF2+NFkB+p50= %.3f" % np.sum(contrib_df.loc[contrib_df["state"] == "$IRF_2\\cdot NF\\kappa B\\cdot p50$","contribution"]))
    # print("Total contribution from IRF1+IRF2= %.3f" % np.sum(contrib_df.loc[contrib_df["state"] == "$IRF_1\\cdot IRF_2$","contribution"]))
    # print("Total contribution from IRF1+NFkB= %.3f" % np.sum(contrib_df.loc[contrib_df["state"] == "$IRF_1\\cdot NF\\kappa B$","contribution"]))
    # print("Total contribution from IRF2+NFkB=  %.3f" % np.sum(contrib_df.loc[contrib_df["state"] == "$IRF_2\\cdot NF\\kappa B$","contribution"]))
    # print("Total contribution from full bound=  %.3f" % np.sum(contrib_df.loc[contrib_df["state"] == "$IRF_1\\cdot IRF_2\\cdot NF\\kappa B$","contribution"]))

    # print(state_names)
    contrib_df["state_type"] = "Neither"
    contrib_df.loc[contrib_df["state"].str.contains("IRF"), "state_type"] = "IRF only"
    contrib_df.loc[contrib_df["state"].str.contains("NF"), "state_type"] = r"NF$\kappa$B only"
    contrib_df.loc[contrib_df["state"].str.contains("NF") & contrib_df["state"].str.contains("IRF"), "state_type"] = "Both"
    print(contrib_df.iloc[:12])

    contrib_df_state_type = contrib_df.groupby([r"NF$\kappa$B", "IRF", "state_type"])["contribution"].sum().reset_index()
    print(contrib_df_state_type)

    make_heatmap(contrib_df_state_type, cmap, model, "contrib_state_type", figures_dir, facet_type="state_type")

    make_heatmap(contrib_df, cmap, model, "contrib_sweep_WT", figures_dir)

    # Filter for four states: IRF1IRF2, IRF1NFkB, IRF1IRF2NFkB, IRF1NFkBp50
    active_states = [r"$IRF_1& NF\kappa B$", r"$IRF_2& NF\kappa B& p50$", 
                     r"$IRF_2& NF\kappa B$", r"$IRF_1& IRF_2$", r"$IRF_1& IRF_2& NF\kappa B$"]
    contrib_df_three = contrib_df.loc[contrib_df["state"].isin(active_states)].copy()
    contrib_df_three["state"] = contrib_df_three["state"].cat.remove_unused_categories()

    contrib_df_other = contrib_df.loc[~contrib_df["state"].isin(active_states)].copy()
    contrib_df_other = contrib_df_other.groupby([r"NF$\kappa$B", "IRF"])["contribution"].sum().reset_index()
    contrib_df_other["state"] = "Other"
    contrib_df_other["state"] = pd.Categorical(contrib_df_other["state"], categories=["Other"], ordered=True)
    contrib_df_three = pd.concat([contrib_df_three, contrib_df_other], ignore_index=True)
    # print(contrib_df_three)
    make_heatmap(contrib_df_three, cmap, model, "contrib_sweep_WT_four_states", figures_dir)

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
    contrib_states = active_states
    # Sum all non-contributing states into "Other"
    contrib_df["state"] = contrib_df["state"].apply(lambda x: x if x in contrib_states else "Other")
    contrib_df["state"] = pd.Categorical(contrib_df["state"], categories=contrib_states + ["Other"], ordered=True)
    contrib_df = contrib_df.groupby(["Condition", "state"])["contribution"].sum().reset_index()

    # new_rc_pars = plot_rc_pars.copy()
    # stack_rc_pars = {"xtick.labelsize":5}
    # new_rc_pars.update(stack_rc_pars)
    # states_cmap_pars = "ch:s=0.9,r=-1,h=0.6,l=0.9,d=0.15"
    with sns.plotting_context("paper", rc=plot_rc_pars):
        states_colors = sns.color_palette(states_cmap_pars, n_colors=len(contrib_states)) + ["#444D59"]
        fig, ax = plt.subplots(figsize=(2.5,1.7))
        ax = sns.histplot(data=contrib_df, x="Condition", hue="state", weights="contribution", multiple="stack", shrink=0.5,
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

def make_state_heatmaps(df, category, cmap, figures_dir, plt_type):
    with sns.plotting_context("paper", rc=plot_rc_pars):
        # Make heatmaps for probability, f-value, or t and corresponding colorbar
        # Create the heatmap
        fig, ax = plt.subplots(figsize=(2, 2), sharey=True)
        if plt_type == "ifnb":
            figwidth = 2/12
            fig, ax = plt.subplots(figsize=(figwidth, 2), sharey=True)
        sns.heatmap(data=df, cbar=False, ax=ax, vmin=0, vmax=1, square=True, cmap=cmap, linecolor="black", linewidths=0.25, clip_on=False)
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.tick_params(axis="both", which="both", length=0)    
        plt.subplots_adjust(top=0.93, right=0.8, hspace=0.1, wspace=0.2)
        # plt.tight_layout()
        cat = category.replace(" ", "_")
        
        if plt_type == "ifnb":
            ax.set_yticklabels([])
        if plt_type == "prob":
            ax.set_xticklabels([])
        

        plt.savefig("%s/state_probabilities_heatmap_%s_%s.png" % (figures_dir, cat,plt_type), bbox_inches="tight")
        plt.close()

def make_state_probabilities_plots():
    figures_dir = "parameter_scan_dist_syn/nice_figures/"
    os.makedirs(figures_dir, exist_ok=True)
    model = "p50_dist_syn"
    training_data = pd.read_csv("../data/p50_training_data.csv")
    conditions = training_data["Stimulus"] + "_" + training_data["Genotype"]
    conditions = pd.concat([conditions, pd.Series("basal_WT")], ignore_index=True)
    results_dir = "parameter_scan_dist_syn/results/"
    names_dir = "parameter_scan_dist_syn/p50_contrib"
    num_t_pars = 5
    num_k_pars = 4

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
    # t pars row: 0       t_1    t_1   t_1     t_3     t_3   0      t_4       t_5      t_6    t_6 1
    # self.t = np.array([0.0 for i in range(12)])
    #     self.t[1] = self.parsT[0] # IRF - t1
    #     self.t[2] = self.parsT[0] # IRF_G - t1
    #     self.t[3] = self.parsT[0] # IRF + p50 - t1
    #     self.t[4] = self.parsT[1] # NFkB - t3
    #     self.t[5] = self.parsT[1] # NFkB + p50 - t3
    #     # 6 is zero
    #     self.t[7] = self.parsT[2] # IRF + IRF_G - t4
    #     self.t[8] = self.parsT[4] # IRF + NFkB - t6
    #     self.t[9] = self.parsT[3] # IRF_G + NFkB - t5
    #     self.t[10] = self.parsT[4] # IRF + NFkB + p50 - t6
    #     self.t[11] = 1
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
                                        best_fit_parameters.loc["t_6"],
                                        best_fit_parameters.loc["t_6"],
                                        1]})
    t_pars_df = t_pars_df.set_index("state").T
    # t_pars_df["Condition"] = ["Transcription \n" r"capability ($t$)"]
    t_pars_df["Condition"] = [r"$t$"]

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

        # cmap_probs = sns.cubehelix_palette(as_cmap=True, light=0.95, dark=0, reverse=True, rot=0.3,start=2, hue=0.6)
        # cmap_t = sns.cubehelix_palette(as_cmap=True, light=0.95, dark=0, reverse=True, rot=0.25,start=0.9, hue=0.6)

        # IFNb column
        ifnb_df = pd.read_csv("%s/%s_best_fits_ifnb_predicted.csv" % (results_dir, model), header=None, names=training_data["Stimulus"] + "_" + training_data["Genotype"]).mean()
        t_pars = best_fit_parameters.iloc[:num_t_pars].values
        k_pars = best_fit_parameters.iloc[num_t_pars:num_t_pars+num_k_pars].values
        h_pars = [3,1]
        ifnb_basal_wt = get_f(t_pars, k_pars, 0.01, 0.01, 1, h_pars=h_pars)
        ifnb_df.loc["basal_WT"] = ifnb_basal_wt
        ifnb_df = ifnb_df.rename(condition_renaming_dict)
        ifnb_df = ifnb_df.rename("")
        ifnb_df = ifnb_df.loc[state_probs_df_category.index]
        ifnb_df = ifnb_df.to_frame()

        t_pars_df = t_pars_df.reindex(columns=state_probs_df_category.columns)

        make_state_heatmaps(state_probs_df_category, category, cmap_probs, figures_dir, "prob")
        make_state_heatmaps(ifnb_df, category, cmap, figures_dir, "ifnb")
        print(ifnb_df)
        
        
        # raise ValueError()
    make_cbars(t_pars_df, cmap, figures_dir, "ifnb")
    make_cbars(t_pars_df, cmap_probs, figures_dir, "prob")
    make_cbars(t_pars_df, cmap_t, figures_dir, "t")
    make_state_heatmaps(t_pars_df, "all", cmap_t, figures_dir, "t")

def modify_supp_contrib_dfs(df,state_name_dict):
    state_names = list(state_name_dict.values())
    df.rename(columns=state_name_dict, inplace=True)
    df = pd.melt(df, id_vars=[r"NF$\kappa$B", "IRF", "par_set"], value_vars=state_names, var_name="state", value_name="contribution")
    df["state"] = pd.Categorical(df["state"], categories=state_names, ordered=True)
    df = df.groupby([r"NF$\kappa$B", "IRF", "state"])["contribution"].mean().reset_index()
    df = df.groupby([r"NF$\kappa$B", "IRF"])["contribution"].sum().reset_index()
    return df

def get_max_residual(ifnb_predictions, beta, conditions):
    # Returns df with maximum residual for each par set
    df = make_predictions_data_frame(ifnb_predictions, beta, conditions)
    df_data_only = df.loc[df["par_set"] == "Data",["Data point", r"IFN$\beta$"]]
    df_data_only["Data point"] = pd.Categorical(df_data_only["Data point"], ordered=True)
    df_predictions_only = df.loc[~(df["par_set"] == "Data"),["par_set", "Data point", r"IFN$\beta$"]]
    df_combine = pd.merge(df_data_only, df_predictions_only, on="Data point", suffixes=(" data", " predictions"))
    df_combine["abs_residual"] = np.abs(df_combine[r"IFN$\beta$ predictions"] - df_combine[r"IFN$\beta$ data"])
    df_max = df_combine.loc[df_combine.groupby("par_set")["abs_residual"].idxmax()]
    # columns: Data point, IFN$\beta$ data, par_set, IFN$\beta$ predictions, abs_residual
    return df_max

def plot_max_resid(df, figures_dir, name=""):
    data_point_list= "../max_resid_data_points.csv"
    if os.path.exists(data_point_list):
        print("Loading existing data point list")
        data_points = pd.read_csv(data_point_list)
        new_data_points = [dp for dp in df["Data point"].unique() if dp not in data_points["Data point"].values]
        if len(new_data_points) > 0:
            data_points = pd.concat([data_points, pd.DataFrame({"Data point": new_data_points,
                                                               "Color": [None] * len(new_data_points)})], ignore_index=True)
        else:
            print("No new data points found")
    else:
        print("Creating new data point list")
        data_points = df.loc[:,["Data point"]]
        print(data_points)
        data_points = data_points.drop_duplicates()
        data_points.sort_values(by="Data point", inplace=True)

    pal = sns.cubehelix_palette(n_colors=len(data_points), light=0.8, dark=0.2, reverse=True, rot=1.4, start=1, hue=0.6)
    data_points["Color"] = pal
    data_points.to_csv(data_point_list, index=False)

    colors_dict = dict(data_points.values)

    with sns.plotting_context("paper", rc=plot_rc_pars):
        fig, ax = plt.subplots(figsize=(3.8,1.8))
        col = sns.color_palette("rocket", n_colors=2)[1]
        sns.stripplot(data=df, x="model", y="abs_residual", hue="Data point", size=3, palette=colors_dict)

        ax.set_ylabel("Max Absolute Residual")
        plt.xticks(rotation=90)
        # sns.move_legend(ax, bbox_to_anchor=(0.5, 1), title="Worst-fit condition", frameon=False, loc="lower center", ncol=2)
        sns.move_legend(ax, bbox_to_anchor=(1, 0.5), title="Worst-fit condition", frameon=False, loc="center left", ncol=1)

        # Remove x-axis labels
        ax.set_xticklabels([])
        # Remove x-axis title
        ax.set_xlabel("")
        # # Remove x-axis ticks
        # ax.set_xticks([])

        # Create a table of h values
        df["Cooperativity"] = df["Cooperativity"].replace({"None": "-"})
        df["Cooperativity"] = df["Cooperativity"].replace({"NFkB": "N"})
        df["Cooperativity"] = df["Cooperativity"].replace({"IRF": "I"})

        # df["Cooperativity"] = df["Cooperativity"].replace({"NFkB": r"NF$\kappa$B"})

        table_data = df[[r"$h_{I_1}$", r"$h_{I_2}$", r"$h_N$", "Cooperativity"]].drop_duplicates().values.tolist()

        # print(table_data)
        table_data = np.array(table_data).T
        table = plt.table(cellText=table_data, 
                          cellLoc='center', 
                          loc='bottom', 
                          rowLabels=[r"$h_{I_1}$", r"$h_{I_2}$", r"$h_N$", "Coop."], 
                          bbox=[0, -0.7, 1, 0.6])

        colors = sns.color_palette("rocket", n_colors=4)
        alpha = 0.5
        colors = [(color[0], color[1], color[2], alpha) for color in colors]

        for key, cell in table.get_celld().items():
            cell.set_linewidth(0.5)

        # # Loop through the cells and change their color based on their text
        # for i in range(len(table_data)):
        #     for j in range(len(table_data[i])):
        #         cell = table[i, j]
        #         if table_data[i][j] in [5,"5","N"]:
        #             cell.set_facecolor(colors[0])
        #         elif table_data[i][j] in [3,"3",""]:
        #             cell.set_facecolor(colors[1])
        #         elif table_data[i][j] in [1,"1"]:
        #             cell.set_facecolor(colors[2])
        #         elif table_data[i][j] == "I":
        #             cell.set_facecolor(colors[3])

        # Adjust layout to make room for the table:
        plt.subplots_adjust(left=0.2, bottom=0.6)
        sns.despine()
        plt.xticks(rotation=90)
        plt.tight_layout()

        plt.savefig("%s/max_resid_%s.png" % (figures_dir, name), bbox_inches="tight")
        plt.close()

def make_supplemental_plots():
    # Param scan
    figures_dir = "parameter_scan_dist_syn/nice_figures/"
    os.makedirs(figures_dir, exist_ok=True)
 
    training_data = pd.read_csv("../data/p50_training_data.csv")
    beta = training_data["IFNb"]
    conditions = training_data["Stimulus"] + "_" + training_data["Genotype"]

    force_t_dir = "parameter_scan_dist_syn/"
    model_t = "p50_dist_syn"

    # Make contribution plots for WT/p50 difference
    contrib_results_dir = "parameter_scan_dist_syn/p50_contrib"
    model = "p50_dist_syn"
    # best_20_pars_df = pd.read_csv("%s/%s_all_best_20_pars_h_%s.csv" % (best_fit_dir, model, h))

    # get_contribution_data()

    t = time.time()
    print("Making supplemental contribution plots, starting at %s" % time.ctime(), flush=True)

    ## Make heatmaps for all states, WT p50 ##
    contrib_df_WT = pd.read_csv("%s/%s_best_params_contributions_sweep_p1.csv" % (contrib_results_dir, model))
    contrib_df_p50ko = pd.read_csv("%s/%s_best_params_contributions_sweep_p0.csv" % (contrib_results_dir, model))

    # Rename the columns in contrib_df
    state_name_dict = get_renaming_dict(contrib_results_dir)
    contrib_df_WT = modify_supp_contrib_dfs(contrib_df_WT, state_name_dict)
    contrib_df_p50ko = modify_supp_contrib_dfs(contrib_df_p50ko, state_name_dict)
       
    # df_combined = contrib_df_WT.join(contrib_df_p50ko, lsuffix="_WT", rsuffix="_p50ko", how="outer", on=["NF$\kappa$B", "IRF"])
    df_combined = contrib_df_WT.merge(contrib_df_p50ko, on=["NF$\kappa$B", "IRF"], suffixes=("_WT", "_p50ko"), how="outer")
    df_combined.rename(columns={"contribution_WT":"WT", "contribution_p50ko":"p50ko"}, inplace=True)
    df_combined["Difference"] = df_combined["p50ko"] - df_combined["WT"]
    # df_combined["contribution_diff"] = df_combined["contribution_p50ko"] - df_combined["contribution_WT"]
    df_combined = df_combined.melt(id_vars=["NF$\kappa$B", "IRF"], value_vars=["WT", "p50ko", "Difference"], var_name="genotype", value_name="contribution")

    # Make heatmap for WT, p50ko, and difference
    with sns.plotting_context("paper", rc=plot_rc_pars):
        ncols = 3 
        p = sns.FacetGrid(df_combined, col="genotype", col_wrap=ncols, sharex=True, sharey=True, height=1.3)
        p.map_dataframe(helper_contrib_heatmap, r"NF$\kappa$B", "IRF", "contribution", data=df_combined, cbar=False, vmin=0, vmax=1,
                        square=True, cmap = heatmap_cmap)
        p.set_titles("{col_name}")
        # plt.subplots_adjust(top=0.8, hspace=0.5, wspace = 0.05)
        plt.tight_layout()
        plt.savefig("%s/transcription_difference_heatmap.png" % (figures_dir))
        plt.close()

    # best fit NFkB cooperativity model with different Hill combinations
    predictions_c_1_1 = np.loadtxt("%s/%s_best_fits_ifnb_predicted.csv" % ("%s/results_h_1_1_1_c_NFkB" % force_t_dir, model_t), delimiter=",")
    predictions_c_1_3 = np.loadtxt("%s/%s_best_fits_ifnb_predicted.csv" % ("%s/results_h_1_3_1_c_NFkB" % force_t_dir, model_t), delimiter=",")
    predictions_c_3_3 = np.loadtxt("%s/%s_best_fits_ifnb_predicted.csv" % ("%s/results_h_3_3_1_c_NFkB" % force_t_dir, model_t), delimiter=",")
    predictions_c_3_1 = np.loadtxt("%s/%s_best_fits_ifnb_predicted.csv" % ("%s/results_c_NFkB" % force_t_dir, model_t), delimiter=",")
    plot_predictions(predictions_c_1_1, predictions_c_1_3, predictions_c_3_1, predictions_c_3_3, beta, conditions, 
                              "best_20_ifnb_c_NFkB", figures_dir, hi2=[1,1,3,3], hi1=[1,3,1,3])
    
    del predictions_c_1_1, predictions_c_1_3, predictions_c_3_1, predictions_c_3_3

    best_20_pars_df_c_1_1 = pd.read_csv("%s/%s_best_fits_pars.csv" % ("%s/results_h_1_1_1_c_NFkB" % force_t_dir, model_t))
    best_20_pars_df_c_1_3 = pd.read_csv("%s/%s_best_fits_pars.csv" % ("%s/results_h_1_3_1_c_NFkB" % force_t_dir, model_t))
    best_20_pars_df_c_3_1 = pd.read_csv("%s/%s_best_fits_pars.csv" % ("%s/results_c_NFkB" % force_t_dir, model_t))
    best_20_pars_df_c_3_3 = pd.read_csv("%s/%s_best_fits_pars.csv" % ("%s/results_h_3_3_1_c_NFkB" % force_t_dir, model_t))
    plot_parameters_one_plot(best_20_pars_df_c_1_1, best_20_pars_df_c_1_3, best_20_pars_df_c_3_1, best_20_pars_df_c_3_3, 
                             "best_20_pars_c_NFkB", figures_dir, False)

    del best_20_pars_df_c_1_1, best_20_pars_df_c_1_3, best_20_pars_df_c_3_1, best_20_pars_df_c_3_3

    # best fit IRF cooperativity model with different Hill combinations
    predictions_c_1_1 = np.loadtxt("%s/%s_best_fits_ifnb_predicted.csv" % ("%s/results_h_1_1_1_c_IRF" % force_t_dir, model_t), delimiter=",")
    predictions_c_1_3 = np.loadtxt("%s/%s_best_fits_ifnb_predicted.csv" % ("%s/results_h_1_3_1_c_IRF" % force_t_dir, model_t), delimiter=",")
    predictions_c_3_3 = np.loadtxt("%s/%s_best_fits_ifnb_predicted.csv" % ("%s/results_h_3_3_1_c_IRF" % force_t_dir, model_t), delimiter=",")
    predictions_c_3_1 = np.loadtxt("%s/%s_best_fits_ifnb_predicted.csv" % ("%s/results_c_IRF" % force_t_dir, model_t), delimiter=",")
    plot_predictions(predictions_c_1_1, predictions_c_1_3, predictions_c_3_1, predictions_c_3_3, beta, conditions, 
                              "best_20_ifnb_c_IRF", figures_dir, hi2=[1,1,3,3], hi1=[1,3,1,3])
    
    del predictions_c_1_1, predictions_c_1_3, predictions_c_3_1, predictions_c_3_3

    best_20_pars_df_c_1_1 = pd.read_csv("%s/%s_best_fits_pars.csv" % ("%s/results_h_1_1_1_c_IRF" % force_t_dir, model_t))
    best_20_pars_df_c_1_3 = pd.read_csv("%s/%s_best_fits_pars.csv" % ("%s/results_h_1_3_1_c_IRF" % force_t_dir, model_t))
    best_20_pars_df_c_3_1 = pd.read_csv("%s/%s_best_fits_pars.csv" % ("%s/results_c_IRF" % force_t_dir, model_t))
    best_20_pars_df_c_3_3 = pd.read_csv("%s/%s_best_fits_pars.csv" % ("%s/results_h_3_3_1_c_IRF" % force_t_dir, model_t))
    plot_parameters_one_plot(best_20_pars_df_c_1_1, best_20_pars_df_c_1_3, best_20_pars_df_c_3_1, best_20_pars_df_c_3_3, 
                             "best_20_pars_c_IRF", figures_dir, False)
    del best_20_pars_df_c_1_1, best_20_pars_df_c_1_3, best_20_pars_df_c_3_1, best_20_pars_df_c_3_3
    
    # No distal synergy model
    no_dist_syn_dir = "parameter_scan_force_t/"
    predictions_1_1 = np.loadtxt("%s/results_h_1_1_1/p50_force_t_best_fits_ifnb_predicted.csv" % no_dist_syn_dir, delimiter=",")
    predictions_1_3 = np.loadtxt("%s/results_h_1_3_1/p50_force_t_best_fits_ifnb_predicted.csv" % no_dist_syn_dir, delimiter=",")
    predictions_3_1 = np.loadtxt("%s/results/p50_force_t_best_fits_ifnb_predicted.csv" % no_dist_syn_dir, delimiter=",")
    predictions_3_3 = np.loadtxt("%s/results_h_3_3_1/p50_force_t_best_fits_ifnb_predicted.csv" % no_dist_syn_dir, delimiter=",")
    plot_predictions(predictions_1_1, predictions_1_3, predictions_3_1, predictions_3_3, beta, conditions, 
                              "best_20_ifnb_no_dist_syn", figures_dir, hi2=[1,1,3,3], hi1=[1,3,1,3])
    del predictions_1_1, predictions_1_3, predictions_3_1, predictions_3_3

    best_20_pars_df_1_1 = pd.read_csv("%s/results_h_1_1_1/p50_force_t_best_fits_pars.csv" % no_dist_syn_dir)
    best_20_pars_df_1_3 = pd.read_csv("%s/results_h_1_3_1/p50_force_t_best_fits_pars.csv" % no_dist_syn_dir)
    best_20_pars_df_3_1 = pd.read_csv("%s/results/p50_force_t_best_fits_pars.csv" % no_dist_syn_dir)
    best_20_pars_df_3_3 = pd.read_csv("%s/results_h_3_3_1/p50_force_t_best_fits_pars.csv" % no_dist_syn_dir)

    # Make t_6 = t_1+t_3 (because t_IRF2&NFkB = t_IRF2 + t_NFkB)
    for df in [best_20_pars_df_1_1, best_20_pars_df_1_3, best_20_pars_df_3_1, best_20_pars_df_3_3]:
        print(df)
        df["t_6"] = df["t_1"] + df["t_3"]
        print(df)

    plot_parameters_one_plot(best_20_pars_df_1_1, best_20_pars_df_1_3, best_20_pars_df_3_1, best_20_pars_df_3_3,
                             "best_20_pars_no_dist_syn", figures_dir, True)
    del best_20_pars_df_1_1, best_20_pars_df_1_3, best_20_pars_df_3_1, best_20_pars_df_3_3

    # Plot max residual for each model
    h1, h2, h3 = np.meshgrid([1,3],[1,3],[1,3])
    models = ["h_%d_%d_%d" % (h1.ravel()[i], h2.ravel()[i], h3.ravel()[i]) for i in range(len(h1.ravel()))]
    c_IRF_models = ["h_%d_%d_%d_c_IRF" % (h1.ravel()[i], h2.ravel()[i], h3.ravel()[i]) for i in range(len(h1.ravel()))]
    c_NFkB_models = ["h_%d_%d_%d_c_NFkB" % (h1.ravel()[i], h2.ravel()[i], h3.ravel()[i]) for i in range(len(h1.ravel()))]
    models += c_IRF_models + c_NFkB_models
    print(models)
    max_residuals_df = pd.DataFrame()
    for m in models:
        if "3_1_1" in m:
            coop = ("_" + "_".join(m.split("_")[4:])) if len(m.split("_")) > 4 else ""
            predictions = np.loadtxt("%s/%s_best_fits_ifnb_predicted.csv" % ("%s/results%s" % (force_t_dir, coop), model_t), delimiter=",")
        else:
            fname = "%s/%s_best_fits_ifnb_predicted.csv" % ("%s/results_%s" % (force_t_dir, m), model_t)
            if not os.path.exists(fname):
                print("File %s does not exist, skipping" % fname)
                continue
            predictions = np.loadtxt(fname, delimiter=",")
        df = get_max_residual(predictions, beta, conditions)
        # print(df)
        df[r"$h_{I_1}$"] = m.split("_")[2]
        df[r"$h_{I_2}$"] = m.split("_")[1]
        df[r"$h_N$"] = m.split("_")[3]
        df["Cooperativity"] = m.split("_")[5] if len(m.split("_")) > 4 else "None"
        df["model"] = m
        # df["model"] = r"$h_{I_1}=$%s, $h_{I_2}=$%s, $h_{N}=$%s" % (df[r"h_{I_1}"].values[0], df[r"h_{I_2}"].values[0], df[r"h_{N}"].values[0])
        max_residuals_df = pd.concat([max_residuals_df, df], ignore_index=True)

    print(max_residuals_df)

    plot_max_resid(max_residuals_df, figures_dir, "all_hill_and_coop_models")



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
    new_par_names = [r"$t_{I}$", r"$t_{I}$", r"$t_N$", r"$t_{I_1I_2}$", r"$t_{I_1N}$", r"$t_{I_2N}$",r"$k_{I_2}$", r"$k_{I_1}$", r"$k_N$", r"$k_P$"]
    old_par_names = ["t_1", "t_2", "t_3", "t_4", "t_5", "t_6", "k_1", "k_2", "k_n", "k_p"]
    old_par_names2 = ["t1", "t2", "t3", "t4", "t5", "t6", "k1", "k2", "kn", "kp"]
    rename_dict = {old:new for old, new in zip(old_par_names, new_par_names)}
    rename_dict2 = {old:new for old, new in zip(old_par_names2, new_par_names)}
    df_pars = df_pars.rename(columns=rename_dict)
    df_pars = df_pars.rename(columns=rename_dict2)
    df_pars = df_pars.drop(columns=["rmsd"])
    column_order = [r"$t_{I}$", r"$t_N$", r"$t_{I_1I_2}$", r"$t_{I_1N}$", r"$t_{I_2N}$", r"$k_{I_1}$", r"$k_{I_2}$", r"$k_N$", r"$k_P$", "Model"]
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
        g = sns.pairplot(df_pars, hue="Model", palette=models_colors, diag_kind="kde", height=2, aspect=1)
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
