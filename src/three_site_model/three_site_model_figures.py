# Make nice version of the plots
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

def make_predictions_data_frame(ifnb_predicted, beta, conditions):
    df_ifnb_predicted = pd.DataFrame(ifnb_predicted, columns=conditions)
    df_ifnb_predicted["par_set"] = np.arange(len(df_ifnb_predicted))
    df_ifnb_predicted = df_ifnb_predicted.melt(var_name="Data point", value_name=r"IFN$\beta$", id_vars="par_set")

    df_ifnb_predicted_data = pd.DataFrame({"Data point":conditions, r"IFN$\beta$":beta, "par_set":"Data"})
    df_ifnb_predicted = pd.concat([df_ifnb_predicted, df_ifnb_predicted_data], ignore_index=True)
    df_ifnb_predicted["Stimulus"] = df_ifnb_predicted["Data point"].str.split("_", expand=True)[0]
    df_ifnb_predicted["Genotype"] = df_ifnb_predicted["Data point"].str.split("_", expand=True)[1]

    # stimuli_levels = ["basal", "CpG", "LPS", "polyIC"]
    # genotypes_levels = ["WT", "irf3irf7KO", "irf3irf5irf7KO", "relacrelKO", "p50KO"]
    # df_ifnb_predicted["Stimulus"] = pd.Categorical(df_ifnb_predicted["Stimulus"], categories=stimuli_levels, ordered=True)
    # df_ifnb_predicted["Genotype"] = pd.Categorical(df_ifnb_predicted["Genotype"], categories=genotypes_levels, ordered=True)
                                                   
    df_ifnb_predicted["Stimulus"] = df_ifnb_predicted["Stimulus"].replace("polyIC", "PolyIC")
    df_ifnb_predicted["Genotype"] = df_ifnb_predicted["Genotype"].replace("relacrelKO", r"$rela^{-/-}crel^{-/-}$")
    df_ifnb_predicted["Genotype"] = df_ifnb_predicted["Genotype"].replace("irf3irf7KO", r"$irf3^{-/-}irf7^{-/-}$")
    df_ifnb_predicted["Genotype"] = df_ifnb_predicted["Genotype"].replace("irf3irf5irf7KO", r"$irf3^{-/-}irf5^{-/-}irf7^{-/-}$")
    df_ifnb_predicted["Genotype"] = df_ifnb_predicted["Genotype"].replace("p50KO", r"$nfkb1^{-/-}$")

    df_ifnb_predicted["Data point"] = df_ifnb_predicted["Stimulus"] + " " + df_ifnb_predicted["Genotype"]   

    stimuli_levels = ["Basal", "CpG", "LPS", "PolyIC"]
    genotypes_levels = [r"WT", r"$irf3^{-/-}irf7^{-/-}$", r"$irf3^{-/-}irf5^{-/-}irf7^{-/-}$", r"$rela^{-/-}crel^{-/-}$", r"$nfkb1^{-/-}$"]
    df_ifnb_predicted["Stimulus"] = pd.Categorical(df_ifnb_predicted["Stimulus"], categories=stimuli_levels, ordered=True)
    df_ifnb_predicted["Genotype"] = pd.Categorical(df_ifnb_predicted["Genotype"], categories=genotypes_levels, ordered=True)
    df_ifnb_predicted = df_ifnb_predicted.sort_values(["Stimulus", "Genotype"])
    return df_ifnb_predicted

def plot_predictions(ifnb_predicted, beta, conditions, name, figures_dir, lines = True):
    with sns.plotting_context("talk", rc={"lines.markersize": 7}):

        df_ifnb_predicted = make_predictions_data_frame(ifnb_predicted, beta, conditions)
        col = sns.color_palette("rocket", n_colors=7)[4]
        col = mcolors.rgb2hex(col) 
        fig, ax = plt.subplots(figsize=(6.5, 6))
        
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
    with sns.plotting_context("talk", rc={"lines.markersize": 7}):
        df_pars = pars.drop(columns=["h1", "h2", "h3", "rmsd"], errors="ignore")

        df_pars["par_set"] = np.arange(len(df_pars))
        df_pars = df_pars.melt(var_name="Parameter", value_name="Value", id_vars="par_set")
        all_param_names = df_pars["Parameter"].unique()

        df_t_pars = df_pars[df_pars["Parameter"].str.startswith("t")]
        num_t_pars = len(df_t_pars["Parameter"].unique())
        
        df_k_pars = df_pars[df_pars["Parameter"].str.startswith("k")]
        num_k_pars = len(df_k_pars["Parameter"].unique())
        # df_k_pars["Parameter"] = df_k_pars["Parameter"].str.replace("k3", r"$k_N$")
        # df_k_pars["Parameter"] = df_k_pars["Parameter"].str.replace("k2", r"$k_2$")
        # df_k_pars["Parameter"] = df_k_pars["Parameter"].str.replace("k1", r"$k_1$")
        # df_k_pars["Parameter"] = df_k_pars["Parameter"].str.replace("kn", r"$k_N$")
        df_k_pars.loc[df_k_pars["Parameter"] == "k1", "Parameter"] = r"$k_1$"
        df_k_pars.loc[df_k_pars["Parameter"] == "k2", "Parameter"] = r"$k_2$"
        df_k_pars.loc[df_k_pars["Parameter"] == "kn", "Parameter"] = r"$k_N$"
        df_k_pars.loc[df_k_pars["Parameter"] == "k3", "Parameter"] = r"$k_N$"
        df_k_pars.loc[df_k_pars["Parameter"] == "k4", "Parameter"] = r"$k_P$"

        if "c" in all_param_names:
            df_c_pars = df_pars[df_pars["Parameter"] == "c"]
            num_c_pars = len(df_c_pars["Parameter"].unique())

            fig, ax = plt.subplots(1,3, figsize=(15,5), gridspec_kw={"width_ratios":[num_t_pars, num_k_pars, num_c_pars]})
            sns.lineplot(data=df_t_pars, x="Parameter", y="Value", units="par_set", estimator=None, legend=False, alpha=0.2, ax=ax[0], color="black")
            sns.scatterplot(data=df_t_pars, x="Parameter", y="Value", color="black", ax=ax[0], legend=False, alpha=0.2, zorder = 10)
            sns.lineplot(data=df_k_pars, x="Parameter", y="Value", units="par_set", estimator=None, ax=ax[1], legend=False, alpha=0.2, color="black")
            sns.scatterplot(data=df_k_pars, x="Parameter", y="Value", color="black", ax=ax[1], legend=False, alpha=0.2, zorder = 10)
            sns.lineplot(data=df_c_pars, x="Parameter", y="Value", units="par_set", estimator=None, ax=ax[2], legend=False, alpha=0.2, color="black")
            sns.scatterplot(data=df_c_pars, x="Parameter", y="Value", color="black", ax=ax[2], legend=False, alpha=0.2, zorder = 10)
            ax[1].set_yscale("log")
            ax[2].set_yscale("log")
            sns.despine()
            plt.tight_layout()
            plt.savefig("%s/%s.png" % (figures_dir, name))
            plt.close()

        else:
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

def plot_parameter_distributions(pars, figures_dir, subset ="All", name="parameter_distributions", param_names=None, num_t_pars=5, num_k_pars=3):
    if param_names is None:
        par_names = ["t%d" % (i+1) for i in range(num_t_pars)] + ["k%d" % (i+1) for i in range(num_k_pars)]
    else:
        par_names = param_names
    df_pars = pd.DataFrame(pars, columns=par_names)
    df_pars["par_set"] = np.arange(len(df_pars))
    
    t_pars = [par for par in par_names if par.startswith("t")]
    k_pars = [par for par in par_names if par.startswith("k")]
    c_pars = [par for par in par_names if par.startswith("c")]

    df_t_pars = df_pars.loc[:,t_pars + ["par_set"]]
    df_k_pars = df_pars.loc[:,k_pars + ["par_set"]]

    colors = ["#bdbdbd", "#fbb4ae"]

    if type(subset) == str:
        if subset == "All":
            df_t_pars = df_t_pars.melt(var_name="Parameter", value_name="Value", id_vars="par_set")

            sns.displot(data=df_t_pars, x="Value", col="Parameter", fill=True, alpha=0.5, color=colors[0], kind="kde", col_wrap=3)
            sns.despine()
            plt.tight_layout()
            plt.savefig("%s/%s_t_pars.png" % (figures_dir, name))
            plt.close()

            df_k_pars = df_k_pars.melt(var_name="Parameter", value_name="Value", id_vars="par_set")
            sns.displot(data=df_k_pars, x="Value", col="Parameter", fill=True, alpha=0.5, color=colors[0], kind="kde", log_scale=(True, False))
            sns.despine()
            plt.tight_layout()
            plt.savefig("%s/%s_k_pars.png" % (figures_dir, name))
            plt.close()

            if len(c_pars) > 0:
                df_c_pars = df_pars.loc[:,c_pars + ["par_set"]]
                df_c_pars = df_c_pars.melt(var_name="Parameter", value_name="Value", id_vars="par_set")
                sns.displot(data=df_c_pars, x="Value", col="Parameter", fill=True, alpha=0.5, color=colors[0], kind="kde", log_scale=(True, False))
                sns.despine()
                plt.tight_layout()
                plt.savefig("%s/%s_c_pars.png" % (figures_dir, name))
                plt.close()
        else:
            raise ValueError("Subset must be a list of indices or 'All'")
    else:
        
        df_t_pars["subset"] = [True if i in subset else False for i in range(len(df_pars))]
        print("There are %d points in the subset" % len(df_t_pars.loc[:, "subset"] == True))
        df_t_pars = df_t_pars.melt(var_name="Parameter", value_name="Value", id_vars=["par_set", "subset"])

        sns.displot(data=df_t_pars, x="Value", hue="subset", col="Parameter", fill=True, alpha=0.5, palette=colors, kind="kde", col_wrap=3, common_norm=False)
        sns.despine()
        plt.tight_layout()
        plt.savefig("%s/%s_t_pars.png" % (figures_dir, name))
        plt.close()

        df_k_pars["subset"] = [True if i in subset else False for i in range(len(df_pars))]
        df_k_pars = df_k_pars.melt(var_name="Parameter", value_name="Value", id_vars=["par_set", "subset"])
        sns.displot(data=df_k_pars, x="Value", hue="subset", col="Parameter", fill=True, alpha=0.5, palette=colors, kind="kde", log_scale=(True, False), common_norm=False)
        sns.despine()
        plt.tight_layout()
        plt.savefig("%s/%s_k_pars.png" % (figures_dir, name))
        plt.close()