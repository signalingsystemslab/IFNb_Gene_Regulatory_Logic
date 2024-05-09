# Make nice version of the plots for the three site model
from three_site_model import *
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

def make_contribution_plots():
    figures_dir = "three_site_final_figures"
    os.makedirs(figures_dir, exist_ok=True)
    results_dir = "three_site_contrib/results"
    num_t_pars = 5
    num_k_pars = 3
    num_h_pars = 2
    best_fit_dir ="three_site_param_scan_hill/results/"
    model = "three_site_only_hill"
    num_threads = 40
    h="3_3_1"
    # best_20_pars_df = pd.read_csv("%s/%s_all_best_20_pars_h_%s.csv" % (best_fit_dir, model, h))
    state_names_old_names = np.loadtxt("%s/%s_state_names.txt" % (results_dir, model), dtype=str, delimiter="\0")
    cmap = sns.cubehelix_palette(as_cmap=True, light=0.95, dark=0, reverse=True, rot=0.5)
    
    state_names = ["Unbound state",
                    r"$IRF_2$ state", 
                    r"$IRF_1$ state",
                    r"$NF\kappa B$ state",
                    r"$IRF_1\cdot IRF_2$ state",
                    r"$IRF_2\cdot NF\kappa B$ state",
                    r"$IRF_1\cdot NF\kappa B$ state",
                    r"$IRF_1\cdot IRF_2\cdot NF\kappa B$ state"]
    state_names_ordered = ["Unbound state",
                    r"$IRF_1$ state",
                    r"$IRF_2$ state",
                    r"$NF\kappa B$ state",
                    r"$IRF_1\cdot IRF_2$ state",
                    r"$IRF_1\cdot NF\kappa B$ state",
                    r"$IRF_2\cdot NF\kappa B$ state",
                    r"$IRF_1\cdot IRF_2\cdot NF\kappa B$ state"]
    # print(state_names)

    t = time.time()
    print("Making contribution plots, starting at %s" % time.ctime(), flush=True)

    ## Make heatmaps for all states ##
    contrib_df = pd.read_csv("%s/%s_best_params_contributions_sweep.csv" % (results_dir, model))
    # Rename IRFs in state names
    column_names = contrib_df.columns
    column_names = state_names + list(column_names[len(state_names):])
    contrib_df.columns = column_names


    contrib_df = pd.melt(contrib_df, id_vars=[r"NF$\kappa$B", "IRF", "par_set"], value_vars=state_names, var_name="state", value_name="contribution")
    contrib_df["state"] = pd.Categorical(contrib_df["state"], categories=state_names_ordered, ordered=True)
    contrib_df = contrib_df.groupby([r"NF$\kappa$B", "IRF", "state"])["contribution"].mean().reset_index()

    p = sns.FacetGrid(contrib_df, col="state", col_wrap=4, sharex=False, sharey=False)
    cbar_ax = p.figure.add_axes([.92, .3, .02, .4])
    p.map_dataframe(helper_contrib_heatmap, r"NF$\kappa$B", "IRF", "contribution", data=contrib_df, cbar_ax=cbar_ax, vmin=0, vmax=1, cmap=cmap)
    # p.set_axis_labels(r"$IRF$", r"$NF\kappa B$")
    p.set_titles("{col_name}")
    plt.subplots_adjust(top=0.93, right=0.9)
    plt.savefig("%s/%s_contrib_sweep_heatmap.png" % (figures_dir, model))
    plt.close()

    ## Make stacked bar plots for LPS/pIC states ##
    with sns.plotting_context("talk", rc={"lines.markersize": 7}):
        contrib_df = pd.read_csv("%s/%s_specific_conds_contributions.csv" % (results_dir, model))
        column_names = contrib_df.columns
        column_names = state_names + list(column_names[len(state_names):])
        contrib_df.columns = column_names
        contrib_df = pd.melt(contrib_df, id_vars=["stimulus", "genotype", "par_set"], value_vars=state_names, var_name="state", value_name="contribution")
        contrib_df["state"] = pd.Categorical(contrib_df["state"], categories=state_names_ordered, ordered=True)
        contrib_df = contrib_df.groupby(["stimulus", "genotype", "state"])["contribution"].mean().reset_index()
        contrib_df["stimulus"] = contrib_df["stimulus"].replace("polyIC", "PolyIC")
        contrib_df["genotype"] = contrib_df["genotype"].replace("relacrelKO", r"$rela^{-/-}crel^{-/-}$")
        contrib_df["Condition"] = contrib_df["stimulus"] + " " + contrib_df["genotype"]

        # Contributing states
        contrib_states = [r"$IRF_1\cdot IRF_2$ state", r"$IRF_1\cdot NF\kappa B$ state", r"$IRF_1\cdot IRF_2\cdot NF\kappa B$ state"]
        # Sum all non-contributing states into "Other"
        contrib_df["state"] = contrib_df["state"].apply(lambda x: x if x in contrib_states else "Other")
        contrib_df["state"] = pd.Categorical(contrib_df["state"], categories=contrib_states + ["Other"], ordered=True)
        contrib_df = contrib_df.groupby(["Condition", "state"])["contribution"].sum().reset_index()

        fig, ax = plt.subplots(figsize=(10, 5))
        ax = sns.histplot(data=contrib_df, x="Condition", hue="state", weights="contribution", multiple="stack", shrink=0.8, palette=sns.color_palette("ch:s=-.2,r=.6", n_colors=len(contrib_states) + 1))
        ax.set_ylabel("Contribution")
        labels = [item.get_text().replace(" ", "\n") for item in ax.get_xticklabels()]
        ax.set_xticklabels(labels)
        # plt.xticks(rotation=90)
        sns.despine()
        sns.move_legend(ax, bbox_to_anchor=(1, 0.5), title=None, frameon=False, loc="center left")
        plt.tight_layout()
        plt.savefig("%s/%s_specific_conds_contributions.png" % (figures_dir, model))
        plt.close()

    print("Finished making contribution plots, took %.2f seconds" % (time.time() - t))

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

def plot_rmsd_boxplot(all_opt_rmsd, model, figures_dir):
    all_opt_rmsd["Hill"] = pd.Categorical(all_opt_rmsd["Hill"], ordered=True)
    all_opt_rmsd = all_opt_rmsd.sort_values("Hill")
    
    rmsd_df = all_opt_rmsd.copy()

    fig, ax = plt.subplots(figsize=(8,6))
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
    table_data = rmsd_df[[r"$h_{I_1}$", r"$h_{I_2}$", r"$h_N$"]].drop_duplicates().values.tolist()
    # print(table_data)
    table_data = np.array(table_data).T
    table = plt.table(cellText=table_data, cellLoc='center', loc='bottom', rowLabels=[r"$h_{I_1}$", r"$h_{I_2}$", r"$h_N$"], bbox=[0, -0.25, 1, 0.2])

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
    figures_dir = "three_site_final_figures"
    os.makedirs(figures_dir, exist_ok=True)
    num_t_pars = 5
    num_k_pars = 3
    num_h_pars = 2
    # results_dir ="three_site_param_scan_hill/results/seed_0/"
    # results_dir = "parameter_scan/"
    optimization_dir = "optimization/results/seed_0/"

    model = "three_site_force_t"
    training_data = pd.read_csv("../data/training_data.csv")
    beta = training_data["IFNb"]
    conditions = training_data["Stimulus"] + "_" + training_data["Genotype"]
    h_values = np.meshgrid([1,3], [1,3], [1,3])
    h_values = np.array(h_values).T.reshape(-1,3)

    force_t_dir = "parameter_scan_force_t/"

    # # change context, but make dot size smaller
    # sns.set_context("talk", rc={"lines.markersize": 7})

    with sns.plotting_context("talk", rc={"lines.markersize": 7}):
        # Best fit model wth h=1_1_1 (best 20, seed 0)
        print("Plotting predictions for best fit model with h=1_1_1", flush=True)
        dir_111 = "%s/results_h_1_1_1/" % force_t_dir
        predictions_1_1_1 = np.loadtxt("%s/%s_best_fits_ifnb_predicted.csv" % (dir_111, model), delimiter=",")
        plot_predictions(predictions_1_1_1, beta, conditions, "optimized_ifnb_h_1_1_1_force_t", figures_dir, lines=True)
        del predictions_1_1_1

        # RMSD distribution for all hill combinations
        print("Plotting RMSD distributions for all hill combinations", flush=True)
        all_best_rmsd = pd.DataFrame(columns=["rmsd", r"$h_{I_1}$", r"$h_{I_2}$"])
        for row in h_values:
            h_vals_str = "_".join([str(i) for i in row])
            if h_vals_str == "3_1_1":
                dir = "%s/results/" % force_t_dir
            else:
                dir = "%s/results_h_%s/" % (force_t_dir, h_vals_str)

             # Check that the file exists
            if not os.path.exists("%s/%s_rmsd.csv" % (dir, model)):
                print("File %s/%s_rmsd.csv does not exist" % (dir, model))
                continue

            rmsd_df = pd.read_csv("%s/%s_rmsd.csv" % (dir, model))
            h1, h2, hn = row
            rmsd_df[r"$h_{I_1}$"] = h2.astype(str)
            rmsd_df[r"$h_{I_2}$"] = h1.astype(str)
            rmsd_df[r"$h_N$"] = hn.astype(str)
            all_best_rmsd = pd.concat([all_best_rmsd, rmsd_df], ignore_index=True)
            del rmsd_df

        cmap = sns.color_palette("rocket", n_colors=3)
        p= sns.displot(data=all_best_rmsd, x="rmsd", col=r"$h_{I_1}$", row=r"$h_{I_2}$", hue=r"$h_N$", kind="kde", fill=True, alpha=0.5, palette=cmap)
        sns.despine()
        plt.xlabel("RMSD")
        sns.move_legend(p, bbox_to_anchor=(1, 0.5), frameon=False, loc="center left")
        plt.tight_layout()
        plt.savefig("%s/%s_rmsd_distributions_top_100.png" % (figures_dir, model), bbox_inches="tight")
        plt.close()

        # Plot box plot of rmsd
        all_best_rmsd["Hill"] = all_best_rmsd[r"$h_{I_1}$"].astype(str) + "_" + all_best_rmsd[r"$h_{I_2}$"].astype(str) + "_" + all_best_rmsd[r"$h_N$"].astype(str)
        
        
        plot_rmsd_boxplot(all_best_rmsd, model, figures_dir)

        del all_best_rmsd

        # Best fit model with h=3_3_1
        print("Plotting predictions for best fit model with h=3_3_1", flush=True)
        dir_331 = "%s/results_h_3_3_1/" % force_t_dir
        predictions_3_3_1 = np.loadtxt("%s/%s_best_fits_ifnb_predicted.csv" % (dir_331, model), delimiter=",")
        plot_predictions(predictions_3_3_1, beta, conditions, "best_20_ifnb_h_3_3_1_force_t", figures_dir, lines=True)
        del predictions_3_3_1

        # Plot best-fit parameters for h=3_3_1
        print("Plotting best-fit parameters for h=3_3_1", flush=True)
        best_20_pars_df = pd.read_csv("%s/%s_best_fits_pars.csv" % (dir_331, model))
        plot_parameters(best_20_pars_df, "best_20_pars_h_3_3_1_force_t", figures_dir)
        del best_20_pars_df

        # Best fit model with h = 3,1,1
        print("Plotting predictions for best fit model with h=3_1_1", flush=True)
        dir_311 = "%s/results/" % force_t_dir
        predictions_3_1_1 = np.loadtxt("%s/%s_best_fits_ifnb_predicted.csv" % (dir_311, model), delimiter=",")
        plot_predictions(predictions_3_1_1, beta, conditions, "best_20_ifnb_h_3_1_1_force_t", figures_dir, lines=True)
        del predictions_3_1_1

        # Plot best-fit parameters for h=3_1_1
        print("Plotting best-fit parameters for h=3_1_1", flush=True)
        best_20_pars_df = pd.read_csv("%s/%s_best_fits_pars.csv" % (dir_311, model))
        plot_parameters(best_20_pars_df, "best_20_pars_h_3_1_1_force_t", figures_dir)

        print("Finished making param scan plots")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c","--contributions", action="store_true")
    parser.add_argument("-p","--param_scan", action="store_true")
    args = parser.parse_args()

    t = time.time()
    if args.contributions:
        make_contribution_plots()

    if args.param_scan:
        make_param_scan_plots()

    print("Finished making all plots, took %.2f seconds" % (time.time() - t))

if __name__ == "__main__":
    main()
