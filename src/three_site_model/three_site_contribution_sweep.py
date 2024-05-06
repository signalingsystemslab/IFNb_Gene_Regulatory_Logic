from three_site_model_force_t import *
import matplotlib.pyplot as plt
import pandas as pd
import os
# import scipy.optimize as opt
import time
from multiprocessing import Pool
import argparse
import seaborn as sns
import scipy.stats.qmc as qmc

figures_dir = "three_site_contrib/figures"
results_dir = "three_site_contrib/results"
os.makedirs(results_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)
num_t_pars = 4
num_k_pars = 3
num_h_pars = 2

def main():
    h_pars = "3_3_1"
    best_fit_dir ="parameter_scan_force_t/results_h_%s" % h_pars
    model = "three_site"
    num_threads = 40
    # col_names = ["t%d" % i for i in range(1, num_t_pars+1)] + ["k%d" % i for i in range(1, num_k_pars+1)]
    best_20_pars_df = pd.read_csv("%s/%s_force_t_best_fits_pars.csv" % (best_fit_dir, model))
    best_20_pars_df["h1"] = int(h_pars.split("_")[0])
    best_20_pars_df["h2"] = int(h_pars.split("_")[1])
    best_20_pars_df["hn"] = int(h_pars.split("_")[2])
    best_tpars, best_kpars = best_20_pars_df.iloc[:, :num_t_pars].values, best_20_pars_df.iloc[:, num_t_pars:num_t_pars+num_k_pars].values
    best_hpars = best_20_pars_df.loc[:, ["h1", "h2", "hn"]].values
    # print("Best t-pars: ", best_tpars)
    # print("Best k-pars: ", best_kpars)
    # print("Best h-pars: ", best_hpars)
    # return 1

    # Calculate relative contributions of each state for values of N and I
    N_vals = np.linspace(0, 1, 51)
    I_vals = np.linspace(0, 1, 51)
    N, I = np.meshgrid(N_vals, I_vals)
    N = N.flatten()
    I = I.flatten()
    inputs = [(tpars, kpars, n, i, None, hpars) for tpars, kpars, hpars in zip(best_tpars, best_kpars, best_hpars) for n, i in zip(N, I)]
    par_set = np.repeat(range(len(best_tpars)), len(N_vals) * len(I_vals))

    start = time.time()
    with Pool(num_threads) as p:
        results = p.starmap(get_contribution, [i for i in inputs])
    f_contributions = np.array([r[0] for r in results])
    state_names = results[0][1]
    state_names = ["%s state" % s.replace("none", "Unbound") for s in state_names]
    np.savetxt("%s/%s_state_names.txt" % (results_dir, model), state_names, fmt="%s")

    contrib_df = pd.DataFrame(f_contributions, columns=state_names)
    contrib_df[r"NF$\kappa$B"] = [inputs[i][2] for i in range(len(inputs))]
    contrib_df["IRF"] = [inputs[i][3] for i in range(len(inputs))]
    contrib_df["par_set"] = par_set
    contrib_df.to_csv("%s/%s_best_params_contributions_sweep.csv" % (results_dir, model), index=False)
    end = time.time()
    print("Took %.2f seconds to calculate contributions" % (end - start), flush=True)

    # save N and I values
    np.savetxt("%s/%s_N_vals.txt" % (results_dir, model), N_vals, fmt="%.2f")
    np.savetxt("%s/%s_I_vals.txt" % (results_dir, model), I_vals, fmt="%.2f")

    # return 1
    # Plots
    start = time.time()
    contrib_df = pd.read_csv("%s/%s_best_params_contributions_sweep.csv" % (results_dir, model))


    # pivot so that each state value goes into a column called "contribution" and the name of the state goes into a column called "state"
    contrib_df = pd.melt(contrib_df, id_vars=[r"NF$\kappa$B", "IRF", "par_set"], value_vars=state_names, var_name="state", value_name="contribution")
    # Make state a categorical variable so that the order of the states is preserved in the plots
    contrib_df["state"] = pd.Categorical(contrib_df["state"], categories=state_names, ordered=True)

    t = time.time()
    # Make heatmap of relative contributions of IRF state
    vmax = contrib_df["contribution"].max()
    for indiv_state in state_names:
        contrib_df_individual1 = contrib_df[contrib_df["state"] == indiv_state]
        # group by nfkb, IRF and average the contributions
        # return mean, min, and max
        # contrib_df_individual = contrib_df_individual1.groupby([r"NF$\kappa$B", "IRF"])["contribution"].agg(["mean", "min", "max", "std"]).reset_index()
        contrib_df_individual = contrib_df_individual1.groupby([r"NF$\kappa$B", "IRF"])["contribution"].mean().reset_index()
        contrib_df_individual = contrib_df_individual.pivot_table(index="IRF", columns=r"NF$\kappa$B", values="contribution")
        # contrib_df_individual = contrib_df_individual.sort_index(ascending=False)

        state_name_no_tex = indiv_state.replace("$", "").replace("\\", "").replace("cdot", " x")

        plt.subplots()
        ax=sns.heatmap(contrib_df_individual, vmin=0, vmax=vmax)
        ax.invert_yaxis()
        # Only label start, end, and 0.5 of each axis
        n_min, n_max = min(N_vals), max(N_vals)
        pt5 = (n_max - n_min) / 2
        plt.xticks([0, 24, 50], [n_min, pt5, n_max])
        i_min, i_max = min(I_vals), max(I_vals)
        pt5 = (i_max - i_min) / 2
        plt.yticks([0, 24, 50], [i_min, pt5, i_max])
        plt.title("Contribution of %s" % indiv_state)
        plt.tight_layout()
        plt.savefig("%s/%s_contrib_sweep_%s_heatmap.png" % (figures_dir, model, state_name_no_tex))
        plt.close()

    print("Took %.2f seconds to plot heatmaps" % (time.time() - t), flush=True)

    ## NFkB contributions
    t = time.time()
    # Remove rows that aren't min or max nfkb values
    contrib_df_min_max = contrib_df[(contrib_df[r"NF$\kappa$B"] == np.min(N_vals)) | (contrib_df[r"NF$\kappa$B"] == np.max(N_vals))]

    # Plot irf vs contribution for nfkb=min and nfkb=max on same plot
    plt.subplots()
    p = sns.relplot(data=contrib_df_min_max, x="IRF", y="contribution", col="state", hue=r"NF$\kappa$B", kind="line", col_wrap=4)
    sns.despine()
    # sns.move_legend(p, "lower center", bbox_to_anchor=(.5, 1), ncol=2, title=None, frameon=False)
    # plt.tight_layout()
    plt.savefig("%s/%s_contrib_sweep_min_max_NFkB_v_IRF.png" % (figures_dir, model))
    plt.close()

    # Make columns for min and max nfkb contributions for each parameter set, IRF value
    contrib_df_min_max = contrib_df_min_max.pivot_table(index=["par_set", "IRF", "state"], columns=r"NF$\kappa$B", values="contribution").reset_index()
    contrib_df_min_max["contribution_diff"] = contrib_df_min_max[np.max(N_vals)] - contrib_df_min_max[np.min(N_vals)]

    # Plot irf vs contribution difference
    plt.subplots()
    p = sns.relplot(data=contrib_df_min_max, x="IRF", y="contribution_diff", col="state", kind="line", col_wrap=4)
    sns.despine()
    plt.tight_layout()
    plt.savefig("%s/%s_contrib_sweep_min_max_NFkB_v_IRF_diff.png" % (figures_dir, model))
    plt.close()

    print("Took %.2f minutes to plot min max nfkb contributions" % ((time.time() - t)/60), flush=True)

    ## IRF contributions
    t = time.time()
    # Remove rows that aren't min or max irf values
    contrib_df_min_max = contrib_df[(contrib_df["IRF"] == np.min(I_vals)) | (contrib_df["IRF"] == np.max(I_vals))]
    # Plot nfkb vs contribution for irf=min and irf=max on same plot
    plt.subplots()
    p = sns.relplot(data=contrib_df_min_max, x=r"NF$\kappa$B", y="contribution", col="state", hue="IRF", kind="line", col_wrap=4)
    sns.despine()
    # sns.move_legend(p, "lower center", bbox_to_anchor=(.5, 1), ncol=2, title=None, frameon=False)
    # plt.tight_layout()
    plt.savefig("%s/%s_contrib_sweep_min_max_IRF_v_NFkB.png" % (figures_dir, model))
    plt.close()

    # Make columns for min and max irf contributions for each parameter set, nfkb value
    contrib_df_min_max = contrib_df_min_max.pivot_table(index=["par_set", r"NF$\kappa$B", "state"], columns="IRF", values="contribution").reset_index()
    contrib_df_min_max["contribution_diff"] = contrib_df_min_max[np.max(I_vals)] - contrib_df_min_max[np.min(I_vals)]

    # Plot nfkb vs contribution difference
    plt.subplots()
    p = sns.relplot(data=contrib_df_min_max, x=r"NF$\kappa$B", y="contribution_diff", col="state", kind="line", col_wrap=4)
    sns.despine()
    plt.tight_layout()
    plt.savefig("%s/%s_contrib_sweep_min_max_IRF_v_NFkB_diff.png" % (figures_dir, model))
    plt.close()

    print("Took %.2f minutes to plot min max irf contributions" % ((time.time() - t)/60), flush=True)

    ## All states contributions (10 values of N and I)
    vals_to_keep = N_vals[::5]
    contrib_df_select = contrib_df[(contrib_df[r"NF$\kappa$B"].isin(vals_to_keep)) & (contrib_df["IRF"].isin(vals_to_keep))]

    t = time.time()
    # Plot relative contributions of each state for each parameter set vs NFKB and IRF
    plt.subplots()
    p = sns.relplot(data=contrib_df_select, x=r"NF$\kappa$B", y="contribution", col="state", hue="IRF", kind="line", col_wrap=4)
    sns.despine()
    plt.savefig("%s/%s_contrib_sweep_all_states_v_NFkB_10vals.png" % (figures_dir, model))
    plt.close()

    plt.subplots()
    p = sns.relplot(data=contrib_df_select, x="IRF", y="contribution", col="state", hue=r"NF$\kappa$B", kind="line", col_wrap=4)
    sns.despine()
    plt.savefig("%s/%s_contrib_sweep_all_states_v_IRF_10vals.png" % (figures_dir, model))
    plt.close()

    # Plot all values
    plt.subplots()
    p = sns.relplot(data=contrib_df, x=r"NF$\kappa$B", y="contribution", col="state", hue="IRF", kind="line", col_wrap=4, **{"errorbar": None})
    sns.despine()
    plt.savefig("%s/%s_contrib_sweep_all_states_v_NFkB_all_vals.png" % (figures_dir, model))
    plt.close()

    plt.subplots()
    p = sns.relplot(data=contrib_df, x="IRF", y="contribution", col="state", hue=r"NF$\kappa$B", kind="line", col_wrap=4, **{"errorbar": None})
    sns.despine()
    plt.savefig("%s/%s_contrib_sweep_all_states_v_IRF_all_vals.png" % (figures_dir, model))
    plt.close()

    print("Took %.2f minutes to plot all states contributions" % ((time.time() - t)/60), flush=True)

    # # Plot relative contributions of two IRF state, IRF+NFkB states, and IRF+IRF+NFkB states
    # two_IRF_state = r"$IRF\cdot IRF_G$"
    # IRF_NFkB_state1 = r"$IRF\cdot NF\kappa B$"
    # IRF_NFkB_state2 = r"$IRF_G\cdot NF\kappa B$"
    # IRF_IRF_NFkB_state = r"$IRF\cdot IRF_G\cdot NF\kappa B$"

    # contrib_df_select = contrib_df[contrib_df["state"].isin([two_IRF_state, IRF_NFkB_state1, IRF_NFkB_state2, IRF_IRF_NFkB_state])]

    # plt.subplots()
    # p = sns.relplot(data=contrib_df_select, x=r"NF$\kappa$B", y="contribution", col="state", hue="IRF", kind="line", col_wrap=2)
    # sns.despine()
    # plt.tight_layout()
    # plt.legend(bbox_to_anchor=(1.1, 0.5))
    # plt.savefig("%s/%s_contrib_sweep_selected_states_v_NFkB.png" % (figures_dir, model))
    # plt.close()

    # Calculate contribution at LPS and polyIC WT and nfkb KO values
    training_data = pd.read_csv("../data/training_data.csv")
    stims = ["LPS", "polyIC"]
    gen_vals = ["WT", "relacrelKO"]
    filtered_training_data = training_data[(training_data["Stimulus"].isin(stims)) & (training_data["Genotype"].isin(gen_vals))]
    
    inputs = [(tpars, kpars, n, i, None, hpars) for tpars, kpars, hpars in zip(best_tpars, best_kpars, best_hpars) for n, i in zip(filtered_training_data["NFkB"].values, filtered_training_data["IRF"].values)]

    with Pool(num_threads) as p:
        results = p.starmap(get_contribution, [i for i in inputs])
    f_contributions = np.array([r[0] for r in results])
    

    contrib_df = pd.DataFrame(f_contributions, columns=state_names)
    contrib_df["stimulus"] = np.tile(filtered_training_data["Stimulus"].values, len(best_tpars))
    contrib_df["genotype"] = np.tile(filtered_training_data["Genotype"].values, len(best_tpars))
    contrib_df[r"NF$\kappa$B"] = [inputs[i][2] for i in range(len(inputs))]
    contrib_df["IRF"] = [inputs[i][3] for i in range(len(inputs))]
    contrib_df["par_set"] = np.repeat(range(len(best_tpars)), len(filtered_training_data))

    # save 
    contrib_df.to_csv("%s/%s_specific_conds_contributions.csv" % (results_dir, model), index=False)
    
    # Plot stacked bar graph where x-axis is condition (LPS WT, LPS KO, polyIC WT, polyIC KO) and y-axis is contribution, and each bar is a different state
    contrib_df = pd.melt(contrib_df, id_vars=["stimulus", "genotype", "par_set"], value_vars=state_names, var_name="state", value_name="contribution")
    contrib_df["state"] = pd.Categorical(contrib_df["state"], categories=state_names, ordered=True)
    contrib_df = contrib_df.groupby(["stimulus", "genotype", "state"])["contribution"].mean().reset_index()
    contrib_df["Condition"] = contrib_df["stimulus"] + " " + contrib_df["genotype"]
    ax = sns.histplot(data=contrib_df, x="Condition", hue="state", weights="contribution", multiple="stack", shrink=0.8, palette="rocket")
    ax.set_ylabel("Contribution")
    plt.xticks(rotation=45)
    sns.despine()
    sns.move_legend(ax, bbox_to_anchor=(1, 0.5), title=None, frameon=False, loc="center left")
    plt.tight_layout()
    plt.savefig("%s/%s_specific_conds_contributions.png" % (figures_dir, model))

    end = time.time()
    print("Took %.2f minutes to make all plots." % ((end - start)/60), flush=True)

if __name__ == "__main__":
    main()