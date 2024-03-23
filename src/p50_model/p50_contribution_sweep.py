from p50_model import *
import matplotlib.pyplot as plt
import pandas as pd
import os
# import scipy.optimize as opt
import time
from multiprocessing import Pool
import argparse
import seaborn as sns
import scipy.stats.qmc as qmc

figures_dir = "no_p50_contrib/figures"
results_dir = "no_p50_contrib/results"
os.makedirs(results_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)
num_t_pars = 6
num_k_pars = 4
num_h_pars = 2

def main():
    best_fit_dir ="param_scan_hill_no_p50/results/"
    model = "hill_no_p50_model"
    num_threads = 40
    best_20_pars_df = pd.read_csv("%s/hill_three_site_all_best_20_pars.csv" % best_fit_dir)
    best_tpars, best_kpars, best_hpars = best_20_pars_df.iloc[:, :num_t_pars].values, best_20_pars_df.iloc[:, num_t_pars:num_t_pars+num_k_pars].values, best_20_pars_df.iloc[:, num_t_pars+num_k_pars:num_t_pars+num_k_pars+num_h_pars].values

    # Calculate relative contributions of each state for values of N and I
    N_vals = np.linspace(0, 1, 51)
    I_vals = np.linspace(0, 1, 51)
    N, I = np.meshgrid(N_vals, I_vals)
    N = N.flatten()
    I = I.flatten()
    inputs = [(tpars, kpars, n, i, 1, None, hpars) for tpars, kpars, hpars in zip(best_tpars, best_kpars, best_hpars) for n, i in zip(N, I)]
    par_set = np.repeat(range(len(best_tpars)), len(N_vals) * len(I_vals))

    start = time.time()
    with Pool(num_threads) as p:
        results = p.starmap(get_contribution, [i for i in inputs])
    f_contributions = np.array([r[0] for r in results])
    state_names = results[0][1]
    state_names = np.array(state_names)
    contrib_df = pd.DataFrame(f_contributions, columns=state_names)
    contrib_df[r"NF$\kappa$B"] = [inputs[i][2] for i in range(len(inputs))]
    contrib_df["IRF"] = [inputs[i][3] for i in range(len(inputs))]
    contrib_df["par_set"] = par_set
    contrib_df.to_csv("%s/%s_best_params_contributions_sweep.csv" % (results_dir, model), index=False)
    end = time.time()
    print("Took %.2f seconds to calculate contributions" % (end - start), flush=True)

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
    for indiv_state in [r"$IRF\cdot IRF_G$", r"$IRF\cdot NF\kappa B$", r"$IRF_G\cdot NF\kappa B$", r"$IRF\cdot IRF_G\cdot NF\kappa B$"]:
        contrib_df_individual1 = contrib_df[contrib_df["state"] == indiv_state]
        # average the contributions of different parameter sets
        contrib_df_individual = contrib_df_individual1.groupby([r"NF$\kappa$B", "IRF"])["contribution"].mean().reset_index()
        contrib_df_individual = contrib_df_individual.pivot_table(index="IRF", columns=r"NF$\kappa$B", values="contribution")
        contrib_df_individual = contrib_df_individual.sort_index(ascending=False)
        plt.subplots()
        ax = sns.heatmap(contrib_df_individual, vmin=0, vmax=vmax)
        ax.invert_yaxis()
        # Only label start, end, and 0.5 of each axis
        n_min, n_max = min(N_vals), max(N_vals)
        pt5 = (n_max - n_min) / 2
        plt.xticks([0, 24, 50], [n_min, pt5, n_max])
        i_min, i_max = min(I_vals), max(I_vals)
        pt5 = (i_max - i_min) / 2
        plt.yticks([0, 24, 50], [i_min, pt5, i_max])
        plt.title("Contribution of %s state" % indiv_state)
        plt.tight_layout()
        plt.savefig("%s/%s_contrib_sweep_%s_state_heatmap.png" % (figures_dir, model, indiv_state))
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


    end = time.time()
    print("Took %.2f minutes to make all plots." % ((end - start)/60), flush=True)


if __name__ == "__main__":
    main()