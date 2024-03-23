from two_site_model import *
import matplotlib.pyplot as plt
import pandas as pd
import os
# import scipy.optimize as opt
import time
from multiprocessing import Pool
import argparse
import seaborn as sns
import scipy.stats.qmc as qmc

figures_dir = "two_site_contrib/figures"
results_dir = "two_site_contrib/results"
os.makedirs(results_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)
num_t_pars = 2
num_k_pars = 2
num_h_pars = 1

def main():
    best_fit_dir ="param_scan_2site/results/"
    model = "two_site_hill"
    num_threads = 40
    best_20_pars_df = pd.read_csv("%s/%s_all_best_20_pars.csv" % (best_fit_dir, model))
    best_tpars, best_kpars, best_hpars = best_20_pars_df.iloc[:, :num_t_pars].values, best_20_pars_df.iloc[:, num_t_pars:num_t_pars+num_k_pars].values, best_20_pars_df.iloc[:, num_t_pars+num_k_pars:num_t_pars+num_k_pars+num_h_pars].values

    # Calculate relative contributions of each state for values of N and I
    N_vals = np.linspace(0, 1, 51)
    I_vals = np.linspace(0, 1, 51)
    N, I = np.meshgrid(N_vals, I_vals)
    N = N.flatten()
    I = I.flatten()
    inputs = [(n, i, None, tpars, kpars, hpars[0]) for tpars, kpars, hpars in zip(best_tpars, best_kpars, best_hpars) for n, i in zip(N, I)]
    par_set = np.repeat(range(len(best_tpars)), len(N_vals) * len(I_vals))

    start = time.time()
    with Pool(num_threads) as p:
        results = p.starmap(get_contribution, [i for i in inputs])
    f_contributions = np.array([r[0] for r in results])
    state_names = results[0][1]
    state_names = np.array(state_names)
    state_names = ["%s state" % s for s in state_names]
    contrib_df = pd.DataFrame(f_contributions, columns=state_names)
    contrib_df[r"NF$\kappa$B"] = [inputs[i][0] for i in range(len(inputs))]
    contrib_df["IRF"] = [inputs[i][1] for i in range(len(inputs))]
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
    # vmax = contrib_df["contribution"].max()
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
        ax=sns.heatmap(contrib_df_individual, vmin=0, vmax=1)
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
        plt.savefig("%s/%s_contrib_sweep_%s_state_heatmap.png" % (figures_dir, model, state_name_no_tex))
        plt.close()

    print("Took %.2f seconds to plot heatmaps" % (time.time() - t), flush=True)

if __name__ == "__main__":
    main()