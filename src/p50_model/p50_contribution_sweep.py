from p50_model_force_t import *
import matplotlib.pyplot as plt
import pandas as pd
import os
# import scipy.optimize as opt
import time
from multiprocessing import Pool
import argparse
import seaborn as sns
import scipy.stats.qmc as qmc

figures_dir = "p50_contrib/figures"
results_dir = "p50_contrib/results"
os.makedirs(results_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)
num_t_pars = 4
num_k_pars = 4
num_h_pars = 2

def main():
    h_pars = "3_1_1"
    best_fit_dir ="parameter_scan_force_t/results/"
    model = "p50"
    num_threads = 40
    # col_names = ["t%d" % i for i in range(1, num_t_pars+1)] + ["k%d" % i for i in range(1, num_k_pars+1)]
    best_20_pars_df = pd.read_csv("%s/%s_force_t_best_fits_pars.csv" % (best_fit_dir, model))
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
    N_vals = np.linspace(0, 1, 51)
    I_vals = np.linspace(0, 1, 51)
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
        np.savetxt("%s/%s_state_names.txt" % (results_dir, model), state_names, fmt="%s")

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
            plt.savefig("%s/%s_contrib_sweep_%s_heatmap_p%d.png" % (figures_dir, model, state_name_no_tex, p50))
            plt.close()

        print("Took %.2f seconds to plot heatmaps" % (time.time() - t), flush=True)


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

    # save 
    contrib_df.to_csv("%s/%s_specific_conds_contributions.csv" % (results_dir, model), index=False)

if __name__ == "__main__":
    main()