# Try different binding values and calculate IRF binding probability as a function of IRF concentration
# Plot the results
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

figures_dir = "explore_binding_pars/figures/"
results_dir = "explore_binding_pars/results/"
os.makedirs(results_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

def calc_irf_state_prob(I,k1,k2,kp,P):
    irf_prob = (I*k1)/(1+I*k1)
    irfg_prob = (I*k2)/(1+I*k2+kp*P)
    irf_tot_prob = I*(k1+k2+I*k1*k2+k1*kp*P)/((1+I*k1)*(1+I*k2+kp*P))
    return irf_prob, irfg_prob, irf_tot_prob

def calc_irf_state_prob_hill(I,k1,k2,kp,P,h):
    # print("I: %f, k1: %f, k2: %f, kp: %f, P: %f, h: %f" % (I, k1, k2, kp, P, h))
    irf_prob=(I**(1+h)*k1*(1+I*k2+kp*P))/(1+I**(2+h)*k1*k2+kp*P+I**(1+h)*(k1+k2+k1*kp*P))
    irfg_prob=(I**(1+h)*(1+I*k1)*k2)/(1+I**(2+h)*k1*k2+kp*P+I**(1+h)*(k1+k2+k1*kp*P))
    irf_tot_prob=(I**(1+h)*(k1+k2+I*k1*k2+k1*kp*P))/(1+I**(2+h)*k1*k2+kp*P+I**(1+h)*(k1+k2+k1*kp*P))
    return irf_prob, irfg_prob, irf_tot_prob

def unpack_irf_probabilities(results):
    # Results is list of tuples
    # Each tuple is (irf_prob, irfg_prob, irf_tot_prob)
    irf_probs = np.array([r[0] for r in results])
    irfg_probs = np.array([r[1] for r in results])
    irf_tot_probs = np.array([r[2] for r in results])
    return irf_probs, irfg_probs, irf_tot_probs

def make_dataframe(inputs, results, hill=False):
    irf_probs, irfg_probs, irf_tot_probs = unpack_irf_probabilities(results)
    if hill:
        df = pd.DataFrame({"IRF": [inputs[i][0] for i in range(len(inputs))],
                        "k_i1": [inputs[i][1] for i in range(len(inputs))],
                        "k_i2": [inputs[i][2] for i in range(len(inputs))],
                        "k_p": [inputs[i][3] for i in range(len(inputs))],
                        "h": [inputs[i][5] for i in range(len(inputs))],
                        "IRF_prob": irf_probs,
                        "IRF_G_prob": irfg_probs,
                        "IRF_tot_prob": irf_tot_probs})
    else:
        df = pd.DataFrame({"IRF": [inputs[i][0] for i in range(len(inputs))],
                            "k_i1": [inputs[i][1] for i in range(len(inputs))],
                            "k_i2": [inputs[i][2] for i in range(len(inputs))],
                            "k_p": [inputs[i][3] for i in range(len(inputs))],
                            "IRF_prob": irf_probs,
                            "IRF_G_prob": irfg_probs,
                            "IRF_tot_prob": irf_tot_probs})
    return df

def main():
    num_processes = 40

    t_pars = np.ones(6)

    k_i_vals = np.logspace(-3, 3, 20).round(3)
    kn=1
    k_p_vals = np.logspace(-3, 3, 3)

    I_vals = np.linspace(0, 1, 100)
    N = 1

    # # Non-Hill
    # inputs = [(I, k1, k2, kp, P) for I in I_vals for k1 in k_i_vals for k2 in k_i_vals for kp in k_p_vals for P in [1]]
    # num_combinations = len(inputs)

    # print("Calculating IRF binding probabilities for %d combinations of parameters" % num_combinations, flush=True)
    # start = time.time()
    # with Pool(num_processes) as p:
    #     results = p.starmap(calc_irf_state_prob, inputs)
    #     irf_probs, irfg_probs, irf_tot_probs = unpack_irf_probabilities(results)
    # print("Time to calculate %d combinations: %f seconds" % (num_combinations, time.time() - start), flush=True)

    # # make dataframe
    # df = pd.DataFrame({"IRF": [inputs[i][0] for i in range(num_combinations)],
    #                     "k_i1": [inputs[i][1] for i in range(num_combinations)],
    #                     "k_i2": [inputs[i][2] for i in range(num_combinations)],
    #                     "k_p": [inputs[i][3] for i in range(num_combinations)],
    #                     "IRF_prob": irf_probs,
    #                     "IRF_G_prob": irfg_probs,
    #                     "IRF_tot_prob": irf_tot_probs})
    # df.to_csv("%s/irf_probabilities.csv" % results_dir, index=False)

    # df["k_i1_linear"] = np.log10(df["k_i1"])

    # # Plot
    # start = time.time()
    # plt.subplots()

    # p = sns.relplot(x="IRF", y="IRF_tot_prob", hue="k_i1_linear", data=df,
    #         kind="line", col="k_p", row="k_i2", legend="full")
    # for ax in p.axes.flat:
    #     ax.axvline(0.25, color="black", linestyle="--", alpha=0.5, ymax=0.25)
    #     ax.axvline(0.75, color="black", linestyle="--", alpha=0.5, ymin=0.75)
    # # legend with log scale
    # p.legend.set_title(r"$k_{i1}$")
    # for i in range(len(p._legend.texts)):
    #     leg_label = p._legend.texts[i]
    #     leg_label.set_text(df["k_i1"].unique()[i])
    # sns.despine()
    # plt.savefig("%s/irf_probabilities.png" % figures_dir)
    # plt.close()
    # print("Time to plot: %f seconds" % (time.time() - start), flush=True)


    # # Plot IRF prob and IRF_G prob
    # for kp in k_p_vals:
    #     dat = df[df["k_p"] == kp]
    #     # pivot IRF prob and IRF_G prob into same column with label
    #     dat = dat.melt(id_vars=["IRF", "k_i1", "k_i2", "k_p","k_i1_linear"], value_vars=["IRF_prob", "IRF_G_prob","IRF_tot_prob"], var_name="IRF_state", value_name="Probability")
    #     start = time.time()
    #     plt.subplots()
    #     p = sns.relplot(x="IRF", y="Probability", hue="k_i1_linear", data=dat,
    #             kind="line", col="IRF_state", row="k_i2", legend="full")
    #     for ax in p.axes.flat:
    #         ax.axvline(0.25, color="black", linestyle="--", alpha=0.5, ymax=0.25)
    #         ax.axvline(0.75, color="black", linestyle="--", alpha=0.5, ymin=0.75)
    #     # legend with log scale
    #     p.legend.set_title(r"$k_{i1}$")
    #     for i in range(len(p._legend.texts)):
    #         leg_label = p._legend.texts[i]
    #         leg_label.set_text(df["k_i1"].unique()[i])
    #     sns.despine()
    #     plt.savefig("%s/irf_probabilities_kp_%f.png" % (figures_dir, kp))
    #     plt.close()
    #     print("Time to plot kp %f: %f seconds" % (kp, time.time() - start), flush=True)

    # Hill
    h_vals = [1, 2, 3, 4, 5]
    inputs_k1 = [(I, k1, k2, kp, P, h) for I in I_vals for k1 in k_i_vals for k2 in [1] for kp in [1] for P in [1] for h in h_vals]
    inputs_k2 = [(I, k1, k2, kp, P, h) for I in I_vals for k1 in [1] for k2 in k_i_vals for kp in k_p_vals for P in [1] for h in h_vals]

    k1_vals_other = [i for i in k_i_vals if i not in [1]]
    k2_vals_other = [i for i in k_i_vals if i not in [1]]
    inputs_new = [(I, k1, k2, kp, P, h) for I in I_vals for k1 in k1_vals_other for k2 in k2_vals_other for kp in k_p_vals for P in [1] for h in h_vals]


    num_combinations_k1 = len(inputs_k1)
    num_combinations_k2 = len(inputs_k2)

    # # k1 plots
    # print("Calculating IRF binding probabilities for %d combinations of parameters" % num_combinations_k1, flush=True)
    # start = time.time()
    # with Pool(num_processes) as p:
    #     results = p.starmap(calc_irf_state_prob_hill, inputs_k1)
    # print("Time to calculate %d combinations: %f seconds" % (num_combinations_k1, time.time() - start), flush=True)

    # # make dataframe
    # df_k1 = make_dataframe(inputs_k1, results, hill=True)
    # df_k1.to_csv("%s/irf_probabilities_hill_k1.csv" % results_dir, index=False)
    # df_k1["k_i1_linear"] = np.log10(df_k1["k_i1"])

    # # Plot
    # start = time.time()
    # plt.subplots()
    # p=sns.relplot(x="IRF", y="IRF_prob", hue="k_i1_linear", data=df_k1,
    #         kind="line", legend="full", row="h")
    # for ax in p.axes.flat:
    #     ax.axvline(0.25, color="black", linestyle="--", alpha=0.5, ymax=0.25)
    #     ax.axvline(0.75, color="black", linestyle="--", alpha=0.5, ymin=0.75)
    # # legend with log scale
    # p.legend.set_title(r"$k_{i1}$")
    # for i in range(len(p._legend.texts)):
    #     leg_label = p._legend.texts[i]
    #     leg_label.set_text(df_k1["k_i1"].unique()[i])
    # sns.despine()
    # # title figure
    # plt.title("First IRF binding probability with Hill coefficient")
    # plt.savefig("%s/irf_probabilities_hill_k1.png" % figures_dir)
    # plt.close()
    # print("Time to plot: %f seconds" % (time.time() - start), flush=True)

    # # k2 plots
    # print("Calculating IRF binding probabilities for %d combinations of parameters" % num_combinations_k2, flush=True)
    # start = time.time()
    # with Pool(num_processes) as p:
    #     results = p.starmap(calc_irf_state_prob_hill, inputs_k2)
    # print("Time to calculate %d combinations: %f seconds" % (num_combinations_k2, time.time() - start), flush=True)

    # # make dataframe
    # df_k2 = make_dataframe(inputs_k2, results, hill=True)
    # df_k2.to_csv("%s/irf_probabilities_hill_k2.csv" % results_dir, index=False)
    # df_k2["k_i2_linear"] = np.log10(df_k2["k_i2"])

    # # Plot
    # start = time.time()
    # plt.subplots()
    # p=sns.relplot(x="IRF", y="IRF_G_prob", hue="k_i2_linear", data=df_k2,
    #         kind="line", legend="full", row="h", col="k_p")
    # for ax in p.axes.flat:
    #     ax.axvline(0.25, color="black", linestyle="--", alpha=0.5, ymax=0.25)
    #     ax.axvline(0.75, color="black", linestyle="--", alpha=0.5, ymin=0.75)
    # # legend with log scale
    # p.legend.set_title(r"$k_{i2}$")
    # for i in range(len(p._legend.texts)):
    #     leg_label = p._legend.texts[i]
    #     leg_label.set_text(df_k2["k_i2"].unique()[i])
    # sns.despine()
    # # title figure
    # plt.title("IRFg binding probability with Hill coefficient")
    # plt.savefig("%s/irf_probabilities_hill_k2.png" % figures_dir)
    # plt.close()

    # print("Time to plot: %f seconds" % (time.time() - start), flush=True)

    # # all plots
    # print("Calculating IRF binding probabilities for %d combinations of parameters" % len(inputs_new), flush=True)
    # start = time.time()
    # with Pool(num_processes) as p:
    #     results = p.starmap(calc_irf_state_prob_hill, inputs_new)
    # print("Time to calculate %d combinations: %f seconds" % (len(inputs_new), time.time() - start), flush=True)

    # # make dataframe
    # df = make_dataframe(inputs_new, results, hill=True)
    # df["k_i1_linear"] = np.log10(df["k_i1"])
    # df = pd.concat([df, df_k1, df_k2])
    # df = df.drop_duplicates(subset=["IRF", "k_i1", "k_i2", "k_p", "h"], keep="first")
    # df.to_csv("%s/irf_probabilities_hill.csv" % results_dir, index=False)

    # # Plot
    # start = time.time()
    # for kp in k_p_vals:
    #     dat=df[df["k_p"] == kp]
    #     # print("Number of rows: %d" % len(dat))
    #     # print(dat.head())
    #     plt.subplots()
    #     p=sns.relplot(x="IRF", y="IRF_tot_prob", hue="k_i1_linear", data=dat,
    #         kind="line", legend="full", row="k_i2", col="h")
    #     for ax in p.axes.flat:
    #         ax.axvline(0.25, color="black", linestyle="--", alpha=0.5, ymax=0.25)
    #         ax.axvline(0.75, color="black", linestyle="--", alpha=0.5, ymin=0.75)
    #     # legend with log scale
    #     # p.legend.set_title(r"$k_{i1}$")
    #     for i in range(len(p._legend.texts)):
    #         leg_label = p._legend.texts[i]
    #         leg_label.set_text(df["k_i1"].unique()[i])
    #     sns.despine()
    #     # title figure
    #     plt.title("IRF total binding probability with Hill coefficient, k_p = %f" % kp)
    #     plt.savefig("%s/irf_probabilities_hill_kp_%f.png" % (figures_dir, kp))
    #     plt.close()

    # Determine values where LPS and pIC IRF values are at least different by 0.5
    # LPS
    LPS_inputs = [(0.25, k1, k2, kp, 1, h) for k1 in k_i_vals for k2 in k_i_vals for kp in k_p_vals for h in h_vals]
    with Pool(num_processes) as p:
        LPS_results = p.starmap(calc_irf_state_prob_hill, LPS_inputs)
    LPS_df = make_dataframe(LPS_inputs, LPS_results, hill=True)
    LPS_df.to_csv("%s/irf_probabilities_hill_LPS.csv" % results_dir, index=False)

    # pIC
    pIC_inputs = [(0.75, k1, k2, kp, 1, h) for k1 in k_i_vals for k2 in k_i_vals for kp in k_p_vals for h in h_vals]
    with Pool(num_processes) as p:
        pIC_results = p.starmap(calc_irf_state_prob_hill, pIC_inputs)
    pIC_df = make_dataframe(pIC_inputs, pIC_results, hill=True)
    pIC_df.to_csv("%s/irf_probabilities_hill_pIC.csv" % results_dir, index=False)

    # Calculate the difference in IRF binding probabilities
    keep = (LPS_df["IRF_tot_prob"] <= 0.25) & (pIC_df["IRF_tot_prob"] >= 0.75)
    print("Keeping %d rows" % keep.sum())
    differences = pIC_df["IRF_tot_prob"] - LPS_df["IRF_tot_prob"]
    differences = differences[keep]
    LPS_df = LPS_df[keep]
    LPS_df["diff"] = differences
    LPS_df.to_csv("%s/irf_probabilities_hill_ultrasensitive_LPS.csv" % results_dir, index=False)
    pIC_df = pIC_df[keep]
    pIC_df["diff"] = differences
    pIC_df.to_csv("%s/irf_probabilities_hill_ultrasensitive_pIC.csv" % results_dir, index=False)
    print("Range of differences: %f to %f" % (differences.min(), differences.max()))

    print(LPS_df.describe())
    quantiles_df = LPS_df.describe()
    quantiles_df.to_csv("%s/irf_probabilities_hill_ultrasensitive_quantiles.csv" % results_dir)

    # Pairplot of k1, k2, kp, h
    log_cols = ["k_i1", "k_i2", "k_p"]
    plt.subplots()
    p = sns.pairplot(LPS_df, vars=["k_i1", "k_i2", "k_p", "h"], hue="diff", diag_kind="kde", diag_kws={"hue": None})
    for ax in p.axes.flat:
        if ax.get_xlabel() in log_cols:
            ax.set_xscale("log")
        if ax.get_ylabel() in log_cols:
            ax.set_yscale("log")
    plt.savefig("%s/irf_probabilities_hill_ultrasensitive_pairplot.png" % figures_dir)
    plt.close()


if __name__ == "__main__":
    main()