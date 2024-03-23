# Try different binding values and calculate IRF binding probability as a function of IRF concentration
# Plot the results
from two_site_model import *
from param_scan_k_pars_hill import plot_parameters, plot_predictions, plot_state_probabilities, calc_state_prob
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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

def calc_irf_total_prob(I,ki,h):
    # irf_prob = \frac{i^{h+1} k_i}{i^{h+1} k_i+1}
    irf_prob = (I**(1+h)*ki)/(1+I**(1+h)*ki)
    return irf_prob

def calc_irf_state_prob(I,N, ki, kn, h):
    # \frac{i^{h+1} k_i}{\left(n k_n+1\right) \left(i^{h+1}k_i+1\right)}
    irf_prob = (I**(1+h)*ki)/((N*kn+1)*(1+I**(1+h)*ki))
    return irf_prob

def calc_two_state_prob(I,N,ki,kn,h):
#     \frac{n i^{h+1} k_i k_n}{\left(n k_n+1\right) \left(i^{h+1}k_i+1\right)}
    irf_nfkb_prob = (N*I**(1+h)*ki*kn)/((N*kn+1)*(1+I**(1+h)*ki))
    return irf_nfkb_prob


def make_dataframe(inputs, results):
    state_probs = results
    if len(inputs[0]) == 3:
        df = pd.DataFrame({"IRF": [inputs[i][0] for i in range(len(inputs))],
                        "k_i": [inputs[i][1] for i in range(len(inputs))],
                        "h": [inputs[i][2] for i in range(len(inputs))],
                        "IRF_prob": state_probs})
    elif len(inputs[0]) == 5:
        df = pd.DataFrame({"IRF": [inputs[i][0] for i in range(len(inputs))],
                        r"NF$\kappa$B": [inputs[i][1] for i in range(len(inputs))],
                        "k_i": [inputs[i][2] for i in range(len(inputs))],
                        "k_n": [inputs[i][3] for i in range(len(inputs))],
                        "h": [inputs[i][4] for i in range(len(inputs))],
                        "state_prob": state_probs})

    return df

def calculate_ifnb(pars, data):
    if len(pars) != 2 + 2 + 1:
        raise ValueError("Number of parameters does not match number of t, k, and h parameters")
    t_pars, k_pars, h_pars = pars[:2], pars[2:4], pars[4:]
    N, I, P = data["NFkB"], data["IRF"], data["p50"]
    ifnb = [get_f(t_pars, k_pars, n, i, p, h_pars=h_pars) for n, i, p in zip(N, I, P)]
    ifnb = np.array(ifnb)
    return ifnb

def make_state_heatmap(*args, **kwargs):
    data = kwargs.pop("data")
    d = data.pivot(index=args[1], columns=args[0], values=args[2])
    ax = sns.heatmap(d, **kwargs)

    # Set x and y tick locators
    x_ticks = np.linspace(np.min(data[args[0]]), np.max(data[args[0]]), 10)
    y_ticks = np.linspace(np.min(data[args[1]]), np.max(data[args[1]]), 10)
    ax.xaxis.set_major_locator(ticker.FixedLocator(x_ticks))
    ax.yaxis.set_major_locator(ticker.FixedLocator(y_ticks))

    # Format x and y tick labels
    ax.set_xticklabels(["%.2f" % i for i in x_ticks])
    ax.set_yticklabels(["%.2f" % i for i in y_ticks])

    # # Only label start, end, and 0.5 of each axis
    # min_x, max_x = np.min(data[args[0]]), np.max(data[args[0]])
    # pt5 = (max_x - min_x) / 2
    # tick_locs = ax.get_xticks()
    # plt.xticks([tick_locs[0], tick_locs[-1], tick_locs[int(len(tick_locs) / 2)]], [min_x, max_x, pt5])
    # min_y, max_y = np.min(data[args[1]]), np.max(data[args[1]])
    # pt5 = (max_y - min_y) / 2
    # tick_locs = ax.get_yticks()
    # plt.yticks([tick_locs[0], tick_locs[-1], tick_locs[int(len(tick_locs) / 2)]], [min_y, max_y, pt5])

def main():
    num_processes = 40

    k_i_vals = np.logspace(-3, 3, 20).round(3)
    I_vals = np.linspace(0, 1, 100)

    # Hill
    h_vals = [1, 2, 3, 4, 5]
    inputs = [(I, k1, h) for I in I_vals for k1 in k_i_vals for h in h_vals]

    num_combinations_k1 = len(inputs)

    # IRF total binding probability
    print("Calculating IRF binding probabilities for %d combinations of parameters" % num_combinations_k1, flush=True)
    start = time.time()
    with Pool(num_processes) as p:
        results = p.starmap(calc_irf_total_prob, inputs)
    print("Time to calculate %d combinations: %f seconds" % (num_combinations_k1, time.time() - start), flush=True)

    # make dataframe
    df = make_dataframe(inputs, results)
    df.to_csv("%s/irf_probabilities_hill.csv" % results_dir, index=False)
    df["k_i_linear"] = np.log10(df["k_i"])

    # Plot
    start = time.time()
    plt.subplots()
    p=sns.relplot(x="IRF", y="IRF_prob", hue="k_i_linear", data=df,
            kind="line", legend="full", col="h", col_wrap=3)
    for ax in p.axes.flat:
        ax.axvline(0.001, color="black", linestyle="--", alpha=0.5, ymax=0.1)
        ax.axvline(0.25, color="black", linestyle="--", alpha=0.5, ymax=0.25)
        ax.axvline(0.75, color="black", linestyle="--", alpha=0.5, ymin=0.75)
    # legend with log scale
    p.legend.set_title(r"$k_{i}$")
    for i in range(len(p._legend.texts)):
        leg_label = p._legend.texts[i]
        leg_label.set_text(df["k_i"].unique()[i])
    sns.despine()
    plt.subplots_adjust(top=0.9)
    plt.suptitle("Two-site IRF binding probability with Hill coefficient")
    plt.savefig("%s/irf_probabilities_hill.png" % figures_dir)
    plt.close()
    print("Time to plot: %f seconds" % (time.time() - start), flush=True)

    # IRF state only binding probability
    k_i_vals = np.logspace(-3, 3, 5).round(3)
    inputs = [(I, N, ki, kn, h) for I in I_vals for N in I_vals for ki in k_i_vals for kn in k_i_vals for h in h_vals]
    num_combinations_k1_k2 = len(inputs)
    print("Calculating IRF state binding probabilities for %d combinations of parameters" % num_combinations_k1_k2, flush=True)
    start = time.time()
    with Pool(num_processes) as p:
        results = p.starmap(calc_irf_state_prob, inputs)
    print("Time to calculate %d combinations: %f seconds" % (num_combinations_k1_k2, time.time() - start), flush=True)

    # make dataframe
    df = make_dataframe(inputs, results)
    df.to_csv("%s/irf_state_probabilities_hill.csv" % results_dir, index=False)

    # Plot heatmap of IRF state binding probabilities where x= IRF, y= NFkB, color= probability, facet cols= ki, facet rows= k_n
    start = time.time()
    for h in h_vals:
        dat = df[(df["h"] == h)]
        p = sns.FacetGrid(dat, col="k_i", row="k_n", margin_titles=True)
        cbar_ax = p.figure.add_axes([.92, .3, .02, .4])
        p.map_dataframe(make_state_heatmap, "IRF", r"NF$\kappa$B", "state_prob", data=df, cbar_ax=cbar_ax)
        p.set_axis_labels(r"$IRF$", r"$NF\kappa B$")
        plt.subplots_adjust(top=0.93, right=0.9)
        p.figure.suptitle("IRF state binding probability with Hill coefficient h=%d" % h)
        plt.savefig("%s/irf_state_probabilities_hill_h%d.png" % (figures_dir, h))
        plt.close()

    print("Time to plot: %f minutes" % ((time.time() - start) / 60), flush=True)

    # Two state binding probability
    print("Calculating two state binding probabilities for %d combinations of parameters" % num_combinations_k1_k2, flush=True)
    start = time.time()
    with Pool(num_processes) as p:
        results = p.starmap(calc_two_state_prob, inputs)
    print("Time to calculate %d combinations: %f seconds" % (num_combinations_k1_k2, time.time() - start), flush=True)

    # make dataframe
    df = make_dataframe(inputs, results)
    df.to_csv("%s/two_state_probabilities_hill.csv" % results_dir, index=False)

    # Plot heatmap of IRF state binding probabilities where x= IRF, y= NFkB, color= probability, facet cols= ki, facet rows= k_n
    start = time.time()
    for h in h_vals:
        dat = df[(df["h"] == h)]
        p = sns.FacetGrid(dat, col="k_i", row="k_n", margin_titles=True)
        cbar_ax = p.figure.add_axes([.92, .3, .02, .4])
        p.map_dataframe(make_state_heatmap, "IRF", r"NF$\kappa$B", "state_prob", data=df, cbar_ax=cbar_ax)
        p.set_axis_labels(r"$IRF$", r"$NF\kappa B$")
        plt.subplots_adjust(top=0.93, right=0.9)
        p.figure.suptitle("Two state binding probability with Hill coefficient h=%d" % h)
        plt.savefig("%s/two_state_probabilities_hill_h%d.png" % (figures_dir, h))
        plt.close()

    print("Time to plot: %f minutes" % ((time.time() - start) / 60), flush=True)


    # # Determine values where LPS and pIC IRF values are at least different by 0.5
    # # LPS
    # LPS_inputs = [(0.25, k1, k2, kp, 1, h) for k1 in k_i_vals for k2 in k_i_vals for kp in k_p_vals for h in h_vals]
    # with Pool(num_processes) as p:
    #     LPS_results = p.starmap(calc_irf_total_prob_hill, LPS_inputs)
    # LPS_df = make_dataframe(LPS_inputs, LPS_results, hill=True)
    # LPS_df.to_csv("%s/irf_probabilities_hill_LPS.csv" % results_dir, index=False)

    # # pIC
    # pIC_inputs = [(0.75, k1, k2, kp, 1, h) for k1 in k_i_vals for k2 in k_i_vals for kp in k_p_vals for h in h_vals]
    # with Pool(num_processes) as p:
    #     pIC_results = p.starmap(calc_irf_total_prob_hill, pIC_inputs)
    # pIC_df = make_dataframe(pIC_inputs, pIC_results, hill=True)
    # pIC_df.to_csv("%s/irf_probabilities_hill_pIC.csv" % results_dir, index=False)

    # # Calculate the difference in IRF binding probabilities
    # keep = (LPS_df["IRF_tot_prob"] <= 0.25) & (pIC_df["IRF_tot_prob"] >= 0.75)
    # print("Keeping %d rows" % keep.sum())
    # differences = pIC_df["IRF_tot_prob"] - LPS_df["IRF_tot_prob"]
    # differences = differences[keep]
    # LPS_df = LPS_df[keep]
    # LPS_df["diff"] = differences
    # LPS_df.to_csv("%s/irf_probabilities_hill_ultrasensitive_LPS.csv" % results_dir, index=False)
    # pIC_df = pIC_df[keep]
    # pIC_df["diff"] = differences
    # pIC_df.to_csv("%s/irf_probabilities_hill_ultrasensitive_pIC.csv" % results_dir, index=False)
    # print("Range of differences: %f to %f" % (differences.min(), differences.max()))

    # print(LPS_df.describe())
    # quantiles_df = LPS_df.describe()
    # quantiles_df.to_csv("%s/irf_probabilities_hill_ultrasensitive_quantiles.csv" % results_dir)

    # LPS_df = pd.read_csv("%s/irf_probabilities_hill_ultrasensitive_LPS.csv" % results_dir)

    # # Pairplot of k1, k2, kp, h
    # log_cols = ["k_i", "k_i2", "k_p"]
    # plt.subplots()
    # p = sns.pairplot(LPS_df, vars=["k_i1", "k_i2", "k_p", "h"], hue="diff", diag_kind="kde", diag_kws={"hue": None})
    # xmin, xmax = min(LPS_df["k_i1"].min(), LPS_df["k_i2"].min(), LPS_df["k_p"].min()) / 5, max(LPS_df["k_i1"].max(), LPS_df["k_i2"].max(), LPS_df["k_p"].max()) * 5
    # ymin, ymax = xmin, xmax

    # for ax in p.axes.flat:
    #     if ax.get_xlabel() in log_cols:
    #         ax.set_xscale("log")
    #         ax.set_xlim([xmin, xmax]) 
    #     if ax.get_ylabel() in log_cols:
    #         ax.set_yscale("log")
    #         ax.set_ylim([ymin, ymax]) 
    # plt.savefig("%s/irf_probabilities_hill_ultrasensitive_pairplot.png" % figures_dir)
    # plt.close()

if __name__ == "__main__":
    main()