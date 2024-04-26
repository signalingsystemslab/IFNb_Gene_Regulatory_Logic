# Find parameters that have a greater effect on cpg than on lps
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import scipy.optimize as opt
import time
from multiprocessing import Pool
import seaborn as sns
import scipy.stats.qmc as qmc
import argparse

def cpg_p50_f_effect(t1, t2, t3, t4, t5, h1, h2, kn, k1, k2, kp):
    # -\frac{100^{\text{h2}} \text{k2} \text{kp} \left(\text{k1} (4
#    \text{kn} (\text{t1}+\text{t3}-1)+5
#    (\text{t1}-\text{t4}))-100^{\text{h1}} (4 \text{kn}
#    (\text{t5}-\text{t3})+5 \text{t2})\right)}{(4 \text{kn}+5)
#    \left(100^{\text{h1}}+\text{k1}\right)
#    \left(100^{\text{h2}}+\text{k2}\right) \left(100^{\text{h2}}
#    (\text{kp}+1)+\text{k2}\right)}
    f_effect = -100**h2 * k2 * kp * (k1 * (4 * kn * (t1 + t3 - 1) + 5 * (t1 - t4)) - 
                                     100**h1 * (4 * kn * (t5 - t3) + 5 * t2)) / (
                                         (4 * kn + 5) * (100**h1 + k1) * (100**h2 + k2) * (100**h2 * (kp + 1) + k2))
    return f_effect
    
def lps_p50_f_effect(t1, t2, t3, t4, t5, h1, h2, kn, k1, k2, kp):
#     -\frac{4^{\text{h2}} \text{k2} \text{kp} \left(\text{k1}
#    (\text{kn}
#    (\text{t1}+\text{t3}-1)+\text{t1}-\text{t4})-4^{\text{h1}}
#    (\text{kn}
#    (\text{t5}-\text{t3})+\text{t2})\right)}{(\text{kn}+1)
#    \left(4^{\text{h1}}+\text{k1}\right)
#    \left(4^{\text{h2}}+\text{k2}\right) \left(4^{\text{h2}}
#    (\text{kp}+1)+\text{k2}\right)}
    f_effect = -4**h2 * k2 * kp * (k1 * (kn * (t1 + t3 - 1) + t1 - t4) - 
                                   4**h1 * (kn * (t5 - t3) + t2)) / (
                                       (kn + 1) * (4**h1 + k1) * (4**h2 + k2) * (4**h2 * (kp + 1) + k2))
    return f_effect

def calculate_grid(h1=3, h2=3, t_bounds=(0,1), k_bounds=(10**-3,10**3), seed=0, num_samples=10**6, num_threads=60, num_t_pars=5, num_k_pars=4, num_h_pars=2):
    min_k_order = np.log10(k_bounds[0])
    max_k_order = np.log10(k_bounds[1])
    min_t = t_bounds[0]
    max_t = t_bounds[1]

    seed += 10

    l_bounds = np.concatenate([np.zeros(num_t_pars)+min_t, np.ones(num_k_pars)*min_k_order])
    u_bounds = np.concatenate([np.zeros(num_t_pars)+max_t, np.ones(num_k_pars)*max_k_order])

    print("Calculating grid with %d samples using Latin Hypercube sampling" % num_samples, flush=True)
    sampler=qmc.LatinHypercube(d=num_t_pars+num_k_pars, seed=seed)
    grid_tk = sampler.random(n=num_samples)
    grid_tk = qmc.scale(grid_tk, l_bounds, u_bounds) # rows are parameter sets
    # convert k parameters to log space
    kgrid = grid_tk[:,num_t_pars:]
    kgrid = 10**kgrid
    grid_tk[:,num_t_pars:] = kgrid

    grid = grid_tk.astype(np.float32)

    # Calculate IFNb value at each point in grid
    print("Calculating effects at %d points in grid" % len(grid), flush=True)
    start = time.time()
    with Pool(num_threads) as p:
        cpg_effect = p.starmap(cpg_p50_f_effect, [(grid[i,0], grid[i,1], grid[i,2], grid[i,3], grid[i,4], h1, h2, grid[i,5], grid[i,6], grid[i,7], grid[i,8]) for i in range(len(grid))])
        lps_effect = p.starmap(lps_p50_f_effect, [(grid[i,0], grid[i,1], grid[i,2], grid[i,3], grid[i,4], h1, h2, grid[i,5], grid[i,6], grid[i,7], grid[i,8]) for i in range(len(grid))])

    end = time.time()
    t = end - start
    if t < 60*60:
        print("Time elapsed: %.2f minutes" % (t/60), flush=True)
    else:
        print("Time elapsed: %.2f hours" % (t/3600), flush=True)

    # Filter for values >0
    cpg_effect = np.array(cpg_effect)
    lps_effect = np.array(lps_effect)
    positive_rows = cpg_effect > 0
    grid = grid[positive_rows]
    cpg_effect = cpg_effect[positive_rows]
    lps_effect = lps_effect[positive_rows]
    print("Number of positive values: %d" % len(grid), flush=True)
    positive_values = len(grid)

    # Filter for values where cpg effect > lps effect
    cpg_lps_diff = cpg_effect - lps_effect
    grid = grid[cpg_lps_diff > 0]
    cpg_effect = cpg_effect[cpg_lps_diff > 0]
    lps_effect = lps_effect[cpg_lps_diff > 0]
    cpg_lps_diff = cpg_lps_diff[cpg_lps_diff > 0]
    print("Number of positive values where cpg effect > lps effect: %d" % len(grid), flush=True)
    positive_cpg_bigger = len(grid)

    minimum_cpg_effect = cpg_effect > 0.01
    postitive_cpg_bigger_minimum = sum(minimum_cpg_effect)

    return positive_values, positive_cpg_bigger, postitive_cpg_bigger_minimum

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--grid", action="store_true")
    args = parser.parse_args()

    # Set up directories
    figures_dir = "cpg_lps_effect"
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    # Calculate grid
    h_params = [(1,1),(3,1),(3,3)]

    if args.grid:
        df = pd.DataFrame()
        for h1, h2 in h_params:
            positive_values, positive_cpg_bigger, postitive_cpg_bigger_minimum = calculate_grid(h1=h1, h2=h2)
            print("h1: %d, h2: %d" % (h1, h2))
            print("Positive values: %d, Positive cpg bigger: %d, Positive cpg bigger minimum: %d" % (positive_values, positive_cpg_bigger, postitive_cpg_bigger_minimum))
            # df = df.append({"h1": h1, "h2": h2, "p50 increases\n IFNb for CpG": positive_values, "p50 affects CpG\n more than LPS": positive_cpg_bigger, "p50 effect > 0.0001": postitive_cpg_bigger_minimum}, ignore_index=True)
            df = pd.concat([df, pd.DataFrame({"h1": h1, "h2": h2, r"p50 increases IFN$\beta$ for CpG": positive_values, "p50 affects CpG\n more than LPS": positive_cpg_bigger, "p50 effect > 0.01": postitive_cpg_bigger_minimum}, index=[0])], ignore_index=True)

        df.to_csv("%s/cpg_lps_effect.csv" % figures_dir, index=False)

    df = pd.read_csv("%s/cpg_lps_effect.csv" % figures_dir)

    print("Plotting results")

    df = df.melt(id_vars=["h1", "h2"], var_name="Effect", value_name="Count")
    df["h_values"] = df["h1"].astype(str) + ", " + df["h2"].astype(str)
    fig, ax = plt.subplots(figsize=(45,25))
    p = sns.FacetGrid(data=df, col="Effect", sharey=False, hue="Effect", palette="magma")
    p.map(sns.barplot, "h_values", "Count", order=["1, 1", "3, 1", "3, 3"], ax=ax)
    p.set_titles("{col_name}")
    sns.despine()
    for ax in p.axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(90)
    # plt.subplots_adjust(wspace=5)
    plt.savefig("%s/boxplot_comparing_h_values_facet.png" % figures_dir, bbox_inches="tight", dpi=300)
    plt.close()

    with sns.plotting_context("talk", rc={"lines.markersize": 7}):
        fig, ax = plt.subplots()
        p = sns.barplot(data=df, x="h_values", y="Count", hue="Effect", ax=ax, palette="magma")
        sns.despine()
        plt.xticks(rotation=90)
        plt.yscale("log")
        sns.move_legend(ax, bbox_to_anchor=(1, 0.5), title=None, frameon=False, loc="center left")
        plt.savefig("%s/boxplot_comparing_h_values.png" % figures_dir, bbox_inches="tight")
        plt.close()

if __name__ == "__main__":
    main()