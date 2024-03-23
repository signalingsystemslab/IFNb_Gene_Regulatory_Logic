# Try different binding values and calculate IRF binding probability as a function of IRF concentration
# Plot the results
from three_site_model import *
import matplotlib.pyplot as plt
import pandas as pd
import os
# import scipy.optimize as opt
import time
from multiprocessing import Pool
import seaborn as sns
import scipy.stats.qmc as qmc

figures_dir = "three_site_explore_binding_pars/figures/"
results_dir = "three_site_explore_binding_pars/results/"
os.makedirs(results_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

def calc_irf_total_prob(I,k1,k2,h1,h2):
    # P(IRF-DNA)=\frac{{k_1} {k_2} [IRF]^{{h_1}+{h_2}}+{k_1}   [IRF]^{{h_1}}+{k_2} [IRF]^{{h_2}}}{\left({k_1}   [IRF]^{{h_1}}+1\right) \left({k_2} [IRF]^{{h_2}}+1\right)}
    irf_prob = (k1*k2*I**(h1+h2) + k1*I**h1 + k2*I**h2)/((k1*I**h1 + 1)*(k2*I**h2 + 1))
    return irf_prob

def calc_irf_state_prob(I,k1,k2,h1,h2):
    # P(IRF-IRE)=\frac{[IRF]^{h_1}k_1}{1+[IRF]^{h_1}k_1}
    irf_prob = (I**h1 * k1)/(1 + I**h1 * k1)
    return irf_prob

def calc_irfg_state_prob(I,k1,k2,h1,h2):
    # P(IRF-gIRE)=\frac{[IRF]^{h_2}k_2}{1+[IRF]^{h_2}k_2}
    irfg_prob = (I**h2 * k2)/(1 + I**h2 * k2)
    return irfg_prob


def make_dataframe(inputs, results,prob_name="IRF_prob"):
    state_probs = results
    df = pd.DataFrame({"IRF": [inputs[i][0] for i in range(len(inputs))],
                    "k_1": [inputs[i][1] for i in range(len(inputs))],
                    "k_2": [inputs[i][2] for i in range(len(inputs))],
                    "h_1": [inputs[i][3] for i in range(len(inputs))],
                    "h_2": [inputs[i][4] for i in range(len(inputs))],
                    prob_name: state_probs})

    return df

def calculate_ifnb(pars, data):
    if len(pars) != 5 + 3 + 2:
        raise ValueError("Number of parameters does not match number of t, k, and h parameters")
    t_pars, k_pars, h_pars = pars[:5], pars[5:8], pars[8:]
    N, I = data["NFkB"], data["IRF"]
    #  get_f(t_pars, k_pars, N, I, c_par=None, h_pars=None, scaling=False):
    ifnb = [get_f(t_pars, k_pars, n, i, h_pars=h_pars) for n, i, p in zip(N, I)]
    ifnb = np.array(ifnb)
    return ifnb


def main():
    num_processes = 40

    k_i_vals = np.logspace(-3, 3, 20).round(3)
    I_vals = np.linspace(0, 1, 100)

    # Hill
    h_vals = [1,3,5]
    inputs = [(I, k1, k2, h1, h2) for I in I_vals for k1 in k_i_vals for k2 in k_i_vals for h1 in h_vals for h2 in h_vals]

    num_combinations_k1 = len(inputs)

    # IRF total binding probability
    print("Calculating IRF binding probabilities for %d combinations of parameters" % num_combinations_k1, flush=True)
    start = time.time()
    with Pool(num_processes) as p:
        irf_tot_prob = p.starmap(calc_irf_total_prob, inputs)
        irf_state_prob = p.starmap(calc_irf_state_prob, inputs)
        irfg_state_prob = p.starmap(calc_irfg_state_prob, inputs)
    print("Time to calculate %d combinations: %f seconds" % (num_combinations_k1, time.time() - start), flush=True)
    
    # make dataframe
    df = make_dataframe(inputs, irf_tot_prob, "IRF_tot_prob")
    df["IRF_state_prob"] = irf_state_prob
    df["IRFg_state_prob"] = irfg_state_prob
    df.to_csv("%s/irf_probabilities_hill.csv" % results_dir, index=False)
    df["k_1_linear"] = np.log10(df["k_1"])
    df["k_2_linear"] = np.log10(df["k_2"])

    #Plot IRF state binding probability
    print("Plotting IRF state binding probability", flush=True)
    start = time.time()
    plt.subplots()
    p=sns.relplot(x="IRF", y="IRF_state_prob", hue="k_1_linear", data=df,
            kind="line", legend="full", col="h_1", col_wrap=3)
    p.legend.set_title(r"$k_{1}$")
    for i in range(len(p._legend.texts)):
        leg_label = p._legend.texts[i]
        leg_label.set_text(df["k_1"].unique()[i])
    sns.despine()
    plt.subplots_adjust(top=0.9)
    plt.suptitle("IRF state binding probability with Hill coefficient")
    plt.savefig("%s/irf_state_probabilities_hill.png" % figures_dir)
    plt.close()
    print("Time to plot: %f seconds" % (time.time() - start), flush=True)

    #Plot IRFg state binding probability
    print("Plotting IRFg state binding probability", flush=True)
    plt.subplots()
    p=sns.relplot(x="IRF", y="IRFg_state_prob", hue="k_2_linear", data=df,
            kind="line", legend="full", col="h_2", col_wrap=3)
    p.legend.set_title(r"$k_{2}$")
    for i in range(len(p._legend.texts)):
        leg_label = p._legend.texts[i]
        leg_label.set_text(df["k_2"].unique()[i])
    sns.despine()
    plt.subplots_adjust(top=0.9)
    plt.suptitle("IRFg state binding probability with Hill coefficient")
    plt.savefig("%s/irfg_state_probabilities_hill.png" % figures_dir)
    plt.close()

if __name__ == "__main__":
    main()