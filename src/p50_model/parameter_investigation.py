from p50_model import *
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
import pandas as pd
from multiprocessing import Pool
import os
import time
plt.style.use("~/IFN_paper/src/theme_bw.mplstyle")

dir = "results/parameter_investigation/"
os.makedirs(dir, exist_ok=True)

def get_params(file):
    params = {}
    with open(file) as f:
        for line in f:
            (key, val) = line.split(",")
            params[key] = float(val)
    return params

def get_input(curve, t):
    if t < 0:
        return 0
    elif t > len(curve) - 1:
        return 0.01
    else:
        f = interp1d(range(len(curve)), curve)
        val = f([t])[0]
    return val

def get_inputs(N_curve, I_curve, P_curve, t):
    inputs = {}
    inputs["nfkb"] = get_input(N_curve, t)
    inputs["irf"] = get_input(I_curve, t)
    inputs["p50"] = get_input(P_curve, t)
    return inputs

def change_equations(t, states, pars, inputs):
    # Unpack states
    ifnb = states[0]

    # Unpack pars
    t1 = pars["t1"]
    t2 = pars["t2"]
    t3 = pars["t3"]
    t4 = pars["t4"]
    t5 = pars["t5"]
    t6 = pars["t6"]
    t_pars = [t1, t2, t3, t4, t5, t6]
    K = pars["K_i2"]
    C = pars["C"]
    p_syn_ifnb = pars["p_syn_ifnb"]
    p_deg_ifnb = pars["p_deg_ifnb"]

    # Unpack inputs
    nfkb = inputs["nfkb"]
    irf = inputs["irf"]
    p50 = inputs["p50"]

    # Calculate derivatives
    f = get_f(t_pars, K, C, nfkb, irf, p50)
    difnb = f * p_syn_ifnb - p_deg_ifnb * ifnb
    return [difnb]

def IFN_model(t, states, params, stim_data):
    N_curve, I_curve, P_curve = stim_data
    inputs = get_inputs(N_curve, I_curve, P_curve, t)
    # States: IFNb, IFNAR, IFNAR*, ISGF3, ISGF3*, ISG mRNA
    ifnb_module = change_equations(t, states[0:1], params, inputs)
    # if t % 10 < 0.0001:
    #     print("t = %.2f, dIFNb = %.2f" % (t, states[0]))
    return ifnb_module

pars = get_params("../p50_model/results/random_opt/ifnb_best_params_random_global.csv")
for p in pars:
    print(("%s: %.4f") % (p, pars[p]))

t_pars = [pars["t1"], pars["t2"], pars["t3"], pars["t4"], pars["t5"], pars["t6"]]

## Investigate fold change as a function of t_pars
N = 0.25
I = 0.1

WT_f = get_f(t_pars, pars["K_i2"], pars["C"], N, I, 0)
KO_f = get_f(t_pars, pars["K_i2"], pars["C"], N, I, 1)
print("WT fold change = %.2f, with f-value = %.2f for WT and %.2f for KO" % (KO_f/WT_f, WT_f, KO_f))

t_vals = np.linspace(0,1,10)
t1, t2, t3, t4, t5, t6 = np.meshgrid(t_vals, t_vals, t_vals, t_vals, t_vals, t_vals)
t_vals = np.array([t1.flatten(), t2.flatten(), t3.flatten(), t4.flatten(), t5.flatten(), t6.flatten()]).T
num_combinations = len(t1.flatten())
f_vals = np.zeros((2, num_combinations))


print("\n\n")
with Pool(30) as p:
    f_vals[0,:] = p.starmap(get_f, [(t_vals[i,:], pars["K_i2"], pars["C"], N, I, 0) for i in range(num_combinations)])
    f_vals[1,:] = p.starmap(get_f, [(t_vals[i,:], pars["K_i2"], pars["C"], N, I, 1) for i in range(num_combinations)])
fold_change = f_vals[0,:] / f_vals[1,:]

max_fold_change = np.max(fold_change)
max_index = np.argmax(fold_change)
max_t_vals = t_vals[max_index,:]
print("Max fold change = %.2f with t_pars = %s" % (max_fold_change, max_t_vals))

# Plot fold change for each parameter
for i in range(6):
    plt.figure()
    plt.scatter(t_vals[0:num_combinations,i], fold_change)
    plt.xlabel("t%d" % (i+1))
    plt.ylabel("Fold change (p50 KO/WT)")
    plt.savefig("%s/t%d_fold_change.png" % (dir, i+1))

# Plot fold change for each parameter pair
fig, ax = plt.subplots(6,6)
# make plot size big
fig.set_size_inches(12,12)
# spread out plots
fig.subplots_adjust(hspace=0.4, wspace=0.5)
for i in range(6):
    for j in range(6):
        a = ax[i,j].scatter(t_vals[0:num_combinations,i], t_vals[0:num_combinations,j], c=fold_change)
        ax[i,j].set_xlabel("t%d" % (i+1))
        ax[i,j].set_ylabel("t%d" % (j+1))
plt.colorbar(a, ax=ax.ravel().tolist(), label="Fold change (p50 KO/WT)")
plt.savefig("%s/t_pair_fold_change.png" % dir)