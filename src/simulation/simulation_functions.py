import numpy as np
from ifnar_module import change_equations as ifnar_change_equations
from isg_module import change_equations as isg_change_equations
from ifnb_module import change_equations as ifnb_change_equations
from ifnb_protein_module import change_equations as ifnb_protein_change_equations
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
import pandas as pd
from multiprocessing import Pool
import os
import time
plt.style.use("~/IFN_paper/src/theme_bw.mplstyle")

#### State names ####
# IFNb mRNA
# IFNb protein
# IFNAR inactive
# IFNAR active
# ISGF3 inactive
# ISGF3 active
# ISG mRNA


def get_params(file):
    params = {}
    with open(file) as f:
        for line in f:
            (key, val) = line.split(",")
            params[key] = float(val)
    return params

def get_t_params(t_params):
    pars = {}
    for i in range(len(t_params)):
        pars["t%d" % (i+1)] = t_params[i]
    return pars

def read_inputs(protein, stimulus, scale = False):
    filename = "../simulation/%s_timecourse.csv" % protein
    df = pd.read_csv(filename)
    
    time = df["Time"].values
    val = df[stimulus].values

    curve = np.array([time, val])

    if scale:
        if protein == "NFkB":
            curve[1] = curve[1] / 255 # max of all stimuli is at 255 nM
        elif protein == "IRF":
            curve[1] = curve[1] / 60 # max of all stimuli is at 60 nM
    return curve

def get_input(curve, t, input_name=""):
    max_time = curve[0,-1]
    
    if t < 0:
        return 0
    elif t > max_time:
        # if input_name != "p50":
        #   val = curve[1,-1] * np.exp(-(t - max_time + 60)/60)
        # else:
        #   val = curve[1,-1]
        val = curve[1,-1]
    else:
        f = interp1d(curve[0], curve[1])
        # f = CubicSpline(curve[0], curve[1], bc_type="natural")
        val = f([t])[0]
    return val

def get_inputs(N_curve, I_curve, P_curve, t):
    inputs = {}
    inputs["nfkb"] = get_input(N_curve, t)
    inputs["irf"] = get_input(I_curve, t)
    inputs["p50"] = get_input(P_curve, t)
    return inputs

def IFN_model(t, states, params, stim_data):
    N_curve, I_curve, P_curve = stim_data
    inputs = get_inputs(N_curve, I_curve, P_curve, t)

    # States: IFNb
    ifnb_module = ifnb_change_equations(t, states[0:1], params, inputs)
    inputs["ifnb_rna"] = states[0]
    ifnb_prot_module =  ifnb_protein_change_equations(t, states[1:2], params, inputs)
    inputs["ifnb"] = states[1]
    ifnar_module = ifnar_change_equations(t, states[2:6], params, inputs)
    inputs["isgf3"] = states[5]
    isg_module = isg_change_equations(t, states[6:7], params, inputs)
    return np.concatenate((ifnb_module, ifnb_prot_module, ifnar_module, isg_module)) 

def get_steady_state(states0, pars, stim_data_ss, t_eval, num_states = "all"):
    end_time = t_eval[-1]

    # print("Starting first iteration", flush=True)
    states0 = solve_ivp(IFN_model, [0, end_time], states0, t_eval=t_eval, args=(pars, stim_data_ss))
    # print("Finished first iteration", flush=True)
    states0 = states0.y
    if num_states == "all":
        diff = np.max(np.abs(states0[:,-1] - states0[:,0]))
    else:
        diff = np.max(np.abs(states0[0:num_states,-1] - states0[0:num_states,0]))
    i = 0
    while diff > 0.01:
        # print("Difference = %.4f after %d iterations" % (diff, i+1))
        states0 = solve_ivp(IFN_model, [0, end_time], states0[:,-1], t_eval=t_eval, args=(pars, stim_data_ss))
        states0 = states0.y
        if num_states == "all":
            diff = np.max(np.abs(states0[:,-1] - states0[:,0]))
        else:
            diff = np.max(np.abs(states0[0:num_states,-1] - states0[0:num_states,0]))
        i += 1
        if i > 100:
            print("No steady state found after %.2f hours. Max difference = %.4f" % (end_time*i/60, diff))
            break
    states0 = states0[:,-1]
    print("Steady state values found after %.2f hours" % (end_time*i/60))
    return states0

def plot_inputs(input_t, N_curve, I_curve, P_curve, name, directory, ylimits= True):
    input_N = [get_input(N_curve, t) for t in input_t]
    input_I = [get_input(I_curve, t) for t in input_t]
    input_P = [get_input(P_curve, t) for t in input_t]

    fig, ax = plt.subplots(1,3)
    fig.set_size_inches(12,4)
    ax[0].plot(input_N)
    ax[0].set_title("N")
    ax[1].plot(input_I)
    ax[1].set_title("I")
    ax[2].plot(input_P)
    ax[2].set_title("P")
    if ylimits:
        for i in range(3):
            ax[i].set_ylim([-0.01, 1.01])
    xticks = np.arange(0, max(input_t)+1, 60)
    for i in range(3):
        ax[i].set_xticks(xticks)
        ax[i].set_xticklabels(xticks/60)
        ax[i].set_xlabel("Time (hours)")
    plt.suptitle("Input curves for %s stimulation" % name)
    plt.savefig("%s/input_curves_%s.png" % (directory, name))
    plt.close()

def plot_state_mult_pars(num_par_reps, t_eval, timecourse, state, state_names, state_titles, colors, dir, condition, ylim=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(num_par_reps):
        ax.plot(t_eval, timecourse[i,state,:], color=colors[state], alpha=0.1)
    ax.set_ylabel(r"%s (nM)" % state_names[state])
    ax.set_title("%s stimulation, all parameter sets" % condition)
    if ylim is not None:
        ax.set_ylim(ylim)
    xticks = np.arange(0, max(t_eval)+1, 60)
    ax.set_xticks(xticks)
    ax.set_xticklabels(["%d" % (t/60) for t in xticks])
    ax.set_xlabel("Time (hours)")
    plt.savefig("%s/%s_timecourses_%s.png" % (dir, condition, state_titles[state]))
    ylim = ax.get_ylim()
    plt.close()
    return ylim