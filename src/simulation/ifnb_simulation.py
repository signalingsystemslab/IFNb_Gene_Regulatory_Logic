from ifnb_module import *
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
import pandas as pd
from multiprocessing import Pool
import os
import time
plt.style.use("~/IFN_paper/src/theme_bw.mplstyle")

dir = "results/ifnb_simulation_syn_pars/"
os.makedirs(dir, exist_ok=True)

def get_params(file):
    params = {}
    with open(file) as f:
        for line in f:
            (key, val) = line.split(",")
            params[key] = float(val)
    return params

def read_inputs(protein, stimulus):
    filename = "../simulation/%s_timecourse.csv" % protein
    df = pd.read_csv(filename)
    
    time = df["Time"].values
    val = df[stimulus].values

    curve = np.array([time, val])
    return curve

def get_input(curve, t, input_name=""):
    max_time = curve[0,-1]
    
    if t < 0:
        return 0
    elif t > max_time:
        if input_name != "p50":
          val = curve[1,-1] * np.exp(-(t - max_time + 1)/60)
        else:
          val = curve[1,-1]
    else:
        f = interp1d(curve[0], curve[1])
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
    ifnb_module = change_equations(t, states[0:1], params, inputs)
    return ifnb_module

def get_steady_state(states0, pars, stim_data_ss, t_eval):
    end_time = t_eval[-1]

    # print("Starting first iteration", flush=True)
    states0 = solve_ivp(IFN_model, [0, end_time], states0, t_eval=t_eval, args=(pars, stim_data_ss))
    # print("Finished first iteration", flush=True)
    states0 = states0.y
    diff = np.abs(states0[0,-1] - states0[0,0])
    i = 0
    while diff > 0.01:
        # print("Difference = %.4f after %d iterations" % (diff, i+1))
        states0 = solve_ivp(IFN_model, [0, end_time], states0[:,-1], t_eval=t_eval, args=(pars, stim_data_ss))
        states0 = states0.y
        diff = np.abs(states0[0,-1] - states0[0,0])
        i += 1
        if i > 100:
            print("No steady state found after %.2f hours. Max difference = %.4f" % (end_time*i/60, diff))
            break
    states0 = states0[:,-1]
    print("Steady state values found after %.2f hours" % (end_time*i/60))
    return states0

def plot_inputs(input_t, N_curve, I_curve, P_curve, name, directory):
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
    for i in range(3):
        ax[i].set_ylim([-0.01, 1.01])
    plt.suptitle("Input curves for %s stimulation" % name)
    plt.savefig("%s/input_curves_%s.png" % (directory, name))
    plt.close()

def full_simulation(states0, pars, name, stimulus, genotype, directory, stim_time = 60*8, stim_data=None, plot = True):
    name = "%s_%s_%s" % (name, stimulus, genotype)

    if stim_data is None:
        # Inputs
        if stimulus in ["CpG", "LPS", "pIC"]:
            I_curve = read_inputs("IRF", stimulus)
            N_curve = read_inputs("NFkB", stimulus)
        else:
            raise ValueError("Stimulus must be CpG, LPS, pIC, or other")

        if genotype in ["WT", "p50KO"]:
            P_values = {"WT": 1, "p50KO": 0}
            P_curve_vals = [P_values[genotype] for i in range(stim_time+180)]
            P_curve = np.array([np.arange(stim_time+180), P_curve_vals])
        elif genotype == "other":
            P_curve = stim_data[2]
        else:
            raise ValueError("Genotype must be WT, p50KO, or other")

        stim_data = [N_curve, I_curve, P_curve]
    else:
        N_curve, I_curve, P_curve = stim_data
        stim_data = [N_curve, I_curve, P_curve]

    stim_data_ss = [[0.00001 for i in range(stim_time+120)] for i in range(2)]
    stim_data_ss = [stim_data_ss[0], stim_data_ss[1], P_curve]

    if plot:
        # Plot inputs
        input_t = np.linspace(0, stim_time+120, stim_time+120+1)
        plot_inputs(input_t, N_curve, I_curve, P_curve, name, directory)

    ## Simulate model
    t_eval = np.linspace(0, stim_time+120, stim_time+120+1)

    # Get steady state
    states0 = get_steady_state(states0, pars, stim_data_ss, t_eval)
    print("Steady state IFNb values for %s: %s" % (name, states0), flush=True)

    # Integrate model
    states = solve_ivp(IFN_model, [0, t_eval[-1]], states0, t_eval=t_eval, args=(pars, stim_data))
    return states, t_eval, stim_data

def main():
    # (states0, pars, name, stimulus, genotype, directory, stim_time = 60*8, stim_data=None, plot = True
    states0 = np.array([0.01])
    genotype = "WT"
    stim_time = 60*8

    # Load all parameter sets
    params = np.loadtxt("../p50_model/opt_syn_datasets/results/p50_all_datasets_pars_local.csv", delimiter=",")
    num_par_reps = np.size(params,0)

    # Simulate for cpg and lps

    with Pool(40) as p:
        cpg_results = p.starmap(full_simulation, [(states0, params[i,:], "CpG_%d" % i, "CpG", genotype, dir, stim_time, None, False) for i in range(num_par_reps)])
        lps_results = p.starmap(full_simulation, [(states0, params[i,:], "LPS_%d" % i, "LPS", genotype, dir, stim_time, None, False) for i in range(num_par_reps)])

    cpg_timecourses = [cpg_results[i][0].y for i in range(num_par_reps)]
    lps_timecourses = [lps_results[i][0].y for i in range(num_par_reps)]
    t_eval = cpg_results[0][1]
    np.savetxt("%s/cpg_results.csv" % dir, cpg_timecourses, delimiter=",")
    np.savetxt("%s/lps_results.csv" % dir, lps_timecourses, delimiter=",")
    np.savetxt("%s/t_eval.csv" % dir, t_eval, delimiter=",")

    # Plot inputs
    cpg_stim_data = cpg_results[0][2]
    lps_stim_data = lps_results[0][2]

    input_t = np.linspace(0, stim_time+120, stim_time+120+1)
    plot_inputs(input_t, cpg_stim_data[0], cpg_stim_data[1], cpg_stim_data[2], "CpG", dir)
    plot_inputs(input_t, lps_stim_data[0], lps_stim_data[1], lps_stim_data[2], "LPS", dir)

    # Load results
    cpg_timecourses = np.loadtxt("%s/cpg_results.csv" % dir, delimiter=",")
    lps_results = np.loadtxt("%s/lps_results.csv" % dir, delimiter=",")
    t_eval = np.loadtxt("%s/t_eval.csv" % dir, delimiter=",")

    # Plot results
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in len(num_par_reps):
        ax.plot(t_eval, cpg_timecourses[i,:], color="b", alpha=0.1)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel(r"IFN$\beta$ mRNA")
    ax.set_title("CpG stimulation, all parameter sets")
    plt.savefig("%s/cpg_timecourses.png" % dir)
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in len(num_par_reps):
        ax.plot(t_eval, lps_timecourses[i,:], color="b", alpha=0.1)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel(r"IFN$\beta$ mRNA")
    ax.set_title("LPS stimulation, all parameter sets")
    plt.savefig("%s/lps_timecourses.png" % dir)

    # Calculate fold change for all parameter sets
    cpg_fc = np.zeros(num_par_reps)
    lps_fc = np.zeros(num_par_reps)

    for i in range(num_par_reps):
        cpg_fc[i] = np.max(cpg_timecourses[i,:]) / cpg_timecourses[i,0]
        lps_fc[i] = np.max(lps_timecourses[i,:]) / lps_timecourses[i,0]

    # Plot fold change
    cpg_fc_sort = np.sort(cpg_fc)
    lps_fc_sort = np.sort(lps_fc)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(cpg_fc_sort, color="b", label="CpG")
    ax.plot(lps_fc_sort, color="r", label="LPS")
    ax.set_xlabel("Parameter set (sorted)")
    ax.set_ylabel("Fold change")
    ax.set_title("Fold change in IFN$\beta$ mRNA")
    plt.legend()
    plt.savefig("%s/fold_change.png" % dir)

    # mRNA data
    mRNA_data = pd.read_csv("../data/ifnb_mRNA_counts_cheng17.csv")
    cpg_mRNA = mRNA_data[mRNA_data["stimulus"] == "CPG" and mRNA_data["time"] > 0]
    cpg_fc_data = max(cpg_mRNA["fold_change"])
    lps_mRNA = mRNA_data[mRNA_data["stimulus"] == "LPS" and mRNA_data["time"] > 0]
    lps_fc_data = max(lps_mRNA["fold_change"])

    print("CpG mRNA fold change: %.2f" % cpg_fc_data)
    print("LPS mRNA fold change: %.2f" % lps_fc_data)

    # Find row in cpg_fc closest to cpg_fc_data
    cpg_fc_closest = np.argmin(np.abs(cpg_fc - cpg_fc_data))
    lps_fc_closest = np.argmin(np.abs(lps_fc - lps_fc_data))

    print("CpG parameter set closest to data: %s" % params[cpg_fc_closest,:])
    print("LPS parameter set closest to data: %s" % params[lps_fc_closest,:])

    # plot results
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in len(num_par_reps):
        if i == cpg_fc_closest:
            ax.plot(t_eval, cpg_timecourses[i,:], color="r", alpha=1)
        else:
            ax.plot(t_eval, cpg_timecourses[i,:], color="b", alpha=0.1)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel(r"IFN$\beta$ mRNA")
    ax.set_title("CpG stimulation, parameter set closest to CpG data")
    plt.savefig("%s/cpg_timecourses_closest.png" % dir)
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in len(num_par_reps):
        if i == lps_fc_closest:
            ax.plot(t_eval, lps_timecourses[i,:], color="r", alpha=1)
        else:
            ax.plot(t_eval, lps_timecourses[i,:], color="b", alpha=0.1)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel(r"IFN$\beta$ mRNA")
    ax.set_title("LPS stimulation, parameter set closest to LPS data")
    plt.savefig("%s/lps_timecourses_closest.png" % dir)
    plt.close()

    # Find parameter set where LPS and CpG are both close to data
    cpg_diffs = np.abs(cpg_fc - cpg_fc_data)
    lps_diffs = np.abs(lps_fc - lps_fc_data)
    both_closest = np.argmin(cpg_diffs + lps_diffs)
    print("Parameter set closest to both data sets: %s" % params[both_closest,:])

    # plot results
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in len(num_par_reps):
        if i == both_closest:
            ax.plot(t_eval, cpg_timecourses[i,:], color="r", alpha=1)
        else:
            ax.plot(t_eval, cpg_timecourses[i,:], color="b", alpha=0.1)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel(r"IFN$\beta$ mRNA")
    ax.set_title("CpG stimulation, parameter set closest to both data sets")
    plt.savefig("%s/cpg_timecourses_both_close.png" % dir)
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in len(num_par_reps):
        if i == both_closest:
            ax.plot(t_eval, lps_timecourses[i,:], color="r", alpha=1)
        else:
            ax.plot(t_eval, lps_timecourses[i,:], color="b", alpha=0.1)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel(r"IFN$\beta$ mRNA")
    ax.set_title("LPS stimulation, parameter set closest to both data sets")
    plt.savefig("%s/lps_timecourses_both_close.png" % dir)



if __name__ == "__main__":
    main()