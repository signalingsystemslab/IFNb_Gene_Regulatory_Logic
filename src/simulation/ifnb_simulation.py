from ifnb_module import *
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.integrate import solve_ivp
import pandas as pd
from multiprocessing import Pool
import os
import time
plt.style.use("~/IFN_paper/src/theme_bw.mplstyle")

parent_dir = "results/ifnb_simulation_syn_pars/"
os.makedirs(parent_dir, exist_ok=True)

def get_params(file):
    params = {}
    with open(file) as f:
        for line in f:
            (key, val) = line.split(",")
            params[key] = float(val)
    return params

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
        # f = interp1d(curve[0], curve[1])
        f = CubicSpline(curve[0], curve[1], bc_type="natural")
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
    plt.suptitle("Input curves for %s stimulation" % name)
    plt.savefig("%s/input_curves_%s.png" % (directory, name))
    plt.close()

def full_simulation(states0, params, name, stimulus, genotype, directory, stim_time = 60*8, stim_data=None, plot = True, p_deg_ifnb = 1, p_syn_ifnb = 1):
    name = "%s_%s_%s" % (name, stimulus, genotype)

    # Make pars a dictionary
    pars = {"K_i2": 1, "C": 1, "p_syn_ifnb": p_syn_ifnb, "p_deg_ifnb": p_deg_ifnb}
    for i in range(len(params)):
        pars["t%d" % (i+1)] = params[i]

    if stim_data is None:
        # Inputs
        if stimulus in ["CpG", "LPS", "pIC"]:
            I_curve = read_inputs("IRF", stimulus, scale=True)
            N_curve = read_inputs("NFkB", stimulus, scale=True)
        else:
            raise ValueError("Stimulus must be CpG, LPS, pIC, or other")

        if genotype in ["WT", "p50KO"]:
            P_values = {"WT": 1, "p50KO": 0}
            P_curve_vals = [P_values[genotype] for i in range(stim_time+1)]
            P_curve = np.array([np.arange(stim_time+1), P_curve_vals])
        elif genotype == "other":
            P_curve = stim_data[2]
        else:
            raise ValueError("Genotype must be WT, p50KO, or other")

        stim_data = [N_curve, I_curve, P_curve]
    else:
        N_curve, I_curve, P_curve = stim_data
        stim_data = [N_curve, I_curve, P_curve]

    stim_data_ss = [0.00001 for i in range(stim_time+1)]
    stim_data_ss = np.array([np.arange(stim_time+1), stim_data_ss])
    stim_data_ss = [stim_data_ss, stim_data_ss, P_curve]

    if plot:
        # Plot inputs
        input_t = np.linspace(0, stim_time+1, stim_time+1+1)
        plot_inputs(input_t, N_curve, I_curve, P_curve, name, directory)

    ## Simulate model
    t_eval = np.linspace(0, stim_time+1, stim_time+1+1)

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

    synthesis_rates = np.logspace(-5, 0, 6)
    degredation_rates = np.logspace(-5, 0, 6)
    fc_diffs = []
    for p_deg_ifnb in degredation_rates:
        dir = "%s/p_deg_ifnb_%.5f" % (parent_dir, p_deg_ifnb)
        os.makedirs(dir, exist_ok=True)

        print("\n##########\nDegradation rate: %.5f\n##########\n" % p_deg_ifnb, flush=True)

        # # Simulate for cpg and lps
        # with Pool(40) as p:
        #     cpg_results = p.starmap(full_simulation, [(states0, params[i,:], "CpG_%d" % i, "CpG", genotype, dir, stim_time, None, False, p_deg_ifnb, 1) for i in range(num_par_reps)])
        #     lps_results = p.starmap(full_simulation, [(states0, params[i,:], "LPS_%d" % i, "LPS", genotype, dir, stim_time, None, False, p_deg_ifnb, 1) for i in range(num_par_reps)])

        # # print(cpg_results[1][0].y[0,:])
        # cpg_timecourses = [cpg_results[i][0].y[0,:] for i in range(num_par_reps)]
        # cpg_timecourses = np.array(cpg_timecourses)
        # lps_timecourses = [lps_results[i][0].y[0,:] for i in range(num_par_reps)]
        # lps_timecourses = np.array(lps_timecourses)
        # t_eval = cpg_results[0][1]
        # np.savetxt("%s/cpg_results.csv" % dir, cpg_timecourses, delimiter=",")
        # np.savetxt("%s/lps_results.csv" % dir, lps_timecourses, delimiter=",")
        # np.savetxt("%s/t_eval.csv" % dir, t_eval, delimiter=",")

        # # Plot inputs
        # cpg_stim_data = cpg_results[0][2]
        # lps_stim_data = lps_results[0][2]

        input_t = np.linspace(0, stim_time+1, stim_time+1+1)
        # plot_inputs(input_t, cpg_stim_data[0], cpg_stim_data[1], cpg_stim_data[2], "CpG", dir)
        # plot_inputs(input_t, lps_stim_data[0], lps_stim_data[1], lps_stim_data[2], "LPS", dir)

        # Plot inputs unscaled
        I_curve_cpg = read_inputs("IRF", "CpG", scale=False)
        N_curve_cpg = read_inputs("NFkB", "CpG", scale=False)
        I_curve_lps = read_inputs("IRF", "LPS", scale=False)
        N_curve_lps = read_inputs("NFkB", "LPS", scale=False)

        # plot_inputs(input_t, N_curve_cpg, I_curve_cpg, cpg_stim_data[2], "CpG_nM", dir, ylimits=False)
        # plot_inputs(input_t, N_curve_lps, I_curve_lps, lps_stim_data[2], "LPS_nM", dir, ylimits=False)

        # Plot nfkb and irf timecourses
        fig, ax = plt.subplots(1,2, figsize=(8,4))
        ax[0].plot(I_curve_cpg[0], I_curve_cpg[1], label="CpG", marker="o", linewidth=0.5)
        ax[0].plot(I_curve_lps[0], I_curve_lps[1], label="LPS", marker="o", linewidth=0.5)
        ax[0].set_xlabel("Time (min)")
        ax[0].set_ylabel("IRF (nM)")
        ax[0].set_title("IRF timecourses")
        ax[0].legend()
        
        ax[1].plot(N_curve_cpg[0], N_curve_cpg[1], label="CpG", marker="o", linewidth=0.5)
        ax[1].plot(N_curve_lps[0], N_curve_lps[1], label="LPS", marker="o", linewidth=0.5)
        ax[1].set_xlabel("Time (min)")
        ax[1].set_ylabel("NFkB (nM)")
        ax[1].set_title("NFkB timecourses")
        ax[1].legend()
        plt.savefig("%s/nfkb_irf_timecourses.png" % dir)

        # Load results
        cpg_timecourses = np.loadtxt("%s/cpg_results.csv" % dir, delimiter=",")
        lps_timecourses = np.loadtxt("%s/lps_results.csv" % dir, delimiter=",")
        t_eval = np.loadtxt("%s/t_eval.csv" % dir, delimiter=",")

        # Plot results
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in range(num_par_reps):
            ax.plot(t_eval, lps_timecourses[i,:], color="b", alpha=0.1)
        ax.set_xlabel("Time (min)")
        ax.set_ylabel(r"IFN$\beta$ mRNA")
        ax.set_title("LPS stimulation, all parameter sets")
        plt.savefig("%s/lps_timecourses.png" % dir)
        ylim = ax.get_ylim()
        plt.close()


        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in range(num_par_reps):
            ax.plot(t_eval, cpg_timecourses[i,:], color="b", alpha=0.1)
        ax.set_xlabel("Time (min)")
        ax.set_ylabel(r"IFN$\beta$ mRNA")
        ax.set_title("CpG stimulation, all parameter sets")
        ax.set_ylim(ylim)
        plt.savefig("%s/cpg_timecourses.png" % dir)
        plt.close()



        # # PLot initial IFNb mRNA for each parameter set
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.plot(cpg_timecourses[:,0], "bo", label="CpG")
        # ax.plot(lps_timecourses[:,0], "ro", label="LPS")
        # ax.set_xlabel("Parameter set")
        # ax.set_ylabel(r"IFN$\beta$ mRNA (initial)")
        # ax.set_title(r"Initial IFN$\beta$ mRNA for all parameter sets")
        # plt.legend()
        # plt.savefig("%s/initial_mRNA_by_par.png" % dir)
        # plt.close()   

        # Calculate fold change between CpG at 1h and LPs at 1h
        t_eval_closest = np.argmin(np.abs(t_eval - 60))
        cpg_1h = cpg_timecourses[:,t_eval_closest]
        lps_1h = lps_timecourses[:,t_eval_closest]

        cpg_lps_1h_fc = lps_1h / cpg_1h

        # Plot 1h fold change
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(cpg_lps_1h_fc, "bo")
        ax.set_xlabel("Parameter set")
        ax.set_ylabel(r"IFN$\beta$ mRNA fold change")
        ax.set_title(r"Fold change in IFN$\beta$ mRNA at 1h between CpG and LPS")
        plt.savefig("%s/1h_fold_change_by_par.png" % dir)
        plt.close()

        # mRNA data
        mRNA_data = pd.read_csv("../data/ifnb_mRNA_counts_cheng17.csv")
        cpg_1h_data = mRNA_data[(mRNA_data["stimulus"] == "CPG") & (mRNA_data["time"] == 1)]
        cpg_1h_data = cpg_1h_data["counts"].values[0]
        lps_1h_data = mRNA_data[(mRNA_data["stimulus"] == "LPS") & (mRNA_data["time"] == 1)]
        lps_1h_data = lps_1h_data["counts"].values[0]

        cpg_lps_1h_fc_data = lps_1h_data / cpg_1h_data

        print("CpG mRNA at 1h: %.2f" % cpg_1h_data)
        print("LPS mRNA at 1h: %.2f" % lps_1h_data)
        print("Fold change CpG to LPS at 1h: %.2f" % cpg_lps_1h_fc_data)

        fc_closest = np.argmin(np.abs(cpg_lps_1h_fc - cpg_lps_1h_fc_data))
        fc_diff = cpg_lps_1h_fc[fc_closest] - cpg_lps_1h_fc_data
        fc_diffs.append(fc_diff)

        print("Parameter set closest to data: %s" % params[fc_closest,:])
        print("Fold change CpG to LPS simulation: %.3f" % cpg_lps_1h_fc[fc_closest])

        # plot results
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in range(num_par_reps):
            if i == fc_closest:
                ax.plot(t_eval, lps_timecourses[i,:], color="r", alpha=1)
            else:
                ax.plot(t_eval, lps_timecourses[i,:], color="b", alpha=0.1)
        ax.set_xlabel("Time (min)")
        ax.set_ylabel(r"IFN$\beta$ mRNA")
        ax.set_title("LPS stimulation, parameter set closest to data")
        plt.savefig("%s/lps_timecourses_closest.png" % dir)
        ylim = ax.get_ylim()
        plt.close()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in range(num_par_reps):
            if i == fc_closest:
                ax.plot(t_eval, cpg_timecourses[i,:], color="r", alpha=1)
            else:
                ax.plot(t_eval, cpg_timecourses[i,:], color="b", alpha=0.1)
        ax.set_xlabel("Time (min)")
        ax.set_ylabel(r"IFN$\beta$ mRNA")
        ax.set_title("CpG stimulation, parameter set closest to data")
        ax.set_ylim(ylim)
        plt.savefig("%s/cpg_timecourses_closest.png" % dir)
        plt.close()


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
        ax.plot(cpg_fc_sort, "bo", label="CpG")
        ax.plot(lps_fc_sort, "ro", label="LPS")
        ax.set_xlabel("Parameter set (sorted)")
        ax.set_ylabel("Fold change")
        ax.set_title(r"Fold change in IFN$\beta$ mRNA")
        plt.legend()
        plt.savefig("%s/fold_change_by_par.png" % dir)
        plt.close()


    print("Fold change differences:")
    for deg_rate, diff in zip(degredation_rates, fc_diffs):
        print("Degradation rate %.5f: difference of %.3f" % (deg_rate, diff))

if __name__ == "__main__":
    main()