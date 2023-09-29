from simulate_isg_expression import *
import os
import time
import scipy.optimize as opt
from multiprocessing import Pool


results_dir = "./results/ifnb_parameter_scan/different_N/"
os.makedirs(results_dir, exist_ok=True)

# States: IFNb, IFNAR, IFNAR*, ISGF3, ISGF3*, ISG mRNA

def run_steady_state(t, states0, params, isg_syn, isg_deg, stim_data=None):
    params["p_syn_isg"] = isg_syn
    params["p_deg_isg"] = isg_deg
    states_ss, t_ss, all_t_ss, all_states_ss = get_steady_state(t, states0, params, stim_data)
    return states_ss, t_ss, all_t_ss, all_states_ss

def run_simulation(t, states0, params, isg_syn, isg_deg, t_eval, stim_data=None):
    params["p_syn_isg"] = isg_syn
    params["p_deg_isg"] = isg_deg
    states = run_model(t, states0, params, t_eval, stim_data)
    # print("Params after simulation:")
    # for k, v in params.items():
    #     print("%s: %.4f" % (k, v))
    return states

def run_simulation_wrapper(t, states0, params, isg_syn, isg_deg, t_eval, stim_data):
    ss_stim_data = [[0.0001 for t in t_eval] for i in range(2)]
    ss_stim_data.append(stim_data[2])
    states_ss, t_ss, all_t_ss, all_states_ss = run_steady_state(t, states0, params, isg_syn, isg_deg, ss_stim_data)
    # print("Steady state values: %s" % states_ss) 
    states = run_simulation(t, states_ss, params, isg_syn, isg_deg, t_eval, stim_data)
    # print("Final values: %s" % states.y[:,-1])
    return states

def initialize_simulation(N = 1, P = 1, I = 1, N_scale=1, genotype="WT", stimulus="CpG", stim_time = 60*8):
    t = [0, stim_time+60]
    states0 = [0.01, 100, 0.01, 100, 0.01, 10]

    P_values = {"WT": P, "p50KO": 0}
    N_values = {"WT": N, "p50KO": N_scale*N}
    P_curve = [P_values[genotype] for i in range(stim_time+60)]

    cell_traj = np.loadtxt("RepresentativeCellTraj_NFkBn_%s.csv" % stimulus, delimiter=",")
    N_curve = cell_traj[1,:]
    # print(N_values[genotype], np.max(N_curve))
    N_curve = N_values[genotype]*N_curve/np.max(N_curve)
    I_curve = [I for i in range(stim_time+60)]

    stim_data = [N_curve, I_curve, P_curve]

    ifnb_params = get_params("../p50_model/results/random_opt/ifnb_best_params_random_global.csv")
    ifnar_params = get_params("ifnar_params.csv")
    params = {**ifnb_params, **ifnar_params}
    return t, states0, params, stim_data

def objective_function(pars, *args):
    N, P, I, N_scale, stimulus, stim_time, ifnb_lims, isg_syn, isg_deg = args
    ifnb_synthesis = pars[0]

    # print("Calculating objective function value for IFNb synthesis = %.4f" % ifnb_synthesis)
    t, states0, params, stim_data = initialize_simulation(N, P, I, N_scale, "WT", stimulus, stim_time)

    # print("Done initializing")
    params["p_syn_ifnb"] = ifnb_synthesis
    # print(params)
    t_eval = np.linspace(0, stim_time+60, 1000)
    # print("Running simulation")
    states_WT = run_simulation_wrapper(t, states0, params, isg_syn, isg_deg, t_eval, stim_data)
    # print("Done with WT")


    t, states0, params, stim_data = initialize_simulation(N, P, I, N_scale, "p50KO", stimulus, stim_time)
    params["p_syn_ifnb"] = ifnb_synthesis
    states_p50KO = run_simulation_wrapper(t, states0, params, isg_syn, isg_deg, t_eval, stim_data)
    # print("Done with p50KO")

    # Objective function components:
    # 1. WT IFNb peak between 50 and 100
    # 2. Maximize IFNb difference between WT and p50KO

    if np.max(states_WT.y[0,:]) > 0:
        ifnb_difference = np.max(states_p50KO.y[0,:]) / np.max(states_WT.y[0,:])
    else:
        ifnb_difference = 1
    ifnb_peak = np.max(states_WT.y[0,:])

    if ifnb_difference > 1:
        obj = 1/ifnb_difference
    else:
        obj = 1
    if ifnb_peak < ifnb_lims[0] or ifnb_peak > ifnb_lims[1]:
        print("IFNb peak outside of limits: %.4f" % ifnb_peak)
        obj = 1
    print("IFNb synthesis = %.4f\tObjective function value = %.4f\n WT IFNb peak = %.4f, KO IFNb peak = %.4f" % (ifnb_synthesis, obj, ifnb_peak, np.max(states_p50KO.y[0,:])))
    return obj

def optimize_model(min_val, max_val, N = 1, P = 1, I = 1, N_scale=0.5, stimulus="CpG", stim_time = 60*8, ifnb_lims = [50, 100], 
                   isgf3_lims = [50, 100], isg_syn = 0.01, isg_deg = 0.5):
    print("Optimizing %s IFNb synthesis" % stimulus)
    start = time.time()
    rgs = slice(min_val, max_val, 0.01)
    rgs = tuple([rgs])
    res = opt.brute(objective_function, rgs, args=(N, P, I, N_scale, stimulus, stim_time, ifnb_lims, isg_syn, isg_deg),
                     full_output=True, finish=None, workers=40)
    print("Optimization took %.2f seconds" % (time.time() - start))
    return res[0], res[1], res[3]

# Optimize IFNb activation and deactivation
print("\n\n\n##############################################\n\n\n")
min_val = 0.01
max_val = 1
N_list = [0.1, 0.25, 0.4, 0.5, 1]
# N_list = [0.25]
P = 1
I = 0.1
N_scale = 1
stimulus = "CpG"
stim_time = 60*8
ifnb_lims = [1, 30]
isgf3_lims = [0, 100]
isg_syn = 0.01
isg_deg = 0.5

states_labels = [r"IFN$\beta$", "IFNAR inactive", "IFNAR active", "ISGF3 inactive", "ISGF3 active", "ISG mRNA"]

for N in N_list:
    print("\n\n\n##############################################\n\n\n")
    print("N = %.2f\n\n" % N)
    # Plot input curves
    t, states0, params, stim_data_WT = initialize_simulation(N, P, I, N_scale, "WT", stimulus, stim_time)
    t, states0, params, stim_data_KO = initialize_simulation(N, P, I, N_scale, "p50KO", stimulus, stim_time)

    fig, ax = plt.subplots(3, 1, figsize=(8, 6))
    for i in range(3):
        ax[i].plot(stim_data_WT[i], label="WT" if i == 0 else "", color="black")
        ax[i].plot(stim_data_KO[i], label="p50KO" if i == 0 else "", color="red")
        # ax[i].legend()
        ax[i].set_ylabel("Concentration (nM)")
        ax[i].set_title([r"NF$\kappa$B", "IRF", "p50"][i])
    ax[2].set_xlabel("Time (min)")
    plt.tight_layout()
    fig.legend(bbox_to_anchor=(1.2,0.5))
    plt.savefig("%s/%s_N-%s_input_curves.png" % (results_dir, stimulus, N), bbox_inches="tight")

    # Optimize
    pars, rho, jout = optimize_model(min_val, max_val, N, P, I, N_scale, stimulus, stim_time, ifnb_lims, isgf3_lims, isg_syn, isg_deg)
    print("Best IFNb synthesis for N = %.2f: %.4f" % (N, pars))
    print("Best objective function value: %.4f, or %.4f fold change in IFNb peak" % (rho, 1/rho))
    np.save("%s/%s_N-%s_best_ifnb_synthesis.npy" % (results_dir, stimulus, N), pars)
    np.save("%s/%s_N-%s_best_ifnb_synthesis_obj.npy" % (results_dir, stimulus, N), rho)
    np.save("%s/%s_N-%s_best_ifnb_synthesis_jout.npy" % (results_dir, stimulus, N), jout)

    # Plot results
    pars = np.load("%s/%s_N-%s_best_ifnb_synthesis.npy" % (results_dir, stimulus, N))
    t, states0, params, stim_data = initialize_simulation(N, P, I, N_scale, "WT", stimulus, stim_time)
    params["p_syn_ifnb"] = pars
    t_eval = np.linspace(0, stim_time+60, 1000)
    states_WT = run_simulation_wrapper(t, states0, params, isg_syn, isg_deg, t_eval, stim_data)

    t, states0, params, stim_data = initialize_simulation(N, P, I, N_scale, "p50KO", stimulus, stim_time)
    params["p_syn_ifnb"] = pars
    states_p50KO = run_simulation_wrapper(t, states0, params, isg_syn, isg_deg, t_eval, stim_data)

    # Calculate fold change in IFNb over 0h for WT and p50KO
    ifnb_WT = states_WT.y[0,:]
    ifnb_p50KO = states_p50KO.y[0,:]
    ifnb_WT_0h = ifnb_WT[0]
    ifnb_p50KO_0h = ifnb_p50KO[0]
    ifnb_WT_fold_change = ifnb_WT/ifnb_WT_0h
    ifnb_p50KO_fold_change = ifnb_p50KO/ifnb_p50KO_0h
    print("Max IFNb fold change for WT: %.4f" % np.max(ifnb_WT_fold_change))
    print("Max IFNb fold change for p50KO: %.4f" % np.max(ifnb_p50KO_fold_change))
    print("Fold change difference: %.4f" % (np.max(ifnb_p50KO_fold_change)/np.max(ifnb_WT_fold_change)))

    # Plot IFNb
    plt.figure(figsize=(8,6))
    plt.plot(t_eval, states_WT.y[0,:], label="WT", color="black")
    plt.plot(t_eval, states_p50KO.y[0,:], label="p50KO", color="red")
    plt.xlabel("Time (min)")
    plt.ylabel(r"IFN$\beta$ (nM)")
    plt.legend()
    plt.savefig("%s/%s_N-%s_best_ifnb_synthesis_timecourse.png" % (results_dir, stimulus, N), bbox_inches="tight")

    # Plot IFNb fold change over 0h
    plt.figure(figsize=(8,6))
    plt.plot(t_eval, ifnb_WT_fold_change, label="WT", color="black")
    plt.plot(t_eval, ifnb_p50KO_fold_change, label="p50KO", color="red")
    plt.xlabel("Time (min)")
    plt.ylabel(r"IFN$\beta$ fold change over 0h")
    plt.legend()
    plt.savefig("%s/%s_N-%s_best_ifnb_synthesis_fold_change.png" % (results_dir, stimulus, N), bbox_inches="tight")

    # Plot all
    fig, ax = plt.subplots(2, 3, figsize=(12, 8))
    for i in range(6):
        ax[i//3, i%3].plot(t_eval, states_WT.y[i,:], label="WT", color="black")
        ax[i//3, i%3].plot(t_eval, states_p50KO.y[i,:], label="p50KO", color="red")
        ax[i//3, i%3].set_title(states_labels[i])
        ax[i//3, i%3].legend()
    plt.tight_layout()
    plt.savefig("%s/%s_N-%s_best_ifnb_synthesis_timecourse_all.png" % (results_dir, stimulus, N), bbox_inches="tight")