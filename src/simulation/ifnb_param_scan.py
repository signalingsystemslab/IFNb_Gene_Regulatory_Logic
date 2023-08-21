from simulate_isg_expression import *
import os
import time
import scipy.optimize as opt
from multiprocessing import Pool


results_dir = "./results/ifnb_parameter_scan/"
os.makedirs(results_dir, exist_ok=True)

# States: IFNb, IFNAR, IFNAR*, ISGF3, ISGF3*, ISG mRNA

def run_steady_state(t, states0, params, isg_syn, isg_deg, stim_data=None):
    params["p_syn_isg"] = isg_syn
    params["p_deg_isg"] = isg_deg
    states_ss, t_ss = get_steady_state(t, states0, params, stim_data)
    return states_ss, t_ss

def run_simulation(t, states0, params, isg_syn, isg_deg, t_eval, stim_data=None):
    params["p_syn_isg"] = isg_syn
    params["p_deg_isg"] = isg_deg
    states = run_model(t, states0, params, t_eval, stim_data)
    return states

def run_simulation_wrapper(t, states0, params, isg_syn, isg_deg, t_eval, stim_data):
    ss_stim_data = [[0.01 for t in t_eval] for i in range(3)]
    states_ss, t_ss = run_steady_state(t, states0, params, isg_syn, isg_deg, ss_stim_data)
    states = run_simulation(t, states_ss, params, isg_syn, isg_deg, t_eval, stim_data)
    return states

def initialize_simulation(N = 1, P = 1, I = 1, genotype="WT", stimulus="CpG", stim_time = 60*8):
    t = [0, stim_time+60]
    states0 = [0.01, 100, 0.01, 100, 0.01, 10]

    P_values = {"WT": P, "p50KO": 0}
    N_values = {"WT": N, "p50KO": N/2}
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
    N, P, I, stimulus, stim_time, ifnb_lims, isg_syn, isg_deg = args
    ifnb_synthesis = pars[0]

    # print("Calculating objective function value for IFNb synthesis = %.4f" % ifnb_synthesis)
    t, states0, params, stim_data = initialize_simulation(N, P, I, "WT", stimulus, stim_time)

    # print("Done initializing")
    params["p_syn_ifnb"] = ifnb_synthesis
    # print(params)
    t_eval = np.linspace(0, stim_time+60, 1000)
    # print("Running simulation")
    states_WT = run_simulation_wrapper(t, states0, params, isg_syn, isg_deg, t_eval, stim_data)
    # print("Done with WT")


    t, states0, params, stim_data = initialize_simulation(N, P, I, "p50KO", stimulus, stim_time)
    params["p_syn_ifnb"] = ifnb_synthesis
    states_p50KO = run_simulation_wrapper(t, states0, params, isg_syn, isg_deg, t_eval, stim_data)
    # print("Done with p50KO")

    # Objective function components:
    # 1. IFNb peak between 50 and 100
    # 2. Maximize IFNb difference between WT and p50KO

    if np.max(states_WT.y[0,:]) > 0:
        ifnb_difference = np.max(states_p50KO.y[0,:]) / np.max(states_WT.y[0,:])
    else:
        ifnb_difference = 0
    ifnb_peak = np.max(states_WT.y[0,:])

    if ifnb_difference > 0:
        obj = 1/(ifnb_difference + 1)
        print("IFNb synthesis = %.4f\tObjective function value = %.4f" % (ifnb_synthesis, obj))
    else:
        obj = 1
    if ifnb_peak < ifnb_lims[0] or ifnb_peak > ifnb_lims[1]:
        obj = 1
    return obj

def optimize_model(min_val, max_val, N = 1, P = 1, I = 1, stimulus="CpG", stim_time = 60*8, ifnb_lims = [50, 100], 
                   isgf3_lims = [50, 100], isg_syn = 0.01, isg_deg = 0.5):
    print("Optimizing %s IFNb synthesis" % stimulus)
    start = time.time()
    rgs = slice(min_val, max_val, 0.01)
    rgs = tuple([rgs])
    res = opt.brute(objective_function, rgs, args=(N, P, I, stimulus, stim_time, ifnb_lims, isg_syn, isg_deg),
                     full_output=True, finish=None, workers=40)
    print("Optimization took %.2f seconds" % (time.time() - start))
    return res[0], res[1], res[3]

# Optimize IFNb activation and deactivation
print("\n\n\n##############################################\n\n\n")
min_val = 0.01
max_val = 10
N = 1
P = 1
I = 1
stimulus = "CpG"
stim_time = 60*8
ifnb_lims = [50, 100]
isgf3_lims = [50, 100]
isg_syn = 0.01
isg_deg = 0.5

# Example simulation
ifnb_synthesis = 0.1
t, states0, params, stim_data = initialize_simulation(N, P, I, "WT", stimulus, stim_time)
params["p_syn_ifnb"] = ifnb_synthesis
t_eval = np.linspace(0, stim_time+60, 1000)
states_WT = run_simulation_wrapper(t, states0, params, isg_syn, isg_deg, t_eval, stim_data)
print("Example simulation: IFNb synthesis = %.4f worked\n" % ifnb_synthesis)

# Optimize
pars, rho, jout = optimize_model(min_val, max_val, N, P, I, stimulus, stim_time, ifnb_lims, isgf3_lims, isg_syn, isg_deg)
print("Best IFNb synthesis: %.4f" % pars)
print("Best objective function value: %.4f, or %.4f fold change in IFNb peak" % (rho, 1/rho - 1))
np.save("%s/%s_best_ifnb_synthesis.npy" % (results_dir, stimulus), pars)
np.save("%s/%s_best_ifnb_synthesis_obj.npy" % (results_dir, stimulus), rho)
np.save("%s/%s_best_ifnb_synthesis_jout.npy" % (results_dir, stimulus), jout)
