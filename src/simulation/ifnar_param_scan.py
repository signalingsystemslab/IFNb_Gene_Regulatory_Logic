from simulate_isg_expression import *
import os
import time
import scipy.optimize as opt
from multiprocessing import Pool


results_dir = "./results/ifnar_parameter_scan/"
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
    return states

def run_simulation_wrapper(t, states0, params, isg_syn, isg_deg, t_eval, stim_data):
    ss_stim_data = [[0.0001 for t in t_eval] for i in range(3)]
    states_ss, t_ss, all_t_ss, all_states_ss = run_steady_state(t, states0, params, isg_syn, isg_deg, ss_stim_data)
    states = run_simulation(t, states_ss, params, isg_syn, isg_deg, t_eval, stim_data)
    return states, all_t_ss, all_states_ss 


def initialize_simulation(N = 1, P = 1, I = 1, genotype="WT", stimulus="CpG", stim_time = 60*8):
    t = [0, stim_time+60]
    states0 = [0.01, 100, 0.01, 100, 0.01, 10]

    P_values = {"WT": P, "p50KO": 0}
    N_values = {"WT": N, "p50KO": N}
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
    N, P, I, stimulus, stim_time, ifnb_lims, isgf3_lims, isg_syn, isg_deg, ifnb_syn = args
    ifnb_ifnar_activation, ifnar_activation, ifnar_deactivation = pars
    t, states0, params, stim_data = initialize_simulation(N, P, I, "WT", stimulus, stim_time)
    params["p_act_ifnar_ifnb"] = ifnb_ifnar_activation
    params["p_act_ifnar_in"] = ifnar_activation
    params["p_deact_ifnar"] = ifnar_deactivation
    params["p_syn_ifnb"] = ifnb_syn
    t_eval = np.linspace(0, stim_time+60, 1000)
    states_WT, tmp1, tmp2 = run_simulation_wrapper(t, states0, params, isg_syn, isg_deg, t_eval, stim_data)

    t, states0, params, stim_data = initialize_simulation(N, P, I, "p50KO", stimulus, stim_time)
    params["p_act_ifnar_ifnb"] = ifnb_ifnar_activation
    params["p_act_ifnar_in"] = ifnar_activation
    params["p_deact_ifnar"] = ifnar_deactivation
    params["p_syn_ifnb"] = ifnb_syn
    states_p50KO, tmp1, tmp2 = run_simulation_wrapper(t, states0, params, isg_syn, isg_deg, t_eval, stim_data)
 
    # Objective function components:
    # 1. IFNb peak between 50 and 100
    # 2. ISGF3 peak between 50 and 100
    # 3. Maximize difference between ISGF3* with WT and p50KO

    if np.max(states_WT.y[0,:]) > 0:
        isgf3_difference = np.max(states_p50KO.y[4,:]) / np.max(states_WT.y[4,:])
    else:
        isgf3_difference = 0
    ifnb_peak = np.max(states_WT.y[0,:])
    isgf3_peak = np.max(states_WT.y[4,:])

    if isgf3_difference > 0:
        obj = 1/(isgf3_difference + 1)
    else:
        obj = 1
    if ifnb_peak < ifnb_lims[0] or ifnb_peak > ifnb_lims[1]:
        print("IFNb peak outside of limits: %.4f" % ifnb_peak)
        obj = 1
    if isgf3_peak < isgf3_lims[0] or isgf3_peak > isgf3_lims[1]:
        print("ISGF3 peak outside of limits: %.4f" % isgf3_peak)
        obj = 1
    if obj != 1:
        print("IFNb synthesis = %.4f\tIFNAR activation = %.4f\tIFNAR deactivation = %.4f\tObjective function value = %.4f" % (ifnb_syn, ifnb_ifnar_activation, ifnar_deactivation, obj))
    return obj



def optimize_model(min_val, max_val, N = 1, P = 1, I = 1, stimulus="CpG", stim_time = 60*8, ifnb_lims = [50, 100], 
                   isgf3_lims = [50, 100], isg_syn = 0.01, isg_deg = 0.5, ifnb_syn = 0.5):
    print("Optimizing %s" % stimulus)
    start = time.time()
    rgs = slice(min_val, max_val, 0.05)
    rgs = tuple([rgs for i in range(3)])
    res = opt.brute(objective_function, rgs, args=(N, P, I, stimulus, stim_time, ifnb_lims, isgf3_lims, isg_syn, isg_deg, ifnb_syn),
                     full_output=True, finish=True, workers=40)
    print("Optimization took %.2f minutes" % ((time.time() - start)/60))
    return res

states_labels = [r"IFN$\beta$", "IFNAR inactive", "IFNAR active", "ISGF3 inactive", "ISGF3 active", "ISG mRNA"]

# Optimize IFNAR parameters
print("\n\n\n##############################################\n\n\n")

min_val = 0.05
max_val = 0.5
N = 0.25
P = 1
I = 0.1
stimulus = "CpG"
stim_time = 60*8
ifnb_lims = [1, 30]
isgf3_lims = [0, 100]
isg_syn = 0.01
isg_deg = 0.5
ifnb_syn = np.load("%s/%s_N-%s_best_ifnb_synthesis.npy" % ("results/ifnb_parameter_scan/different_N/", stimulus, N))
print("IFNb synthesis: %.4f for N = %.2f" % (ifnb_syn, N))

print("Optimizing IFNAR activation and deactivation")
res = optimize_model(min_val, max_val, N, P, I, stimulus, stim_time, ifnb_lims, isgf3_lims, isg_syn, isg_deg, ifnb_syn)
np.save("%s/%s_best_ifnar_activation.npy" % (results_dir, stimulus), res[0])
np.save("%s/%s_best_ifnar_activation_obj.npy" % (results_dir, stimulus), res[1])
np.save("%s/%s_best_ifnar_activation_jout.npy" % (results_dir, stimulus), res[3])

print("Best IFNAR activation: %.4f, basal activation: %.4f, deactivation: %.4f" % (res[0][0], res[0][1], res[0][2]))
print("Best objective function value: %.4f, or %.4f fold change in ISGF3 peak" % (res[1], 1/res[1] - 1))


# Plot results
print("Plotting results")
pars = np.load("%s/%s_best_ifnar_activation.npy" % (results_dir, stimulus))

t, states0_WT, params, stim_data = initialize_simulation(N, P, I, "WT", stimulus, stim_time)
# print("Max P value for WT: %.4f" % np.max(stim_data[2]))
params["p_act_ifnar_ifnb"] = pars[0]
params["p_act_ifnar_in"] = pars[1]
params["p_deact_ifnar"] = pars[2]
params["p_syn_ifnb"] = ifnb_syn
t_eval = np.linspace(0, stim_time+60, 1000)
states_WT, WT_t_ss, WT_ss = run_simulation_wrapper(t, states0_WT, params, isg_syn, isg_deg, t_eval, stim_data)
print("Max IFNb value for WT: %.4f" % np.max(states_WT.y[0,:]))

t, states0_KO, params, stim_data = initialize_simulation(N, P, I, "p50KO", stimulus, stim_time)
# print("Max P value for p50KO: %.4f" % np.max(stim_data[2]))
params["p_act_ifnar_ifnb"] = pars[0]
params["p_act_ifnar_in"] = pars[1]
params["p_deact_ifnar"] = pars[2]
params["p_syn_ifnb"] = ifnb_syn
states_p50KO, p50KO_t_ss, p50KO_ss = run_simulation_wrapper(t, states0_KO, params, isg_syn, isg_deg, t_eval, stim_data)
print("Max IFNb value for p50KO: %.4f" % np.max(states_p50KO.y[0,:]))

print(WT_ss[:,0])

# Plot steady state values
fig, ax = plt.subplots(np.ceil(len(states0_WT)/3).astype(int), 3, figsize=(12, 8))
for i in range(len(states0_WT)):
    ax[i//3, i%3].plot(WT_t_ss, WT_ss[i,:], label="WT" if i == 0 else "", color="black")
    ax[i//3, i%3].plot(p50KO_t_ss, p50KO_ss[i,:], label="p50KO" if i == 0 else "", color="red")
    ax[i//3, i%3].set_title(states_labels[i])
fig.legend(bbox_to_anchor=(1.1,0.5))
plt.tight_layout()
plt.savefig("%s/%s_best_ifnar_steady_state.png" % (results_dir, stimulus))

# Plot IFNb only
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.plot(t_eval, states_WT.y[0,:], label="WT", color="black")
ax.plot(t_eval, states_p50KO.y[0,:], label="p50KO", color="red")
ax.set_xlabel("Time (min)")
ax.set_ylabel(r"IFN$\beta$ (nM)")
fig.legend(bbox_to_anchor=(1.2,0.5))
plt.tight_layout()
plt.savefig("%s/%s_best_ifnar_timecourse_ifnb.png" % (results_dir, stimulus))

# Plot all
fig, ax = plt.subplots(2, 3, figsize=(12, 8))
for i in range(6):
    ax[i//3, i%3].plot(t_eval, states_WT.y[i,:], label="WT" if i == 0 else "", color="black")
    ax[i//3, i%3].plot(t_eval, states_p50KO.y[i,:], label="p50KO" if i == 0 else "", color="red")
    ax[i//3, i%3].set_title(states_labels[i])
    ax[i//3, i%3].set_xlabel("Time (min)")
    ax[i//3, i%3].set_ylabel("Concentration (nM)")
    # ax[i//3, i%3].legend()
fig.legend(bbox_to_anchor=(1.1,0.5))
plt.tight_layout()
plt.savefig("%s/%s_best_ifnar_timecourse.png" % (results_dir, stimulus))