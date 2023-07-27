from simulate_isg_expression import *
import os
from multiprocessing import Pool

results_dir = "./results/isg_parameter_scan/"
os.makedirs(results_dir, exist_ok=True)


def get_stim_data(N, I, P, N_times="All", I_times="All", P_times="All"):
    stim_data = [N, N_times, I, I_times, P, P_times]
    return stim_data

def run_steady_state(t, states0, params, isg_syn, isg_deg, ifnb_syn, stim_data=None):
    params["p_syn_isg"] = isg_syn
    params["p_deg_isg"] = isg_deg
    params["p_syn_ifnb"] = ifnb_syn
    states_ss, t_ss = get_steady_state(t, states0, params, stim_data)
    return states_ss, t_ss

def run_simulation(t, states0, params, isg_syn, isg_deg, ifnb_syn, t_eval, stim_data=None):
    params["p_syn_isg"] = isg_syn
    params["p_deg_isg"] = isg_deg
    params["p_syn_ifnb"] = ifnb_syn
    states = run_model(t, states0, params, t_eval, stim_data)
    return states

def run_simulation_wrapper(t, states0, params, isg_syn, isg_deg, ifnb_syn, t_eval, stim_data):
    states_ss, t_ss = run_steady_state(t, states0, params, isg_syn, isg_deg, ifnb_syn, stim_data=get_stim_data(0.01,0.01,0.01, "All", "All", "All"))
    states = run_simulation(t, states_ss, params, isg_syn, isg_deg, ifnb_syn, t_eval, stim_data)
    return states

def initialize_simulation():
    t = [0,500]
    states0 = [0, 100, 0, 100, 0, 10]

    N, I, P = 1,1,0.01
    N_times, I_times, P_times = [0, 60], [0, 60], [0, 60]
    stim_data = get_stim_data(N, I, P, N_times, I_times, P_times)

    ifnb_params = get_params("../p50_model/results/random_opt/ifnb_best_params_random_global.csv")
    ifnar_params = get_params("ifnar_params.csv")
    params = {**ifnb_params, **ifnar_params}
    return t, states0, params, stim_data

t, states0, params, stim_data = initialize_simulation()
states_labels = [r"IFN$\beta$", "IFNAR inactive", "IFNAR active", "ISGF3 inactive", "ISGF3 active", "ISG mRNA"]


# # Example simulation
# isg_syn = 0.5
# isg_deg = 0.5
# ifnb_syn = 50

# states_ss, t_ss = run_steady_state(t, states0, params, isg_syn, isg_deg, ifnb_syn,
#                                     get_stim_data(0.01,0.01,0.01, "All", "All", "All"))
# for state, label in zip(states_ss, states_labels):
#     print("Steady state value of %s: %.4f" % (label, state))

# states = run_simulation(t, states_ss, params, isg_syn, isg_deg, ifnb_syn, None, stim_data)
# print("\n\nMax value of IFNb: %.4f" % np.max(states.y[0,:]))
# print("Max value of ISG mRNA: %.4f" % np.max(states.y[5,:]))
# state_list = [states.y[i,:] for i in range(states.y.shape[0])]
# plot_model(state_list, states_labels, states.t, "%sisg_model_example" % results_dir, 
#            "ISG model example", "Time (min)", "Concentration (nM)")

# # Parameter scan
print("Running parameter scan")
ifnb_syn = 43
isg_syn_list = np.linspace(0, 1, 50)
isg_deg_list = np.linspace(0, 1, 50)
t_eval = np.linspace(t[0], t[1], 1000)

# all_states = np.zeros((len(isg_syn_list), len(isg_deg_list), len(states0), len(t_eval)))
# with Pool(30) as p:
#     for i in range(len(isg_syn_list)):
#         print("Running row %d of %d" % (i+1, len(isg_syn_list)))
#         results = p.starmap(run_simulation_wrapper, 
#                             [(t, states0, params, isg_syn_list[i], isg_deg_list[j], ifnb_syn, t_eval, stim_data) for j in range(len(isg_deg_list))])
#         for j in range(len(isg_deg_list)):
#             all_states[i,j,:,:] = results[j].y

# np.save("%sall_states.npy" % results_dir, all_states)

# all_states = np.load("%sall_states.npy" % results_dir)
# all_isg_syn, all_isg_deg = np.meshgrid(isg_syn_list, isg_deg_list)
# # Make grid of max ISG values corresponding to all_isg_syn and all_isg_deg
# max_isg = np.zeros(all_isg_syn.shape)
# for i in range(all_isg_syn.shape[0]):
#     for j in range(all_isg_syn.shape[1]):
#         max_isg[i,j] = np.max(all_states[i,j,5,:])
# print(all_states[49,0,5,np.arange(0,1000,100)])

# max_max_loc = np.argmax(max_isg)
# print("Max ISG mRNA level: %.4f at ISG synthesis rate = %.4f, ISG degradation rate = %.4f" %
#         (np.max(max_isg), all_isg_syn.flatten()[max_max_loc], all_isg_deg.flatten()[max_max_loc]))
# # states = run_simulation_wrapper(t, states0, params, all_isg_syn.flatten()[max_max_loc], all_isg_deg.flatten()[max_max_loc], ifnb_syn, t_eval, stim_data)
# # state_list = [states.y[i,:] for i in range(states.y.shape[0])]
# # plot_model(state_list, states_labels, states.t, "%sisg_model_max_isg" % results_dir,
# #               "ISG model example", "Time (min)", "Concentration (nM)")
# # print(np.max(states.y[-1,:]))

# # max_isg_log = np.log10(max_isg)
# # Plot max ISG values as contour plot
# fig = plt.figure()
# plt.contourf(all_isg_deg, all_isg_syn, max_isg, 50, cmap="viridis")
# plt.colorbar()
# plt.xlabel("ISG synthesis rate")
# plt.ylabel("ISG degradation rate")
# plt.title("Max ISG mRNA level (log10)")
# plt.savefig("%smax_isg_contour" % results_dir)

# 3 hr stimulation
N, I, P = 1,1,0.01
N_times, I_times, P_times = [0, 120], [0, 120], [0, 120]
stim_data = get_stim_data(N, I, P, N_times, I_times, P_times)

for isg_syn, isg_deg in zip([0.5, 0.5, 0.5], [0.01, 0.1, 0.5]):
    states_ss, t_ss = run_steady_state(t, states0, params, isg_syn, isg_deg, ifnb_syn, stim_data)
    states = run_simulation(t, states_ss, params, isg_syn, isg_deg, ifnb_syn, t_eval, stim_data)
    state_list = [states.y[i,:] for i in range(states.y.shape[0])]
    plot_model(state_list, states_labels, states.t, "%sisg_model_3hr_stim_%.2f-syn_%.2f-deg" % (results_dir, isg_syn, isg_deg),
                "ISG model %.2f ISG synthesis rate, %.2f ISG degradation rate" % (isg_syn, isg_deg), "Time (min)", "Concentration (nM)")


