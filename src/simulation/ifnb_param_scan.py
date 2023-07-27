from simulate_isg_expression import *
import os

results_dir = "./results/ifnb_parameter_scan/"
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


def initialize_simulation():
    t = [0,500]
    states0 = [0, 50, 0, 50, 0, 10]

    N, I, P = 1,1,0.01
    N_times, I_times, P_times = [0, 60], [0, 60], [0, 60]
    stim_data = get_stim_data(N, I, P, N_times, I_times, P_times)

    ifnb_params = get_params("../p50_model/results/random_opt/ifnb_best_params_random_global.csv")
    ifnar_params = get_params("ifnar_params.csv")
    params = {**ifnb_params, **ifnar_params}
    return t, states0, params, stim_data

t, states0, params, stim_data = initialize_simulation()


# # Example simulation
# isg_syn = 0.5
# isg_deg = 0.5
# ifnb_syn = 50

# states_ss, t_ss = run_steady_state(t, states0, params, isg_syn, isg_deg, ifnb_syn,
#                                     get_stim_data(0.01,0.01,0.01, "All", "All", "All"))
# states_labels = [r"IFN$\beta$", "IFNAR inactive", "IFNAR active", "ISGF3 inactive", "ISGF3 active", "ISG mRNA"]
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
isg_syn = 0.5
isg_deg = 0.5
ifnb_syn_list = np.arange(15, 100, 1)
states_labels = [r"IFN$\beta$", "IFNAR inactive", "IFNAR active", "ISGF3 inactive", "ISGF3 active", "ISG mRNA"]

best_ifnb = [0, 0, 0]
acceptable_ifnb = {}
all_ifnb = {}
for ifnb_syn in ifnb_syn_list:
    print("Finding steady state for IFNb synthesis rate of %.4f" % ifnb_syn)
    states_ss, t_ss = run_steady_state(t, states0, params, isg_syn, isg_deg, ifnb_syn,
                                        get_stim_data(0.01,0.01,0.01, "All", "All", "All"))
    states = run_simulation(t, states_ss, params, isg_syn, isg_deg, ifnb_syn, None, stim_data)
    ifnb_max = np.max(states.y[0,:])
    isgf3_max = np.max(states.y[4,:])
    isgf3_ss = states_ss[4]
    all_ifnb[ifnb_syn] = [ifnb_max, isgf3_max]
    if ifnb_max > 50 and ifnb_max < 100:
        acceptable_ifnb[ifnb_syn] = [ifnb_max, isgf3_max]
        if isgf3_max > best_ifnb[2]:
            best_ifnb = [ifnb_syn, ifnb_max, isgf3_max]
            print("IFNb synthesis rate of %.4f is good: max IFNb of %.4f and max ISGF3 of %.4f" % (ifnb_syn, ifnb_max, isgf3_max))
        
        # states_list = [states.y[i,:] for i in range(states.y.shape[0])]
        # plot_model(states_list, states_labels, states.t, "%sisg_model_ifnb_syn_%.4f" % (results_dir, ifnb_syn),
        #              "IFNb synthesis rate = %.4f" % ifnb_syn, "Time (min)", "Concentration (nM)")
    else:
        print("\n\n")

states_ss, t_ss = run_steady_state(t, states0, params, isg_syn, isg_deg, best_ifnb[0],
                                    get_stim_data(0.01,0.01,0.01, "All", "All", "All"))
states = run_simulation(t, states_ss, params, isg_syn, isg_deg, best_ifnb[0], None, stim_data)
states_list = [states.y[i,:] for i in range(states.y.shape[0])]
plot_model(states_list, states_labels, states.t, "%sisg_model_best_ifnb_syn" % results_dir,
                "IFNb synthesis rate = %.4f" % best_ifnb[0], "Time (min)", "Concentration (nM)")

np.save("%sisg_model_best_ifnb_syn" % results_dir, best_ifnb[0])
np.save("%sacceptable_ifnb_syn" % results_dir, acceptable_ifnb)
np.save("%sall_ifnb_syn" % results_dir, all_ifnb)

ifnb_syn_list = list(all_ifnb.keys())
ifnb_max_list = [all_ifnb[ifnb_syn][0] for ifnb_syn in ifnb_syn_list]
isgf3_max_list = [all_ifnb[ifnb_syn][1] for ifnb_syn in ifnb_syn_list]
fig = plt.figure()
plt.plot(ifnb_syn_list, ifnb_max_list, "o", label=r"IFN$\beta$")
plt.plot(ifnb_syn_list, isgf3_max_list, "o", label="ISGF3")
plt.xlabel(r"IFN$\beta$ synthesis rate")
plt.ylabel("Max concentration")
fig.legend(bbox_to_anchor=(1.2,0.5))
plt.savefig("%sisg_model_max_conc_vs_ifnb_syn" % results_dir, bbox_inches="tight")

