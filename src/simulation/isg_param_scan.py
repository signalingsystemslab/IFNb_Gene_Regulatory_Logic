from simulate_isg_expression import *
import os
import time
from multiprocessing import Pool


results_dir = "./results/isg_parameter_scan/"
os.makedirs(results_dir, exist_ok=True)

# States: IFNb, IFNAR, IFNAR*, ISGF3, ISGF3*, ISG mRNA

def run_steady_state(t, states0, params, isg_syn, isg_deg, ifnb_syn, stim_data=None):
    params["p_syn_isg"] = isg_syn
    params["p_deg_isg"] = isg_deg
    params["p_syn_ifnb"] = ifnb_syn
    states_ss, t_ss, all_t_ss, all_states_ss = get_steady_state(t, states0, params, stim_data)
    return states_ss, t_ss, all_t_ss, all_states_ss

def run_simulation(t, states0, params, isg_syn, isg_deg, ifnb_syn, t_eval, stim_data=None):
    params["p_syn_isg"] = isg_syn
    params["p_deg_isg"] = isg_deg
    params["p_syn_ifnb"] = ifnb_syn
    states = run_model(t, states0, params, t_eval, stim_data)
    return states

def run_simulation_wrapper(t, states0, params, isg_syn, isg_deg, ifnb_syn, t_eval, stim_data):
    ss_stim_data = [[0.0001 for t in t_eval] for i in range(3)]
    states_ss, t_ss, all_t_ss, all_states_ss = run_steady_state(t, states0, params, isg_syn, isg_deg, ifnb_syn, ss_stim_data)
    states = run_simulation(t, states_ss, params, isg_syn, isg_deg, ifnb_syn, t_eval, stim_data)
    return states

def initialize_simulation(N = 1, P = 1, I = 1, genotype="WT", stimulus="CpG", stim_time = 60):
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
    new_ifnar_pars = np.load("%s/%s_best_ifnar_activation.npy" % ("results/ifnar_parameter_scan/", stimulus))
    ifnar_params["p_act_ifnar_ifnb"] = new_ifnar_pars[0]
    ifnar_params["p_act_ifnar_in"] = new_ifnar_pars[1]
    ifnar_params["p_deact_ifnar"] = new_ifnar_pars[2]

    params = {**ifnb_params, **ifnar_params}
    return t, states0, params, stim_data

states_labels = [r"IFN$\beta$", "IFNAR inactive", "IFNAR active", "ISGF3 inactive", "ISGF3 active", "ISG mRNA"]

# # Parameter scan for IFNb synthesis rate
# print("#################################################################################################")
# print("Running parameter scan for IFNb synthesis rate")
# isg_syn = 0.5
# isg_deg = 0.5
# stimulus = "CpG"
# ifnb_syn_list = np.linspace(0, 100, 50)
# t_eval = np.linspace(0, 60*8, 60*8+1)
# t, states0, params, stim_data = initialize_simulation(N= 0.5, I=0.01, stim_time=60*8, genotype="WT")

# # Plot input curves
# stim_data_ko = initialize_simulation(N= 0.5, I=0.5, stim_time=60*8, genotype="p50KO")[3]
# fig = plt.figure()
# plt.plot(stim_data[0], label="NFkB-WT", color = "darkorange")
# plt.plot(stim_data_ko[0], label="NFkB-p50KO", color = "orange")
# plt.plot(stim_data[1], label="IRF", color = "blue")
# plt.plot(stim_data[2], label="p50-WT", color = "black")
# plt.plot(stim_data_ko[2], label="p50-KO", color = "red")
# plt.xlabel("Time (min)")
# plt.ylabel("Concentration (nM)")
# plt.legend()
# plt.savefig("%sinput_curves_%s.png" % (results_dir, stimulus))

# # print("Max NFkB in WT: %.4f\tMax NFkB in p50KO: %.4f" % (np.max(stim_data[0]), np.max(stim_data_ko[0])))

# all_states = np.zeros((len(ifnb_syn_list), len(states0), len(t_eval)))
# print("Running parameter scan in WT for IFNb synthesis rate")

# best_ifnb_syn = 0
# best_isgf3_max = 0
# start = time.time()
# with Pool(30) as p:
#     for i in range(len(ifnb_syn_list)):
#         print("Running row %d of %d for IFNb synthesis" % (i+1, len(ifnb_syn_list)))
#         all_states[i,:,:] = p.starmap(run_simulation_wrapper, 
#                             [(t, states0, params, isg_syn, isg_deg, ifnb_syn_list[i], t_eval, stim_data)])[0].y
#         max_ifnb = np.max(all_states[i,0,:])
#         max_isgf3 = np.max(all_states[i,4,:])
#         if max_ifnb > 50 and max_ifnb < 100:
#             print("Max IFNb: %.4f\tMax ISGF3: %.4f\tIFNb synthesis rate: %.4f" % (max_ifnb, max_isgf3, ifnb_syn_list[i]))
#             if max_isgf3 > best_isgf3_max:
#                 best_isgf3_max = max_isgf3
#                 best_ifnb_syn = ifnb_syn_list[i]
        
# end = time.time()
# print("Time elapsed: %.2f min" % ((end-start)/60))
# np.save("%sall_states_WT_IFNb_synthesis.npy" % results_dir, all_states)

# print("Best IFNb synthesis rate: %.4f" % best_ifnb_syn)

# all_states = np.load("%sall_states_WT_IFNb_synthesis.npy" % results_dir)
# # Filter for rows with max IFNb > 50 and < 100
# good_rows = np.where(np.max(all_states[:,0,:], axis=1) > 50)[0]
# good_rows = np.intersect1d(good_rows, np.where(np.max(all_states[:,0,:], axis=1) < 100)[0])
# all_states = all_states[good_rows,:,:]
# ifnb_syn_list = ifnb_syn_list[good_rows]

# all_states_ko = np.zeros((len(ifnb_syn_list), len(states0), len(t_eval)))
# print("Running parameter scan in p50KO for good IFNb synthesis rate")
# t, states0, params, stim_data = initialize_simulation(N= 0.5, I=0.5, stim_time=60*8, genotype="p50KO")
# start = time.time()
# with Pool(30) as p:
#     for i in range(len(ifnb_syn_list)):
#         print("Running row %d of %d for IFNb synthesis" % (i+1, len(ifnb_syn_list)))
#         all_states_ko[i,:,:] = p.starmap(run_simulation_wrapper, 
#                             [(t, states0, params, isg_syn, isg_deg, ifnb_syn_list[i], t_eval, stim_data)])[0].y
# end = time.time()
# print("Time elapsed: %.2f min" % ((end-start)/60))

# # Calculate fold change of max ISGF3*
# isgf3_max_WT = np.max(all_states[:,4,:], axis=1)
# isgf3_max_p50KO = np.max(all_states_ko[:,4,:], axis=1)
# isgf3_max_FC = isgf3_max_p50KO / isgf3_max_WT
# print("Quantiles of ISGF3* fold change: %s" % str(np.quantile(isgf3_max_FC, [0.25, 0.5, 0.75])))
# print("Max ISGF3* fold change: %.4f" % np.max(isgf3_max_FC))

# best_ifnb_syn = ifnb_syn_list[np.argmax(isgf3_max_FC)]
# print("Best IFNb synthesis rate: %.4f" % best_ifnb_syn)

# np.save("best_ifnb_syn.npy", best_ifnb_syn)

# print("/n/n#################################################################################################")

# # Parameter scan
print("Running parameter scan for CpG stimulation")
stimulus = "CpG"
stim_time = 60*8
N=0.25
I=0.1
t, states0, params, stim_data = initialize_simulation(N= N, I=I, stim_time=stim_time, genotype="WT")
# ifnb_syn = np.load("%s/%s_best_ifnb_synthesis.npy" % ("results/ifnb_parameter_scan/", stimulus))
ifnb_syn = np.load("%s/%s_N-%s_best_ifnb_synthesis.npy" % ("results/ifnb_parameter_scan/different_N/", stimulus, N))
isg_syn_list = np.linspace(0, 1, 50)
isg_deg_list = np.linspace(0, 1, 50)
t_eval = np.linspace(t[0], t[1], t[1]*2+1)

all_states = np.zeros((len(isg_syn_list), len(isg_deg_list), len(states0), len(t_eval)))
print("Running parameter scan for WT")
start = time.time()
with Pool(30) as p:
    for i in range(len(isg_syn_list)):
        print("Running row %d of %d for WT" % (i+1, len(isg_syn_list)))
        results = p.starmap(run_simulation_wrapper, 
                            [(t, states0, params, isg_syn_list[i], isg_deg_list[j], ifnb_syn, t_eval, stim_data) for j in range(len(isg_deg_list))])
        for j in range(len(isg_deg_list)):
            all_states[i,j,:,:] = results[j].y

end = time.time()
print("Time elapsed: %.2f min" % ((end-start)/60))
np.save("%sall_states_WT_CpG_8hr_stim.npy" % results_dir, all_states)

t, states0, params, stim_data = initialize_simulation(N= N, I=I, stim_time=stim_time, genotype="p50KO")

all_states = np.zeros((len(isg_syn_list), len(isg_deg_list), len(states0), len(t_eval)))
print("Running parameter scan for p50KO")
start = time.time()
with Pool(30) as p:
    for i in range(len(isg_syn_list)):
        print("Running row %d of %d for p50KO" % (i+1, len(isg_syn_list)))
        results = p.starmap(run_simulation_wrapper, 
                            [(t, states0, params, isg_syn_list[i], isg_deg_list[j], ifnb_syn, t_eval, stim_data) for j in range(len(isg_deg_list))])
        for j in range(len(isg_deg_list)):
            all_states[i,j,:,:] = results[j].y

end = time.time()
print("Time elapsed: %.2f min" % ((end-start)/60))
np.save("%sall_states_p50KO_CpG_8hr_stim.npy" % results_dir, all_states)

# Plot ISG mRNA level
all_states_WT = np.load("%sall_states_WT_CpG_8hr_stim.npy" % results_dir)
all_states_p50KO = np.load("%sall_states_p50KO_CpG_8hr_stim.npy" % results_dir)

all_isg_syn, all_isg_deg = np.meshgrid(isg_syn_list, isg_deg_list)


isg_stim_WT = all_states_WT[:,:,5,stim_time].squeeze()
isg_stim_p50KO = all_states_p50KO[:,:,5,stim_time].squeeze()

isg_stim_p50KO[isg_stim_p50KO == 0] = 10**-20

# Plot distribution of ISG values at stim time
fig = plt.figure()
plt.hist(isg_stim_WT.flatten(), 50, alpha=0.5, label="WT")
plt.hist(isg_stim_p50KO.flatten(), 50, alpha=0.5, label="p50KO")
plt.legend()
plt.xlabel("ISG mRNA level at t=%d" % stim_time)
plt.ylabel("Frequency")
plt.savefig("%sisg_%s_stim_hist" % (results_dir, stim_time))

# Plot scatter plot of ISG values WT vs p50KO at stim time
fig = plt.figure()
plt.scatter(isg_stim_WT.flatten(), isg_stim_p50KO.flatten(), alpha=0.5)
plt.xlabel("ISG mRNA level at t=%d WT" % stim_time)
plt.ylabel("ISG mRNA level at t=%d p50KO" % stim_time)
plt.savefig("%sisg_%s_stim_scatter" % (results_dir, stim_time))

isg_FC_genotype = isg_stim_WT / isg_stim_p50KO
# Remove inf values
isg_FC_genotype[isg_FC_genotype == np.inf] = 0
# print(min(isg_FC_genotype.flatten()), max(isg_FC_genotype.flatten()))
# Plot distribution of fold change in ISG values at t=180
fig = plt.figure()
plt.hist(isg_FC_genotype.flatten(), 50, alpha=0.5, log=True)
plt.xlabel("Fold change in ISG mRNA (WT/KO) level at t=%d" % stim_time)
plt.ylabel("Frequency")
plt.savefig("%sisg_%s_stim_FC_hist" % (results_dir, stim_time))

# Plot ISG fold change over 0hr for WT and p50KO
isg_0h_WT = all_states_WT[:,:,5,0].squeeze()
isg_0h_p50KO = all_states_p50KO[:,:,5,0].squeeze()

print("Shape of 8hr ISG mRNA level: %s" % str(isg_stim_WT.shape))
print("Shape of 0hr ISG mRNA level: %s" % str(isg_0h_WT.shape))
for row, column in zip(np.where(isg_0h_WT == 0)[0], np.where(isg_0h_WT == 0)[1]):
    isg_0h_WT[row, column] = 10**-20
    print("Setting 0hr ISG mRNA level to 10^-2 for WT, syn=%.2f, deg=%.2f" % (isg_syn_list[row], isg_deg_list[column]))

for row, column in zip(np.where(isg_0h_p50KO == 0)[0], np.where(isg_0h_p50KO == 0)[1]):
    isg_0h_p50KO[row, column] = 10**-20
    print("Setting 0hr ISG mRNA level to 10^-2 for p50KO, syn=%.2f, deg=%.2f" % (isg_syn_list[row], isg_deg_list[column]))


isg_FC_WT = isg_stim_WT / isg_0h_WT
isg_FC_p50KO = isg_stim_p50KO / isg_0h_p50KO

fig = plt.figure()
plt.scatter(isg_FC_WT.flatten(), isg_FC_p50KO.flatten(), alpha=0.5)
plt.xlabel("ISG mRNA fold change (WT) at t=%d" % stim_time)
plt.ylabel("ISG mRNA fold change (p50KO) at t=%d" % stim_time)
plt.savefig("%sisg_0-%s_stim_FC_scatter" % (results_dir, stim_time))

print("Quantiles of WT ISG t=%d: %s\n Quantiles of p50KO ISG t=%d: %s\n Quantiles of ISG t=%d FC: %s" %
        (stim_time, np.quantile(isg_stim_WT.flatten(), [0.25, 0.5, 0.75]), stim_time, np.quantile(isg_stim_p50KO.flatten(), [0.25, 0.5, 0.75]), stim_time, np.quantile(isg_FC_genotype.flatten(), [0.25, 0.5, 0.75])))

#  Plot ISG fold change over 0hr for WT as a contour plot
fig = plt.figure()
plt.contourf(all_isg_deg, all_isg_syn, isg_FC_WT, 50, cmap="viridis")
plt.colorbar()
plt.xlabel("ISG synthesis rate")
plt.ylabel("ISG degradation rate")
plt.title("ISG mRNA fold change (WT) at t=%d" % stim_time)
plt.savefig("%sisg_0-%s_stim_FC_WT_contour" % (results_dir, stim_time))

#  Plot ISG fold change over 0hr for p50KO as a contour plot
fig = plt.figure()
plt.contourf(all_isg_deg, all_isg_syn, isg_FC_p50KO, 50, cmap="viridis")
plt.colorbar()
plt.xlabel("ISG synthesis rate")
plt.ylabel("ISG degradation rate")
plt.title("ISG mRNA fold change (p50KO) at t=%d" % stim_time)
plt.savefig("%sisg_0-%s_stim_FC_p50KO_contour" % (results_dir, stim_time))

# Find location where p50KO ISG FC is much higher than WT ISG FC
isg_FC_diff = isg_FC_p50KO - isg_FC_WT
isg_FC_diff[isg_FC_p50KO < 0 ] = 0
isg_FC_diff[isg_FC_WT < 0 ] = 0
isg_FC_diff[all_isg_syn == 0] = 0
isg_FC_diff[all_isg_deg == 0] = 0

if np.max(isg_FC_diff) > 0:
    for loc_row, loc_col in zip(np.where(isg_FC_diff == np.max(isg_FC_diff))[0], np.where(isg_FC_diff == np.max(isg_FC_diff))[1]):
        print("Max ISG FC difference at ISG synthesis rate = %.4f, ISG degradation rate = %.4f. WT FC = %.4f, p50KO FC = %.4f, diff = %.4f" %
                (all_isg_syn[loc_row, loc_col], all_isg_deg[loc_row, loc_col], isg_FC_WT[loc_row, loc_col], isg_FC_p50KO[loc_row, loc_col], isg_FC_diff[loc_row, loc_col]))
        print("ISG peaks at %.2f for WT and %.2f for p50KO" % (np.max(all_states_WT[loc_row, loc_col, 5, :]), np.max(all_states_p50KO[loc_row, loc_col, 5, :])))

        # Plot ISG timecourse at loc
        WT_timecourse = all_states_WT[loc_row, loc_col, 5, :]
        p50KO_timecourse = all_states_p50KO[loc_row, loc_col, 5, :]
        fig = plt.figure()
        plt.plot(t_eval, WT_timecourse, label="WT", color = "black")
        plt.plot(t_eval, p50KO_timecourse, label="p50KO", color = "red")
        fig.legend(bbox_to_anchor=(1.1, 0.5))
        plt.xlabel("Time (min)")
        plt.ylabel("ISG mRNA level")
        plt.title("ISG mRNA level at ISG synthesis rate = %.4f, ISG degradation rate = %.4f" % (all_isg_syn[loc_row, loc_col], all_isg_deg[loc_row, loc_col]))
        plt.savefig("%sisg_%s_stim_syn_%.4f_deg_%.4f_timecourse.png" % (results_dir, stim_time, all_isg_syn[loc_row, loc_col], all_isg_deg[loc_row, loc_col]))
        plt.close()

        # Plot ISG FC timecourse at loc fold change over 0hr
        WT_FC_timecourse = WT_timecourse/WT_timecourse[0]
        p50KO_FC_timecourse = p50KO_timecourse/p50KO_timecourse[0]
        fig = plt.figure()
        plt.plot(t_eval, WT_FC_timecourse, label="WT", color = "black")
        plt.plot(t_eval, p50KO_FC_timecourse, label="p50KO", color = "red")
        fig.legend(bbox_to_anchor=(1.1, 0.5))
        plt.xlabel("Time (min)")
        plt.ylabel("ISG mRNA fold change")
        plt.title("ISG mRNA fold change at ISG synthesis rate = %.4f, ISG degradation rate = %.4f" % (all_isg_syn[loc_row, loc_col], all_isg_deg[loc_row, loc_col]))
        plt.savefig("%sisg_%s_stim_syn_%.4f_deg_%.4f_FC_timecourse.png" % (results_dir, stim_time, all_isg_syn[loc_row, loc_col], all_isg_deg[loc_row, loc_col]))
        plt.close()

        # Plot timecourse of all states at loc
        WT_timecourse = all_states_WT[loc_row, loc_col, :, :]
        # print(WT_timecourse.shape)
        p50KO_timecourse = all_states_p50KO[loc_row, loc_col, :, :]
        fig, ax = plt.subplots(2, 3, figsize=(12,8))
        for i in range(6):
            ax[i//3, i%3].plot(t_eval, WT_timecourse[i,:], label="WT" if i == 0 else "", color = "black")
            ax[i//3, i%3].plot(t_eval, p50KO_timecourse[i,:], label="p50KO" if i == 0 else "", color = "red")
            ax[i//3, i%3].set_title(states_labels[i])
        fig.legend(bbox_to_anchor=(1.1,0.5))
        plt.tight_layout()
        plt.savefig("%sisg_%s_stim_syn_%.4f_deg_%.4f_all_states_timecourse.png" % (results_dir, stim_time, all_isg_syn[loc_row, loc_col], all_isg_deg[loc_row, loc_col]))
        plt.close()


else:
    print("No ISG FC difference in any parameter combination")

# max IFNb value for WT and p50KO
ifnb_WT = all_states_WT[:,:,0,:].squeeze()
ifnb_p50KO = all_states_p50KO[:,:,0,:].squeeze()
print("IFNb peaks at %.2f for WT and %.2f for p50KO" % (np.max(ifnb_WT), np.max(ifnb_p50KO)))
print("IFNAR* peaks at %.2f for WT and %.2f for p50KO" % (np.max(all_states_WT[:,:,2,:].squeeze()), np.max(all_states_p50KO[:,:,2,:].squeeze())))
print("ISGF3* peaks at %.2f for WT and %.2f for p50KO" % (np.max(all_states_WT[:,:,4,:].squeeze()), np.max(all_states_p50KO[:,:,4,:].squeeze())))



# ## old
# # # Make grid of max ISG values corresponding to all_isg_syn and all_isg_deg
# # max_isg = np.zeros(all_isg_syn.shape)
# # for i in range(all_isg_syn.shape[0]):
# #     for j in range(all_isg_syn.shape[1]):
# #         max_isg[i,j] = np.max(all_states[i,j,5,:])
# # print(all_states[49,0,5,np.arange(0,1000,100)])

# # max_max_loc = np.argmax(max_isg)
# # print("Max ISG mRNA level: %.4f at ISG synthesis rate = %.4f, ISG degradation rate = %.4f" %
# #         (np.max(max_isg), all_isg_syn.flatten()[max_max_loc], all_isg_deg.flatten()[max_max_loc]))
# # # states = run_simulation_wrapper(t, states0, params, all_isg_syn.flatten()[max_max_loc], all_isg_deg.flatten()[max_max_loc], ifnb_syn, t_eval, stim_data)
# # # state_list = [states.y[i,:] for i in range(states.y.shape[0])]
# # # plot_model(state_list, states_labels, states.t, "%sisg_model_max_isg" % results_dir,
# # #               "ISG model example", "Time (min)", "Concentration (nM)")
# # # print(np.max(states.y[-1,:]))

# # # max_isg_log = np.log10(max_isg)
# # # Plot max ISG values as contour plot
# # fig = plt.figure()
# # plt.contourf(all_isg_deg, all_isg_syn, max_isg, 50, cmap="viridis")
# # plt.colorbar()
# # plt.xlabel("ISG synthesis rate")
# # plt.ylabel("ISG degradation rate")
# # plt.title("Max ISG mRNA level (log10)")
# # plt.savefig("%smax_isg_contour" % results_dir)

# # # 3 hr stimulation
# # N, I, P = 1,1,0.01
# # N_times, I_times, P_times = [0, 120], [0, 120], [0, 120]
# # stim_data = get_stim_data(N, I, P, N_times, I_times, P_times)

# # for isg_syn, isg_deg in zip([0.5, 0.5, 0.5], [0.01, 0.1, 0.5]):
# #     states_ss, t_ss, all_t_ss, all_states_ss = run_steady_state(t, states0, params, isg_syn, isg_deg, ifnb_syn, stim_data)
# #     states = run_simulation(t, states_ss, params, isg_syn, isg_deg, ifnb_syn, t_eval, stim_data)
# #     state_list = [states.y[i,:] for i in range(states.y.shape[0])]
# #     plot_model(state_list, states_labels, states.t, "%sisg_model_3hr_stim_%.2f-syn_%.2f-deg" % (results_dir, isg_syn, isg_deg),
# #                 "ISG model %.2f ISG synthesis rate, %.2f ISG degradation rate" % (isg_syn, isg_deg), "Time (min)", "Concentration (nM)")


