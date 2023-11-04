# Test simulation of isg expression
from simulate_isg_expression import *
import os
import time
import scipy.optimize as opt
from multiprocessing import Pool
import pandas as pd

def plot_ifnb(states, name, directory):
    t = states.t
    ifnb = states.y[0,:]
    fig, ax = plt.subplots()
    if "WT" in name:
        ax.set_prop_cycle(plt.cycler("color", ["k"]))        
    plt.plot(t, ifnb)
    plt.xlabel("Time (min)")
    plt.ylabel(r"IFN$\beta$ (nM)")
    plt.title(r"IFN$\beta$ timecourse for %s" % name)
    plt.savefig("%s/IFNb_timecourse_%s.png" % (directory, name))
    plt.close()

def plot_all(states, name, directory):
    t = states.t
    states = states.y
    states_labels = [r"IFN$\beta$", "IFNAR", "IFNAR*", "ISGF3", "ISGF3*", "ISG mRNA"]

    fig, ax = plt.subplots(2, 3)
    for i in range(6):
        ax[i//3, i%3].plot(t, states[i,:])
        ax[i//3, i%3].set_title(states_labels[i])
        ax[i//3, i%3].set_xlabel("Time (min)")
        ax[i//3, i%3].set_ylabel("Concentration (nM)")
    plt.tight_layout()
    fig.subplots_adjust(top=0.9)
    plt.suptitle(r"IFN$\beta$ simulations for %s" % name)
    plt.savefig("%s/all_states_timecourse_%s.png" % (directory, name))
    plt.close()

def plot_CpG_LPS(states_cpg, states_lps, name, directory):
    t = states_cpg.t
    states_cpg = states_cpg.y
    states_lps = states_lps.y
    states_labels = [r"IFN$\beta$", "IFNAR", "IFNAR*", "ISGF3", "ISGF3*", "ISG mRNA"]

    fig, ax = plt.subplots(2, 3)
    for i in range(6):
        ax[i//3, i%3].set_prop_cycle(plt.cycler("color", plt.cm.viridis(np.linspace(0,1,2))))
        ax[i//3, i%3].plot(t, states_cpg[i,:], label="CpG" if i == 0 else "")
        ax[i//3, i%3].plot(t, states_lps[i,:], label="LPS" if i == 0 else "")
        ax[i//3, i%3].set_title(states_labels[i])
        ax[i//3, i%3].set_xlabel("Time (min)")
        ax[i//3, i%3].set_ylabel("Concentration (nM)")
    plt.tight_layout()
    fig.subplots_adjust(right=0.9)
    fig.legend(bbox_to_anchor=(1.1,0.5))
    fig.subplots_adjust(top=0.9)
    plt.suptitle(r"IFN$\beta$ simulations for %s" % name)
    plt.savefig("%s/all_states_CpG_LPS_timecourse_%s.png" % (directory, name))
    plt.close()

    # plot each state separately as a small square
    for i in range(6):
        fig, ax = plt.subplots()
        fig.set_size_inches(2,2)
        ax.set_prop_cycle(plt.cycler("color", plt.cm.viridis(np.linspace(0,1,2))))
        plt.plot(t, states_cpg[i,:], label="CpG")
        plt.plot(t, states_lps[i,:], label="LPS")
        plt.xlabel("Time (min)")
        plt.ylabel("Concentration (nM)")
        plt.title(states_labels[i])
        plt.legend()
        plt.savefig("%s/%s_CpG_LPS_timecourse_%s.png" % (directory, states_labels[i], name))
        plt.close()

def full_simulation_final_ifnb(states0, pars, name, stimulus, genotype, directory, stim_time = 60*8, stim_data=None):
    states, t_eval, stim_data = full_simulation(states0 = states0, pars = pars, name = name, stimulus=stimulus, genotype=genotype, 
                                                directory = directory, stim_time = stim_time, stim_data=stim_data, plot=False)
    ifnb_final = states.y[0,-1]
    isg_1h = states.y[-1,60]
    return ifnb_final, isg_1h

def main():
    start = time.time()
    results_dir = "./results/default_vals_simulation/"
    os.makedirs(results_dir, exist_ok=True)

    # Load ifnb model parameters
    print("Loading parameters")
    pars = get_params("../p50_model/results/grid_opt/ifnb_best_params_grid_local.csv")
    print("Simulating with parameters:")
    
    ifnar_pars = get_params("ifnar_params.csv")
    other_pars = get_params("other_params.csv")
    pars.update(ifnar_pars)
    pars.update(other_pars)

    for par in pars:
        print("%s: %.4f" % (par, pars[par]))

    # Set up inputs
    print("Setting up inputs")
    states0 = [0.01, 10, 1, 90, 10, 1]

    # # Simulation with default parameters
    # print("Simulating", flush=True)
    # wt_cpg_states, t_eval, stim_data = full_simulation(states0 = states0, pars = pars, name = "default_pars", 
    #                                             stimulus="CpG", genotype="WT", directory = results_dir, stim_time = 60*8)
    # print("For %s, IFNb peaks at %.4f nM" % ("WT CpG", np.max(wt_cpg_states.y[0,:])))
    # ko_cpg_states, t_eval, stim_data = full_simulation(states0 = states0, pars = pars, name = "default_pars",
    #                                             stimulus="CpG", genotype="p50KO", directory = results_dir, stim_time = 60*8)
    # print("For %s, IFNb peaks at %.4f nM" % ("p50KO CpG", np.max(ko_cpg_states.y[0,:])))
    # wt_lps_states, t_eval, stim_data = full_simulation(states0 = states0, pars = pars, name = "default_pars",
    #                                             stimulus="LPS", genotype="WT", directory = results_dir, stim_time = 60*8)
    # print("For %s, IFNb peaks at %.4f nM" % ("WT LPS", np.max(wt_lps_states.y[0,:])))
    # ko_lps_states, t_eval, stim_data = full_simulation(states0 = states0, pars = pars, name = "default_pars",
    #                                             stimulus="LPS", genotype="p50KO", directory = results_dir, stim_time = 60*8)
    # print("For %s, IFNb peaks at %.4f nM" % ("p50KO LPS", np.max(ko_lps_states.y[0,:])))
    
    # # Plot
    # print("Plotting", flush=True) 
    # plot_ifnb(wt_cpg_states, "default_pars_CpG_WT", results_dir)
    # plot_ifnb(ko_cpg_states, "default_pars_CpG_p50KO", results_dir)
    # plot_ifnb(wt_lps_states, "default_pars_LPS_WT", results_dir)
    # plot_ifnb(ko_lps_states, "default_pars_LPS_p50KO", results_dir)

    # plot_all(wt_cpg_states, "default_pars_CpG_WT", results_dir)
    # plot_all(ko_cpg_states, "default_pars_CpG_p50KO", results_dir)
    # plot_all(wt_lps_states, "default_pars_LPS_WT", results_dir)
    # plot_all(ko_lps_states, "default_pars_LPS_p50KO", results_dir)

    # plot_CpG_LPS(wt_cpg_states, wt_lps_states, "default_pars_WT", results_dir)
    # plot_CpG_LPS(ko_cpg_states, ko_lps_states, "default_pars_p50KO", results_dir)

    # end = time.time()
    # print("Finished after %.2f minutes" % ((end-start)/60), flush=True)

    # Compare predicted IFNb to testing data
    print("Comparing predicted IFNb to testing data")
    test_data = pd.read_csv("../data/p50_testing_data.csv")
    all_params = pd.read_csv("../p50_model/results/grid_opt/p50_grid_local_optimization_results.csv", index_col=0)
    model_names = all_params.index

    p_deg_ifnb = pars["p_deg_ifnb"]
    p50_range = np.arange(0, 2.1, 0.2)
    print("Testing %d p50 values" % len(p50_range), flush=True)

    WT_p50_row = test_data.loc[test_data["p50"]==1,:].index[0]
    test_WT_p50_loc = np.where(p50_range==1)[0][0]

    # for model in model_names:
    #     start = time.time()
    #     stim_time = 8*60
    #     pars = all_params.loc[model,:].to_dict()
    #     if np.isnan(pars["K_i2"]):
    #         pars["K_i2"] = 0
    #     if np.isnan(pars["C"]):
    #         pars["C"] = 0
    #     ifnar_pars = get_params("ifnar_params.csv")
    #     other_pars = get_params("other_params.csv")
    #     pars.update(ifnar_pars)
    #     pars.update(other_pars)
    #     pars["p_deg_ifnb"] = p_deg_ifnb
    #     print("\n\n\n")
    #     for par in pars:
    #         print("%s: %.4f" % (par, pars[par]))

    #     N_curves = [[test_data["NFkB"][0] for i in range(stim_time+10)] for j in range(len(p50_range))]
    #     I_curves = [[test_data["IRF"][0] for i in range(stim_time+10)] for j in range(len(p50_range))]
    #     P_curves = [[p50_range[j] for i in range(stim_time+10)] for j in range(len(p50_range))]

    #     # stim_data_all = [(N_curves[j], I_curves[j], P_curves[j]) for j in range(len(p50_range))]

    #     print("Simulating %s" % model, flush=True)
    #     ifnb_peaks = np.zeros(len(p50_range))
    #     isg_1h_vals = np.zeros(len(p50_range))
    #     # with Pool(40) as p:
    #     #     ifnb_peaks = p.starmap(full_simulation_final_ifnb, [(states0, pars, "p50_test_%d" %j, "LPS", "p50_%d" %j, results_dir, stim_time, stim_data_all[j]) for j in range(len(p50_range))])
    #     for j in range(len(p50_range)):
    #         print("Simulating p50 = %.2f" % p50_range[j], flush=True)
    #         stim_time = 8*60

    #         N_curve = N_curves[j]
    #         I_curve = I_curves[j]
    #         P_curve = P_curves[j]
    #         stim_data = [N_curve, I_curve, P_curve]

    #         ifnb_final, isg_1h = full_simulation_final_ifnb(states0, pars, "p50_test_%d" %j, "LPS", "p50_%d" %j, results_dir, stim_time, stim_data)
    #         ifnb_peaks[j] = ifnb_final
    #         isg_1h_vals[j] = isg_1h
    #         print("IFNb peaks at %.4f nM and ISG 1h at %.4f nM" % (ifnb_final, isg_1h), flush=True)

    #     ifnb_peaks = np.array(ifnb_peaks) / ifnb_peaks[test_WT_p50_loc] * test_data["IFNb"][WT_p50_row]

    #     # Save ifnb peaks and isg 1h
    #     print("Saving results")
    #     p50_peaks_df = pd.DataFrame({"p50": p50_range, "IFNb": ifnb_peaks, "ISG_1h": isg_1h_vals})
    #     p50_peaks_df.to_csv("%s/p50_values_peaks_%s.csv" % (results_dir, model))

    for model in model_names:
        p50_peaks_df = pd.read_csv("%s/p50_values_peaks_%s.csv" % (results_dir, model), index_col=0)
        p50_range = p50_peaks_df["p50"]
        ifnb_peaks = p50_peaks_df["IFNb"]
        isg_1h_vals = p50_peaks_df["ISG_1h"]
        # Plot
        print("Plotting IFNb peaks")
        fig, ax = plt.subplots()
        ax.set_prop_cycle(plt.cycler("color", plt.cm.viridis(np.linspace(0,1,2))))
        plt.plot(p50_range, ifnb_peaks)
        # plt.scatter(test_data["p50"], test_data["IFNb"], s=50, label="Testing Data", zorder =2, color="black")
        plt.xlabel("p50")
        plt.ylabel(r"IFN$\beta$ max value")
        plt.ylim([0,1])
        plt.title(r"IFN$\beta$ 3h peak for %s" % model)
        plt.grid(False)
        plt.legend()
        plt.savefig("%s/different_p50_IFNb_predictions_%s.png" % (results_dir, model), bbox_inches="tight")
        plt.close()

        print("Plotting ISG 1h \n")
        fig, ax = plt.subplots()
        ax.set_prop_cycle(plt.cycler("color", plt.cm.viridis(np.linspace(0,1,2))))
        plt.plot(p50_range, isg_1h_vals)
        plt.xlabel("p50")
        plt.ylabel(r"ISG mRNA at 1h")
        # plt.ylim([0,1])
        plt.title(r"ISG mRNA at 1h for %s" % model)
        plt.grid(False)
        plt.legend()
        plt.savefig("%s/different_p50_ISG_1h_predictions_%s.png" % (results_dir, model), bbox_inches="tight")
        plt.close()

        end = time.time()
        print("Finished after %.2f minutes" % ((end-start)/60))

if __name__ == "__main__":
    main()