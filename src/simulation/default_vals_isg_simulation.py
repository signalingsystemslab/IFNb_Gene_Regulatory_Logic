# Test simulation of isg expression
from simulate_isg_expression import *
import os
import time
import scipy.optimize as opt
from multiprocessing import Pool
import pandas as pd

### Test simulation of IFNb model
# def full_simulation(states0, pars, name, stimulus, genotype, directory = results_dir, stim_time = 60*8):
#     name = "%s_%s_%s" % (name, stimulus, genotype)

#     if stimulus not in ["CpG", "LPS", "pIC"]:
#         raise ValueError("Stimulus must be CpG, LPS, or pIC")

#     ## Inputs
#     P_values = {"WT": 1, "p50KO": 0}
#     I_values = {"CpG": 0.05, "LPS": 0.25, "pIC": 0.75}
#     N_values = {"CpG": 0.25, "LPS": 1, "pIC": 0.5}
#     P_curve = [P_values[genotype] for i in range(stim_time+120)]
#     traj_dir = "../simulation/"
#     cell_traj = np.loadtxt("%sRepresentativeCellTraj_NFkBn_%s.csv" % (traj_dir, stimulus), delimiter=",")

#     N_curve = cell_traj[1,:]
#     N_curve = N_values[stimulus]*N_curve/np.max(N_curve)
#     I_curve = [I_values[stimulus] for i in range(stim_time+60)]

#     stim_data = [N_curve, I_curve, P_curve]
#     stim_data_ss = [[0.00001 for i in range(stim_time+120)] for i in range(2)]
#     stim_data_ss = [stim_data_ss[0], stim_data_ss[1], P_curve]

#     # Plot inputs
#     input_t = np.linspace(0, stim_time+120, stim_time+120+1)
#     input_N = [get_input(N_curve, t) for t in input_t]
#     input_I = [get_input(I_curve, t) for t in input_t]
#     input_P = [get_input(P_curve, t) for t in input_t]

#     fig, ax = plt.subplots(1,3)
#     if genotype == "WT":
#         for i in range(3):
#             ax[i].set_prop_cycle(plt.cycler("color", ["k"]))
#     fig.set_size_inches(12,4)
#     ax[0].plot(input_N)
#     ax[0].set_title("N")
#     ax[1].plot(input_I)
#     ax[1].set_title("I")
#     ax[2].plot(input_P)
#     ax[2].set_title("P")
#     for i in range(3):
#         ax[i].set_ylim([-0.01, 1.01])
#     plt.suptitle("Input curves for %s stimulation" % name)
#     plt.savefig("%s/input_curves_%s.png" % (directory, name))
#     plt.close()

#     ## Simulate model
#     t_eval = np.linspace(0, stim_time+120, stim_time+120+1)

#     # Get steady state
#     states0 = get_steady_state(states0, pars, stim_data_ss, t_eval)
#     print("Steady state IFNb values: %s" % states0)

#     # Integrate model
#     states = solve_ivp(IFN_model, [0, t_eval[-1]], states0, t_eval=t_eval, args=(pars, stim_data))
#     return states, t_eval, stim_data

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

def full_simulation_final_ifnb(states0, pars, name, stimulus, genotype, directory, stim_time = 60*8, stim_data=None):
    states, t_eval, stim_data = full_simulation(states0 = states0, pars = pars, name = name, stimulus=stimulus, genotype=genotype, 
                                                directory = directory, stim_time = stim_time, stim_data=stim_data, plot=False)
    ifnb_final = states.y[0,-1]
    return ifnb_final

def main():
    start = time.time()
    results_dir = "./results/test_simulation/"
    os.makedirs(results_dir, exist_ok=True)

    # Load ifnb model parameters
    print("Loading parameters")
    pars = get_params("../p50_model/results/random_opt/ifnb_best_params_random_global.csv")
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

    # Simulation with default parameters
    print("Simulating")
    wt_cpg_states, t_eval, stim_data = full_simulation(states0 = states0, pars = pars, name = "default_pars", 
                                                stimulus="CpG", genotype="WT", directory = results_dir, stim_time = 60*8)
    print("For %s, IFNb peaks at %.4f nM" % ("WT CpG", np.max(wt_cpg_states.y[0,:])))
    ko_cpg_states, t_eval, stim_data = full_simulation(states0 = states0, pars = pars, name = "default_pars",
                                                stimulus="CpG", genotype="p50KO", directory = results_dir, stim_time = 60*8)
    print("For %s, IFNb peaks at %.4f nM" % ("p50KO CpG", np.max(ko_cpg_states.y[0,:])))
    wt_lps_states, t_eval, stim_data = full_simulation(states0 = states0, pars = pars, name = "default_pars",
                                                stimulus="LPS", genotype="WT", directory = results_dir, stim_time = 60*8)
    print("For %s, IFNb peaks at %.4f nM" % ("WT LPS", np.max(wt_lps_states.y[0,:])))
    ko_lps_states, t_eval, stim_data = full_simulation(states0 = states0, pars = pars, name = "default_pars",
                                                stimulus="LPS", genotype="p50KO", directory = results_dir, stim_time = 60*8)
    print("For %s, IFNb peaks at %.4f nM" % ("p50KO LPS", np.max(ko_lps_states.y[0,:])))
    
    # Plot
    print("Plotting")
    plot_ifnb(wt_cpg_states, "default_pars_CpG_WT", results_dir)
    plot_ifnb(ko_cpg_states, "default_pars_CpG_p50KO", results_dir)
    plot_ifnb(wt_lps_states, "default_pars_LPS_WT", results_dir)
    plot_ifnb(ko_lps_states, "default_pars_LPS_p50KO", results_dir)

    plot_all(wt_cpg_states, "default_pars_CpG_WT", results_dir)
    plot_all(ko_cpg_states, "default_pars_CpG_p50KO", results_dir)
    plot_all(wt_lps_states, "default_pars_LPS_WT", results_dir)
    plot_all(ko_lps_states, "default_pars_LPS_p50KO", results_dir)

    plot_CpG_LPS(wt_cpg_states, wt_lps_states, "default_pars_WT", results_dir)
    plot_CpG_LPS(ko_cpg_states, ko_lps_states, "default_pars_p50KO", results_dir)

    end = time.time()
    print("Finished after %.2f minutes" % ((end-start)/60))

    # Compare predicted IFNb to testing data
    print("Comparing predicted IFNb to testing data")
    test_data = pd.read_csv("../data/p50_testing_data.csv")
    all_params = pd.read_csv("../p50_model/results/random_opt/p50_random_global_optimization_results.csv", index_col=0)
    # print(all_params)

    p_deg_ifnb = pars["p_deg_ifnb"]
    p50_range = np.arange(0, 2.1, 0.2)
    print("Testing %d p50 values" % len(p50_range))

    WT_p50_row = test_data.loc[test_data["p50"]==1,:].index[0]
    test_WT_p50_loc = np.where(p50_range==1)[0][0]
    # print("WT p50 row: %d" % WT_p50_row)
    # print("WT p50 loc: %d" % test_WT_p50_loc)

    for model in ["B1", "B2", "B3", "B4"]:
        stim_time = 3*60
        pars = all_params.loc[model,:].to_dict()
        ifnar_pars = get_params("ifnar_params.csv")
        other_pars = get_params("other_params.csv")
        pars.update(ifnar_pars)
        pars.update(other_pars)
        pars["p_deg_ifnb"] = p_deg_ifnb
        N_curves = [[test_data["NFkB"][0] for i in range(stim_time+10)] for j in range(len(p50_range))]
        I_curves = [[test_data["IRF"][0] for i in range(stim_time+10)] for j in range(len(p50_range))]
        P_curves = [[p50_range[j] for i in range(stim_time+10)] for j in range(len(p50_range))]

        stim_data_all = [(N_curves[j], I_curves[j], P_curves[j]) for j in range(len(p50_range))]

        print("Simulating")
        ifnb_peaks = np.zeros(len(p50_range))
        with Pool(40) as p:
            ifnb_peaks = p.starmap(full_simulation_final_ifnb, [(states0, pars, "p50_test_%d" %j, "LPS", "p50_%d" %j, results_dir, stim_time, stim_data_all[j]) for j in range(len(p50_range))])
        ifnb_peaks = np.array(ifnb_peaks) / ifnb_peaks[test_WT_p50_loc] * test_data["IFNb"][WT_p50_row]

        # Plot
        print("Plotting")
        plt.figure()
        plt.plot(p50_range, ifnb_peaks, label=model)
        plt.scatter(test_data["p50"], test_data["IFNb"], s=50, label="Testing Data", zorder =2, color="black")
        plt.xlabel("p50")
        plt.ylabel(r"IFN$\beta$ max value")
        plt.ylim([0,1])
        plt.title("Model predictions for all models")
        plt.grid(False)
        plt.legend()
        plt.savefig("%s/random_opt_testing_predictions_%s.png" % (results_dir, model), bbox_inches="tight")
        plt.close()

if __name__ == "__main__":
    main()