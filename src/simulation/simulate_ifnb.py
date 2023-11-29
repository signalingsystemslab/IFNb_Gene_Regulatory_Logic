from simulation_functions import *

def initialize(t_pars, other_pars, stimulus, genotype, stim_time = 60*8, stim_data=None, plot = True):
    pars = get_t_params(t_pars)
    pars = {**pars, **other_pars}
    
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

    return pars, stim_data, stim_data_ss

def full_simulation(states0, t_params, params, name, stimulus, genotype, directory, num_states=1, stim_time = 60*8, stim_data=None, plot = True):
    name = "%s_%s_%s" % (name, stimulus, genotype)

    pars, stim_data, stim_data_ss = initialize(t_params, params, stimulus, genotype,stim_time, stim_data, plot)

    if plot:
        # Plot inputs
        input_t = np.linspace(0, stim_time+1, stim_time+2)
        plot_inputs(input_t, stim_data[0], stim_data[1], stim_data[2], name, directory)

    ## Simulate model
    t_eval = np.linspace(0, stim_time+1, stim_time+2)

    # Get steady state
    states0 = get_steady_state(states0, pars, stim_data_ss, t_eval, num_states=num_states)

    # Integrate model
    states = solve_ivp(IFN_model, [0, t_eval[-1]], states0, t_eval=t_eval, args=(pars, stim_data))
    return states, t_eval, stim_data

def main():

    states0 = np.array([0.01, 0.01, 10, 1, 90, 10, 1])
    state_names = [r"IFN$\beta$ mRNA", r"IFN$\beta$ protein", r"IFNAR inactive", r"IFNAR active", r"ISGF3 inactive", r"ISGF3 active", r"ISG mRNA"]
    state_titles = ["ifnb_rna", "ifnb_protein", "ifnar_in", "ifnar", "isgf3_in", "isgf3", "isg_rna"]
    genotype = "WT"
    stim_time = 60*8
    

    # Load all parameter sets
    t_params = np.loadtxt("../p50_model/opt_syn_datasets/results/p50_all_datasets_pars_local.csv", delimiter=",")
    num_par_reps = np.size(t_params,0)

    ifnar_pars = get_params("ifnar_params.csv")
    other_pars = get_params("other_params.csv")
    pars = {**ifnar_pars, **other_pars}

    p_syn_ifnb = 1

    # mRNA only
    parent_dir = "results/ifnb_mRNA/"
    os.makedirs(parent_dir, exist_ok=True)
    num_states = 1

    print("\n##########\nSimulating mRNA only\n##########\n", flush=True)

    for i in range(1):
        if "p_syn_ifnb" in pars.keys():
            p_syn_ifnb = pars["p_syn_ifnb"]
        else:
            pars["p_syn_ifnb"] = p_syn_ifnb

        dir = "%s/p_syn_ifnb_%.5f" % (parent_dir, p_syn_ifnb)
        os.makedirs(dir, exist_ok=True)

        print("\n##########\nSynthesis rate: %.5f\n##########\n" % p_syn_ifnb, flush=True)

        # Simulate for cpg and lps
        with Pool(40) as p:
            cpg_results = p.starmap(full_simulation, [(states0, t_params[i,:], pars, "CpG_%d" % i, "CpG", genotype, dir, num_states,
                                                       stim_time, None, False) for i in range(num_par_reps)])
            lps_results = p.starmap(full_simulation, [(states0, t_params[i,:], pars, "LPS_%d" % i, "LPS", genotype, dir, num_states,
                                                         stim_time, None, False) for i in range(num_par_reps)])
            
        # print(cpg_results[1][0].y[0,:])
        cpg_timecourses = [cpg_results[i][0].y[0,:] for i in range(num_par_reps)]
        cpg_timecourses = np.array(cpg_timecourses)
        lps_timecourses = [lps_results[i][0].y[0,:] for i in range(num_par_reps)]
        lps_timecourses = np.array(lps_timecourses)
        t_eval = cpg_results[0][1]
        np.savetxt("%s/cpg_results.csv" % dir, cpg_timecourses, delimiter=",")
        np.savetxt("%s/lps_results.csv" % dir, lps_timecourses, delimiter=",")
        np.savetxt("%s/t_eval.csv" % dir, t_eval, delimiter=",")

        # Plot inputs
        cpg_stim_data = cpg_results[0][2]
        lps_stim_data = lps_results[0][2]

        input_t = np.linspace(0, stim_time+1, stim_time+1+1)
        plot_inputs(input_t, cpg_stim_data[0], cpg_stim_data[1], cpg_stim_data[2], "CpG", dir)
        plot_inputs(input_t, lps_stim_data[0], lps_stim_data[1], lps_stim_data[2], "LPS", dir)

        # Plot inputs unscaled
        # I_curve_cpg = read_inputs("IRF", "CpG", scale=False)
        # N_curve_cpg = read_inputs("NFkB", "CpG", scale=False)
        # I_curve_lps = read_inputs("IRF", "LPS", scale=False)
        # N_curve_lps = read_inputs("NFkB", "LPS", scale=False)

        # plot_inputs(input_t, N_curve_cpg, I_curve_cpg, cpg_stim_data[2], "CpG_nM", dir, ylimits=False)
        # plot_inputs(input_t, N_curve_lps, I_curve_lps, lps_stim_data[2], "LPS_nM", dir, ylimits=False)

        # # Plot nfkb and irf timecourses
        # fig, ax = plt.subplots(1,2, figsize=(8,4))
        # ax[0].plot(I_curve_cpg[0], I_curve_cpg[1], label="CpG", marker="o", linewidth=0.5)
        # ax[0].plot(I_curve_lps[0], I_curve_lps[1], label="LPS", marker="o", linewidth=0.5)
        # ax[0].set_xlabel("Time (min)")
        # ax[0].set_ylabel("IRF (nM)")
        # ax[0].set_title("IRF timecourses")
        # ax[0].legend()
        
        # ax[1].plot(N_curve_cpg[0], N_curve_cpg[1], label="CpG", marker="o", linewidth=0.5)
        # ax[1].plot(N_curve_lps[0], N_curve_lps[1], label="LPS", marker="o", linewidth=0.5)
        # ax[1].set_xlabel("Time (min)")
        # ax[1].set_ylabel("NFkB (nM)")
        # ax[1].set_title("NFkB timecourses")
        # ax[1].legend()
        # plt.savefig("%s/nfkb_irf_timecourses.png" % dir)

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

        pars = {**ifnar_pars, **other_pars}

    # mRNA and protein
    parent_dir = "results/ifnb_mRNA_protein/"
    os.makedirs(parent_dir, exist_ok=True)
    num_states = 2

    print("\n##########\nSimulating mRNA and protein\n##########\n", flush=True)

    p_syn_ifnb_vals = [1, 0.1, 0.01, 0.001]
    for i in range(len(p_syn_ifnb_vals)):
        p_syn_ifnb = p_syn_ifnb_vals[i]

        if "p_syn_ifnb" in pars.keys():
            p_syn_ifnb = pars["p_syn_ifnb"]
        else:
            pars["p_syn_ifnb"] = p_syn_ifnb

        dir = "%s/p_syn_ifnb_%.5f" % (parent_dir, p_syn_ifnb)
        os.makedirs(dir, exist_ok=True)

        print("\n##########\nSynthesis rate: %.5f\n##########\n" % p_syn_ifnb, flush=True)

        # Simulate for cpg and lps
        with Pool(40) as p:
            cpg_results = p.starmap(full_simulation, [(states0, t_params[i,:], pars, "CpG_%d" % i, "CpG", genotype, dir, num_states,
                                                       stim_time, None, False) for i in range(num_par_reps)])
            lps_results = p.starmap(full_simulation, [(states0, t_params[i,:], pars, "LPS_%d" % i, "LPS", genotype, dir, num_states,
                                                         stim_time, None, False) for i in range(num_par_reps)])
            
        # print(cpg_results[1][0].y[0,:])
        cpg_timecourses = [cpg_results[i][0].y[:,:] for i in range(num_par_reps)]
        cpg_timecourses = np.array(cpg_timecourses)
        np.save('%s/cpg_results.npy' % dir, cpg_timecourses)
        lps_timecourses = [lps_results[i][0].y[:,:] for i in range(num_par_reps)]
        lps_timecourses = np.array(lps_timecourses)
        np.save('%s/lps_results.npy' % dir, lps_timecourses)

        # Load results
        cpg_timecourses = np.load("%s/cpg_results.npy" % dir)
        lps_timecourses = np.load("%s/lps_results.npy" % dir)

        # Plot results
        for state in range(num_states):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            for i in range(num_par_reps):
                ax.plot(t_eval, lps_timecourses[i,state,:], color="b", alpha=0.1)
            ax.set_xlabel("Time (min)")
            ax.set_ylabel(r"%s (nM)" % state_names[state])
            ax.set_title("LPS stimulation, all parameter sets")
            plt.savefig("%s/lps_timecourses_%s.png" % (dir, state_titles[state]))
            ylim = ax.get_ylim()
            plt.close()

            fig = plt.figure()
            ax = fig.add_subplot(111)
            for i in range(num_par_reps):
                ax.plot(t_eval, cpg_timecourses[i,state,:], color="b", alpha=0.1)
            ax.set_xlabel("Time (min)")
            ax.set_ylabel(r"%s (nM)" % state_names[state])
            ax.set_title("CpG stimulation, all parameter sets")
            ax.set_ylim(ylim)
            plt.savefig("%s/cpg_timecourses_%s.png" % (dir, state_titles[state]))
            plt.close()

        pars = {**ifnar_pars, **other_pars}

        

if __name__ == "__main__":
    main()