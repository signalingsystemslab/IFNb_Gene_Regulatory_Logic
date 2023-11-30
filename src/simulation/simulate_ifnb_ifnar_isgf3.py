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

     # mRNA and protein and ISGF3
    parent_dir = "results/ifnb_ifnar_isgf3/"
    os.makedirs(parent_dir, exist_ok=True)
    num_states = 6

    print("\n##########\nSimulating IFNb, IFNAR, and ISGF3\n##########\n", flush=True)

    for i in range(1):
        dir = "%s/default_pars/" % parent_dir
        os.makedirs(dir, exist_ok=True)

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
        t_eval = cpg_results[0][1]
        np.savetxt("%s/t_eval.csv" % dir, t_eval, delimiter=",")

        # Plot inputs
        cpg_stim_data = cpg_results[0][2]
        lps_stim_data = lps_results[0][2]

        input_t = np.linspace(0, stim_time+1, stim_time+1+1)
        plot_inputs(input_t, cpg_stim_data[0], cpg_stim_data[1], cpg_stim_data[2], "CpG", dir)
        plot_inputs(input_t, lps_stim_data[0], lps_stim_data[1], lps_stim_data[2], "LPS", dir)

        # Load results
        cpg_timecourses = np.load("%s/cpg_results.npy" % dir)
        lps_timecourses = np.load("%s/lps_results.npy" % dir)
        t_eval = np.loadtxt("%s/t_eval.csv" % dir, delimiter=",")

        cmap = plt.get_cmap('viridis')
        colors = [cmap(i) for i in np.linspace(0, 1, 7)]
        
        # Plot results
        for state in range(num_states):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            for i in range(num_par_reps):
                ax.plot(t_eval, lps_timecourses[i,state,:], color=colors[state], alpha=0.1)
            ax.set_xlabel("Time (min)")
            ax.set_ylabel(r"%s (nM)" % state_names[state])
            ax.set_title("LPS stimulation, all parameter sets")
            plt.savefig("%s/lps_timecourses_%s.png" % (dir, state_titles[state]))
            ylim = ax.get_ylim()
            plt.close()

            fig = plt.figure()
            ax = fig.add_subplot(111)
            for i in range(num_par_reps):
                ax.plot(t_eval, cpg_timecourses[i,state,:], color=colors[state], alpha=0.1)
            ax.set_xlabel("Time (min)")
            ax.set_ylabel(r"%s (nM)" % state_names[state])
            ax.set_title("CpG stimulation, all parameter sets")
            ax.set_ylim(ylim)
            plt.savefig("%s/cpg_timecourses_%s.png" % (dir, state_titles[state]))
            plt.close()

        pars = {**ifnar_pars, **other_pars}

    genotype = "p50KO"
    print("\n##########\nSimulating IFNb, IFNAR, and ISGF3 for p50 KO \n##########\n", flush=True)

    for i in range(1):
        # Simulate for cpg and lps
        with Pool(40) as p:
            cpg_results = p.starmap(full_simulation, [(states0, t_params[i,:], pars, "CpG_%d" % i, "CpG", genotype, dir, num_states,
                                                       stim_time, None, False) for i in range(num_par_reps)])
            lps_results = p.starmap(full_simulation, [(states0, t_params[i,:], pars, "LPS_%d" % i, "LPS", genotype, dir, num_states,
                                                         stim_time, None, False) for i in range(num_par_reps)])
            
        # print(cpg_results[1][0].y[0,:])
        cpg_timecourses = [cpg_results[i][0].y[:,:] for i in range(num_par_reps)]
        cpg_timecourses = np.array(cpg_timecourses)
        np.save('%s/p50KO_cpg_results.npy' % dir, cpg_timecourses)
        lps_timecourses = [lps_results[i][0].y[:,:] for i in range(num_par_reps)]
        lps_timecourses = np.array(lps_timecourses)
        np.save('%s/p50KO_lps_results.npy' % dir, lps_timecourses)
        t_eval = cpg_results[0][1]
        np.savetxt("%s/p50KO_t_eval.csv" % dir, t_eval, delimiter=",")

        # Plot inputs
        cpg_stim_data = cpg_results[0][2]
        lps_stim_data = lps_results[0][2]

        input_t = np.linspace(0, stim_time+1, stim_time+1+1)
        plot_inputs(input_t, cpg_stim_data[0], cpg_stim_data[1], cpg_stim_data[2], "CpG_p50KO", dir)
        plot_inputs(input_t, lps_stim_data[0], lps_stim_data[1], lps_stim_data[2], "LPS_p50KO", dir)

        # Load results
        cpg_timecourses = np.load("%s/p50KO_cpg_results.npy" % dir)
        lps_timecourses = np.load("%s/p50KO_lps_results.npy" % dir)
        t_eval = np.loadtxt("%s/p50KO_t_eval.csv" % dir, delimiter=",")

        # Plot results
        for state in range(num_states):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            for i in range(num_par_reps):
                ax.plot(t_eval, lps_timecourses[i,state,:], color=colors[state], alpha=0.1)
            ax.set_xlabel("Time (min)")
            ax.set_ylabel(r"%s (nM)" % state_names[state])
            ax.set_title("LPS stimulation, p50 KO")
            plt.savefig("%s/p50KO_lps_timecourses_%s.png" % (dir, state_titles[state]))
            ylim = ax.get_ylim()
            plt.close()

            fig = plt.figure()
            ax = fig.add_subplot(111)
            for i in range(num_par_reps):
                ax.plot(t_eval, cpg_timecourses[i,state,:], color=colors[state], alpha=0.1)
            ax.set_xlabel("Time (min)")
            ax.set_ylabel(r"%s (nM)" % state_names[state])
            ax.set_title("CpG stimulation, p50 KO")
            ax.set_ylim(ylim)
            plt.savefig("%s/p50KO_cpg_timecourses_%s.png" % (dir, state_titles[state]))
            plt.close()
        
        # Load WT results
        cpg_timecourses_WT = np.load("%s/cpg_results.npy" % dir)
        lps_timecourses_WT = np.load("%s/lps_results.npy" % dir)

        # Calculate fold change p50KO/WT
        cpg_fold_change = np.zeros(np.shape(cpg_timecourses))
        lps_fold_change = np.zeros(np.shape(lps_timecourses))
        for i in range(num_par_reps):
            cpg_fold_change[i,:,:] = cpg_timecourses[i,:,:]/cpg_timecourses_WT[i,:,:]
            lps_fold_change[i,:,:] = lps_timecourses[i,:,:]/lps_timecourses_WT[i,:,:]

        # Plot fold change
        for state in range(num_states):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            for i in range(num_par_reps):
                ax.plot(t_eval, lps_fold_change[i,state,:], color=colors[state], alpha=0.1)
            ax.set_xlabel("Time (min)")
            ax.set_ylabel(r"%s fold change" % state_names[state])
            ax.set_title("LPS stimulation fold change p50 KO/WT")
            plt.savefig("%s/p50KO_lps_fold_change_%s.png" % (dir, state_titles[state]))
            plt.close()

            fig = plt.figure()
            ax = fig.add_subplot(111)
            for i in range(num_par_reps):
                ax.plot(t_eval, cpg_fold_change[i,state,:], color=colors[state], alpha=0.1)
            ax.set_xlabel("Time (min)")
            ax.set_ylabel(r"%s fold change" % state_names[state])
            ax.set_title("CpG stimulation fold change p50 KO/WT")
            plt.savefig("%s/p50KO_cpg_fold_change_%s.png" % (dir, state_titles[state]))
            plt.close()

if __name__ == "__main__":
    main()