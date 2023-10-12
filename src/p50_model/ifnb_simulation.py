from p50_model import *
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
import pandas as pd
from multiprocessing import Pool
import os
import time
plt.style.use("~/IFN_paper/src/theme_bw.mplstyle")

dir = "results/ifnb_simulation/"
os.makedirs(dir, exist_ok=True)

def get_params(file):
    params = {}
    with open(file) as f:
        for line in f:
            (key, val) = line.split(",")
            params[key] = float(val)
    return params

def get_input(curve, t):
    if t < 0:
        return 0
    elif t > len(curve) - 1:
        return 0.01
    else:
        f = interp1d(range(len(curve)), curve)
        val = f([t])[0]
    return val

def get_inputs(N_curve, I_curve, P_curve, t):
    inputs = {}
    inputs["nfkb"] = get_input(N_curve, t)
    inputs["irf"] = get_input(I_curve, t)
    inputs["p50"] = get_input(P_curve, t)
    return inputs

def change_equations(t, states, pars, inputs):
    # Unpack states
    ifnb = states[0]

    # Unpack pars
    t1 = pars["t1"]
    t2 = pars["t2"]
    t3 = pars["t3"]
    t4 = pars["t4"]
    t5 = pars["t5"]
    t6 = pars["t6"]
    t_pars = [t1, t2, t3, t4, t5, t6]
    K = pars["K_i2"]
    C = pars["C"]
    p_syn_ifnb = pars["p_syn_ifnb"]
    p_deg_ifnb = pars["p_deg_ifnb"]

    # Unpack inputs
    nfkb = inputs["nfkb"]
    irf = inputs["irf"]
    p50 = inputs["p50"]

    # Calculate derivatives
    f = get_f(t_pars, K, C, nfkb, irf, p50)
    difnb = f * p_syn_ifnb - p_deg_ifnb * ifnb
    return [difnb]

def IFN_model(t, states, params, stim_data):
    N_curve, I_curve, P_curve = stim_data
    inputs = get_inputs(N_curve, I_curve, P_curve, t)
    # States: IFNb, IFNAR, IFNAR*, ISGF3, ISGF3*, ISG mRNA
    ifnb_module = change_equations(t, states[0:1], params, inputs)
    # if t % 10 < 0.0001:
    #     print("t = %.2f, dIFNb = %.2f" % (t, states[0]))
    return ifnb_module

pars = get_params("../p50_model/results/random_opt/ifnb_best_params_random_global.csv")
for p in pars:
    print(("%s: %.4f") % (p, pars[p]))

t_pars = [pars["t1"], pars["t2"], pars["t3"], pars["t4"], pars["t5"], pars["t6"]]

### Simulation of IFNb model
def ifnb_simulation(pars,name, stimulus="CpG"):
    name = "%s_%s" % (name, stimulus)
    p_syn_ifnb = [10, 1, 0.1, 0.01]
    t_pars = [pars["t1"], pars["t2"], pars["t3"], pars["t4"], pars["t5"], pars["t6"]]
    P_values = {"WT": 1, "p50KO": 0}

    stim_time = 60*8
    t_eval = np.linspace(0, stim_time+120, stim_time+120+1)
    if stimulus == "CpG":
        I = 0.05
        N_value = 0.25
    elif stimulus == "LPS":
        I = 0.25
        N_value = 1
    elif stimulus == "pIC":
        I = 0.75
        N_value = 0.5
    
    genotypes = ["WT", "p50KO"]

    ifnb_values_all = np.zeros((2, len(t_eval), len(p_syn_ifnb)))
    for p_syn in p_syn_ifnb:
        pars["p_syn_ifnb"] = p_syn

        traj_dir = "../simulation/"
        cell_traj = np.loadtxt("%sRepresentativeCellTraj_NFkBn_%s.csv" % (traj_dir, stimulus), delimiter=",")
        N_curve = cell_traj[1,:]
        N_curve = N_value*N_curve/np.max(N_curve)
        I_curve = [I for i in range(stim_time+120)]


        ifnb_values = np.zeros((2, len(t_eval)))
        for g in range(len(genotypes)):
            genotype = genotypes[g]
            P_curve = [P_values[genotype] for i in range(stim_time+120)]

            stim_data = [N_curve, I_curve, P_curve]
            stim_data_ss = [[0.00001 for i in range(stim_time+120)] for i in range(2)]
            stim_data_ss = [stim_data_ss[0], stim_data_ss[1], P_curve]
            # for i in range(3):
            #     print(max(stim_data_ss[i]))
            # plot input curve
            if p_syn_ifnb.index(p_syn) == 0:
                fig, ax = plt.subplots(1,3)
                if genotype == "WT":
                    for i in range(3):
                        ax[i].set_prop_cycle(plt.cycler("color", ["k"]))
                fig.set_size_inches(12,4)
                ax[0].plot(N_curve)
                ax[0].set_title("N")
                ax[1].plot(I_curve)
                ax[1].set_title("I")
                ax[2].plot(P_curve)
                ax[2].set_title("P")
                for i in range(3):
                    ax[i].set_ylim([-0.01, 1.01])

                plt.suptitle("Input curves for %s stimulation, %s" % (stimulus, genotype))
                plt.savefig("%s/input_curves_%s_%s.png" % (dir, stimulus, genotype))
                plt.close()

            # integrate IFN model to get steady state
            states0 = [0.01]
            states0 = solve_ivp(IFN_model, [0, stim_time+120], states0, t_eval=t_eval, args=(pars, stim_data_ss))
            states0 = states0.y
            diff = np.max(np.abs(states0[:,-1] - states0[:,0]))
            i = 0
            while diff > 0.01:
                states0 = solve_ivp(IFN_model, [0, stim_time+120], states0[:,-1], t_eval=t_eval, args=(pars, stim_data_ss))
                states0 = states0.y
                diff = np.max(np.abs(states0[:,-1] - states0[:,0]))
                i += 1
                if i > 100:
                    print("No steady state found for genotype %s" % genotype)
                    print("IFNb values: %s, diff = %.4f" % (states0[:,-1], diff))
                    break
            states0 = states0[:,-1]
            print("Steady state IFNb values: %s" % states0)

            # integrate IFN model with steady state as initial condition
            states = solve_ivp(IFN_model, [0, stim_time+120], states0, t_eval=t_eval, args=(pars, stim_data))
            ifnb_values[g,:] = states.y[0,:]
        ifnb_values_all[:,:,p_syn_ifnb.index(p_syn)] = ifnb_values

    genotype_folds = ifnb_values_all[1,:,:] / ifnb_values_all[0,:,:]
    max_fold_change = np.max(genotype_folds, axis=0)
    print("Fold change for KO/WT for %s stimulation: %s" % (name, max_fold_change))

    ifnb_fold_change = np.zeros((2, len(t_eval), len(p_syn_ifnb)))
    # calculate fold change for each row over first column
    for i in range(len(p_syn_ifnb)):
        for j in range(len(genotypes)):
            ifnb_fold_change[j,:,i] = ifnb_values_all[j,:,i] / ifnb_values_all[j,0,i]
    # plot fold change
    fig, ax = plt.subplots(2,2)
    ax = ax.ravel()
    for i in range(len(p_syn_ifnb)):
        for g in range(len(genotypes)):
            ax[i].plot(t_eval, ifnb_fold_change[g,:,i], label=genotypes[g] if i == 0 else "")
        ax[i].set_title("p_syn_ifnb = %.3f" % p_syn_ifnb[i])
        ax[i].set_xlabel("Time (min)")
        ax[i].set_ylabel("IFNb fold change")
    plt.tight_layout()
    fig.subplots_adjust(right=0.8)
    fig.legend(bbox_to_anchor=(1, 0.5))
    fig.subplots_adjust(top=0.85)
    plt.suptitle("IFNb fold change for %d min %s stimulation" % (stim_time, stimulus), fontsize=16)
    plt.savefig("%s/IFNb_fold_change_timecourse_%s.png" % (dir, name))
    plt.close()

    # plot IFNb values in nM
    fig, ax = plt.subplots(2,2)
    ax = ax.ravel()
    for i in range(len(p_syn_ifnb)):
        for g in range(len(genotypes)):
            ax[i].plot(t_eval, ifnb_values_all[g,:,i], label=genotypes[g] if i == 0 else "")
        ax[i].set_title("p_syn_ifnb = %.3f" % p_syn_ifnb[i])
        ax[i].set_xlabel("Time (min)")
        ax[i].set_ylabel("IFNb (nM)")
    plt.tight_layout()
    fig.subplots_adjust(right=0.8)
    fig.legend(bbox_to_anchor=(1, 0.5))
    fig.subplots_adjust(top=0.85)
    plt.suptitle("IFNb values for %d min %s stimulation" % (stim_time, stimulus), fontsize=16)
    plt.savefig("%s/IFNb_values_timecourse_%s.png" % (dir, name))
    plt.close()
    return ifnb_values_all, ifnb_fold_change

def main():
    print("Testing IFNb model simulation")
    cpg_values, cpg_fc = ifnb_simulation(pars,"fit_t_pars", stimulus="CpG")
    lps_values, lps_fc = ifnb_simulation(pars,"fit_t_pars", stimulus="LPS")
    pic_values, pic_fc = ifnb_simulation(pars,"fit_t_pars", stimulus="pIC")
    t_eval = np.linspace(0, 60*8+120, 60*8+120+1)
    p_syn_ifnb = [10, 1, 0.1, 0.01]

    print("Plotting IFNb values")
    for i in range(cpg_values.shape[2]):
        fig, ax = plt.subplots(1,2)
        fig.set_size_inches(8,4)
        ax[0].plot(t_eval, cpg_values[0,:,i], label="CpG")
        ax[0].plot(t_eval, lps_values[0,:,i], label="LPS")
        # ax[0].plot(t_eval, pic_values[0,:,i], label="pIC")
        ax[0].set_title("WT")
        ax[0].set_xlabel("Time (min)")
        ax[0].set_ylabel("IFNb (nM)")
        ax[1].plot(t_eval, cpg_values[1,:,i])
        ax[1].plot(t_eval, lps_values[1,:,i])
        # ax[1].plot(t_eval, pic_values[1,:,i])
        ax[1].set_title("p50KO")
        ax[1].set_xlabel("Time (min)")
        ax[1].set_ylabel("IFNb (nM)")
        # make subplot axes the same
        ylim = max(ax[0].get_ylim()[1], ax[1].get_ylim()[1])
        ax[0].set_ylim([0, ylim])
        ax[1].set_ylim([0, ylim])
        plt.tight_layout()
        fig.subplots_adjust(right=0.9)
        fig.legend(bbox_to_anchor=(1, 0.5))
        fig.subplots_adjust(top=0.85)
        plt.suptitle("IFNb values for %d min stimulations, p_syn_ifnb = %.3f" % (60*8, p_syn_ifnb[i]), fontsize=16)
        plt.savefig("%s/IFNb_values_timecourse_CpG_LPS_%d.png" % (dir, i))
        plt.close()

    # print maximum IFNb values for each stimulation and genotype
    print("Max IFNb values for each stimulation and genotype:")
    print("CpG WT: %.4f" % np.max(cpg_values[0,:,1]))
    print("CpG KO: %.4f" % np.max(cpg_values[1,:,1]))
    print("CpG fold change: %.4f" % (np.max(cpg_values[1,:,1]) / np.max(cpg_values[0,:,1])))
    print("LPS WT: %.4f" % np.max(lps_values[0,:,1]))
    print("LPS KO: %.4f" % np.max(lps_values[1,:,1]))
    print("LPS fold change: %.4f" % (np.max(lps_values[1,:,1]) / np.max(lps_values[0,:,1])))
    print("pIC WT: %.4f" % np.max(pic_values[0,:,1]))
    print("pIC KO: %.4f" % np.max(pic_values[1,:,1]))
    print("pIC fold change: %.4f" % (np.max(pic_values[1,:,1]) / np.max(pic_values[0,:,1])))
    print("with p_syn_ifnb = %.3f" % p_syn_ifnb[1])

    # for each time point, calcualte genotype fold change
    cpg_genotype_folds = cpg_values[1,:,:] / cpg_values[0,:,:]
    lps_genotype_folds = lps_values[1,:,:] / lps_values[0,:,:]
    pic_genotype_folds = pic_values[1,:,:] / pic_values[0,:,:]

    # plot genotype fold change for each stimulation
    print("Plotting genotype fold change")
    for i in range(cpg_values.shape[2]):
        fig, ax = plt.subplots()
        ax.set_prop_cycle(plt.cycler("color", plt.cm.Set2(np.linspace(0,1,3))))
        ax.plot(t_eval, cpg_genotype_folds[:,i], label="CpG")
        ax.plot(t_eval, lps_genotype_folds[:,i], label="LPS")
        ax.plot(t_eval, pic_genotype_folds[:,i], label="pIC")
        ax.set_title("KO/WT for p_syn_ifnb = %.3f" % p_syn_ifnb[i])
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("IFNb fold change KO/WT")
        plt.tight_layout()
        fig.subplots_adjust(right=0.9)
        fig.legend(bbox_to_anchor=(1, 0.5))
        fig.subplots_adjust(top=0.85)
        # plt.suptitle("IFNb fold change for %d min stimulations" % (60*8), fontsize=16)
        plt.savefig("%s/IFNb_fold_change_timecourse_%d.png" % (dir, i))


    # plot timecourse fold change for CpG and LPS
    print("Plotting timecourse fold change for CpG and LPS")
    for i in range(cpg_fc.shape[2]):
        fig, ax = plt.subplots(1,2)
        fig.set_size_inches(8,4)
        ax[0].plot(t_eval, cpg_fc[0,:,i], label="CpG")
        ax[0].plot(t_eval, lps_fc[0,:,i], label="LPS")
        # ax[0].plot(t_eval, pic_fc[0,:,i], label="pIC")
        ax[0].set_title("WT")
        ax[0].set_xlabel("Time (min)")
        ax[0].set_ylabel("IFNb fold change")
        ax[1].plot(t_eval, cpg_fc[1,:,i])
        ax[1].plot(t_eval, lps_fc[1,:,i])
        # ax[1].plot(t_eval, pic_fc[1,:,i])
        ax[1].set_title("p50KO")
        ax[1].set_xlabel("Time (min)")
        ax[1].set_ylabel("IFNb fold change")
        # make subplot axes the same
        ylim = max(ax[0].get_ylim()[1], ax[1].get_ylim()[1])
        ax[0].set_ylim([0, ylim])
        ax[1].set_ylim([0, ylim])
        plt.tight_layout()
        fig.subplots_adjust(right=0.9)
        fig.legend(bbox_to_anchor=(1, 0.5))
        fig.subplots_adjust(top=0.85)
        plt.suptitle("IFNb fold change for %d min stimulation, p_syn_ifnb = %.3f" % (60*8, p_syn_ifnb[i]), fontsize=16)
        plt.savefig("%s/IFNb_fold_change_timecourse_CpG_LPS_%d.png" % (dir, i))
        plt.close()

    # Plot each stimulus on a separate plot colored by genotype for each p_syn_ifnb
    print("Plotting timecourse fold change for each stimulus")
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['k', 'r'])
    for i in range(cpg_fc.shape[2]):
        fig, ax = plt.subplots(1,3)
        fig.set_size_inches(12,4)
        ax[0].plot(t_eval, cpg_fc[0,:,i], label="WT")
        ax[0].plot(t_eval, cpg_fc[1,:,i], label="p50KO")
        ax[0].set_title("CpG")
        ax[0].set_xlabel("Time (min)")
        ax[0].set_ylabel("IFNb fold change")
        ax[1].plot(t_eval, lps_fc[0,:,i])
        ax[1].plot(t_eval, lps_fc[1,:,i])
        ax[1].set_title("LPS")
        ax[1].set_xlabel("Time (min)")
        ax[1].set_ylabel("IFNb fold change")
        ax[2].plot(t_eval, pic_fc[0,:,i])
        ax[2].plot(t_eval, pic_fc[1,:,i])
        ax[2].set_title("pIC")
        ax[2].set_xlabel("Time (min)")
        ax[2].set_ylabel("IFNb fold change")
        plt.tight_layout()
        fig.subplots_adjust(right=0.9)
        fig.legend(bbox_to_anchor=(1, 0.5))
        fig.subplots_adjust(top=0.85)
        plt.subplots_adjust(wspace=0.3)
        plt.suptitle("IFNb fold change for %d min stimulations, p_syn_ifnb = %.3f" % (60*8, p_syn_ifnb[i]), fontsize=16)
        plt.savefig("%s/IFNb_fold_change_timecourse_by_stim_%d.png" % (dir, i))
        plt.close()
        
    for i in range(cpg_fc.shape[2]):
        fig, ax = plt.subplots(1,3)
        fig.set_size_inches(12,4)
        ax[0].plot(t_eval, cpg_values[0,:,i], label="WT")
        ax[0].plot(t_eval, cpg_values[1,:,i], label="p50KO")
        ax[0].set_title("CpG")
        ax[0].set_xlabel("Time (min)")
        ax[0].set_ylabel("IFNb (nM)")
        ax[1].plot(t_eval, lps_values[0,:,i])
        ax[1].plot(t_eval, lps_values[1,:,i])
        ax[1].set_title("LPS")
        ax[1].set_xlabel("Time (min)")
        ax[1].set_ylabel("IFNb (nM)")
        ax[2].plot(t_eval, pic_values[0,:,i])
        ax[2].plot(t_eval, pic_values[1,:,i])
        ax[2].set_title("pIC")
        ax[2].set_xlabel("Time (min)")
        ax[2].set_ylabel("IFNb (nM)")
        plt.tight_layout()
        fig.subplots_adjust(right=0.9)
        fig.legend(bbox_to_anchor=(1, 0.5))
        fig.subplots_adjust(top=0.85)
        plt.subplots_adjust(wspace=0.3)
        plt.suptitle("IFNb values for %d min stimulations, p_syn_ifnb = %.3f" % (60*8, p_syn_ifnb[i]), fontsize=16)
        plt.savefig("%s/IFNb_values_timecourse_by_stim_%d.png" % (dir, i))
        plt.close()

if __name__ == "__main__":
    main()