import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pandas as pd
from ifnar_module import change_equations as ifnar_change_equations
from isg_module import change_equations as isg_change_equations
from ifnb_module import change_equations as ifnb_change_equations
plt.style.use("~/IFN_paper/src/theme_bw.mplstyle")

def get_params(file):
    params = {}
    with open(file) as f:
        for line in f:
            (key, val) = line.split(",")
            params[key] = float(val)
    return params

def read_inputs(protein, stimulus):
    filename = "../simulation/%s_timecourse.csv" % protein
    df = pd.read_csv(filename)
    
    time = df["Time"].values
    val = df[stimulus].values

    curve = np.array([time, val])
    return curve

def get_input(curve, t, input_name=""):
    max_time = curve[0,-1]
    
    if t < 0:
        return 0
    elif t > max_time:
        if input_name != "p50":
          val = curve[1,-1] * np.exp(-(t - max_time + 1)/60)
        else:
          val = curve[1,-1]
    else:
        f = interp1d(curve[0], curve[1])
        val = f([t])[0]
    return val

def get_inputs(N_curve, I_curve, P_curve, t):
    inputs = {}
    inputs["nfkb"] = get_input(N_curve, t)
    inputs["irf"] = get_input(I_curve, t)
    inputs["p50"] = get_input(P_curve, t)
    return inputs

def IFN_model(t, states, params, stim_data):
    N_curve, I_curve, P_curve = stim_data
    inputs = get_inputs(N_curve, I_curve, P_curve, t)

    # States: IFNb, IFNAR, IFNAR*, ISGF3, ISGF3*, ISG mRNA
    ifnb_module = ifnb_change_equations(t, states[0:1], params, inputs)
    inputs["ifnb"] = states[0]
    ifnar_module = ifnar_change_equations(t, states[1:5], params, inputs)
    inputs["isgf3"] = states[4]
    isg_module = isg_change_equations(t, states[5:6], params, inputs)
    return np.concatenate((ifnb_module, ifnar_module, isg_module))

def run_model(t_span, states0, params, t_eval=None, stim_data=None):
    states = solve_ivp(IFN_model, t_span, states0, args=(params, stim_data), t_eval=t_eval)
    return states

# def get_steady_state(t_span, states0, params, stim_data=None):
#     # print("Max NFkB value: %.4f" % np.max(stim_data[0]))
#     state_names = ["IFNb", "IFNAR", "IFNAR*", "ISGF3", "ISGF3*", "ISG mRNA"]
#     # print("Starting IFNb value: %.4f" % states0[0])
#     states = run_model(t_span, states0, params, stim_data=stim_data)
#     states_ss = states.y
#     states_t = states.t
#     difference = 1
#     i = 1
#     # print("First round IFNb value: %.4f" % states.y[0,-1])
#     while difference > 0.01:
#         states = run_model(t_span, states.y[:,-1], params, stim_data=stim_data)
#         states_ss = np.concatenate((states_ss, states.y), axis=1)
#         states_t = np.concatenate((states_t, states.t + states_t[-1]), axis=0)
#         difference = np.max(np.abs(states.y[0:-2,-1] - states.y[0:-2,-2]))
#         # print("Change in ISG mRNA: %.4f" % (states.y[-1,-1] - states.y[-1,-2]))
#         # print("IFNb value: %.4f" % states.y[0,-1])
#         i += 1
#         if i > 200:
#             max_diff_state = np.argmax(np.abs(states.y[0:-2,-1] - states.y[0:-2,-2]))
#             print("No steady state found after %.2f hours. Max difference = %.4f, occuring for %s" % (t_span[1]*i/60, difference, state_names[max_diff_state]))
#             break
#     final_time = t_span[1]*i

#     return states.y[:,-1], final_time, states_t, states_ss

def get_steady_state(states0, pars, stim_data_ss, t_eval):
    state_names = ["IFNb", "IFNAR", "IFNAR*", "ISGF3", "ISGF3*", "ISG mRNA"]
    end_time = t_eval[-1]

    # print("Starting first iteration", flush=True)
    states0 = solve_ivp(IFN_model, [0, end_time], states0, t_eval=t_eval, args=(pars, stim_data_ss))
    # print("Finished first iteration", flush=True)
    states0 = states0.y
    diff = np.max(np.abs(states0[:,-1] - states0[:,0]))
    i = 0
    while diff > 0.01:
        # print("Difference = %.4f after %d iterations" % (diff, i+1))
        states0 = solve_ivp(IFN_model, [0, end_time], states0[:,-1], t_eval=t_eval, args=(pars, stim_data_ss))
        states0 = states0.y
        diff = np.max(np.abs(states0[:,-1] - states0[:,0]))
        i += 1
        if i > 100:
            max_diff_state = np.argmax(np.abs(states0[:,-1] - states0[:,0]))
            print("No steady state found after %.2f hours. Max difference = %.4f, occuring for %s" % (end_time*i/60, diff, state_names[max_diff_state]))
            break
    states0 = states0[:,-1]
    print("Steady state values found after %.2f hours" % (end_time*i/60))
    return states0

def scale_inputs():
    pass

def full_simulation(states0, pars, name, stimulus, genotype, directory, stim_time = 60*8, stim_data=None, plot = True):
    name = "%s_%s_%s" % (name, stimulus, genotype)

    if stim_data is None:
        # Inputs
        if stimulus in ["CpG", "LPS", "pIC"]:
            I_curve = read_inputs("IRF", stimulus)
            N_curve = read_inputs("NFkB", stimulus)
        elif stimulus == "other":
            N_curve, I_curve, P_curve = stim_data
        else:
            raise ValueError("Stimulus must be CpG, LPS, pIC, or other")

        if genotype in ["WT", "p50KO"]:
            P_values = {"WT": 1, "p50KO": 0}
            P_curve_vals = [P_values[genotype] for i in range(stim_time+180)]
            P_curve = np.array([np.arange(stim_time+180), P_curve_vals])
        elif genotype == "other":
            P_curve = stim_data[2]
        else:
            raise ValueError("Genotype must be WT, p50KO, or other")

        stim_data = [N_curve, I_curve, P_curve]
    else:
        N_curve, I_curve, P_curve = stim_data
        stim_data = [N_curve, I_curve, P_curve]

    stim_data_ss = [[0.00001 for i in range(stim_time+120)] for i in range(2)]
    stim_data_ss = [stim_data_ss[0], stim_data_ss[1], P_curve]

    if plot:
        # Plot inputs
        input_t = np.linspace(0, stim_time+120, stim_time+120+1)
        input_N = [get_input(N_curve, t) for t in input_t]
        input_I = [get_input(I_curve, t) for t in input_t]
        input_P = [get_input(P_curve, t) for t in input_t]

        fig, ax = plt.subplots(1,3)
        if genotype == "WT":
            for i in range(3):
                ax[i].set_prop_cycle(plt.cycler("color", ["k"]))
        fig.set_size_inches(12,4)
        ax[0].plot(input_N)
        ax[0].set_title("N")
        ax[1].plot(input_I)
        ax[1].set_title("I")
        ax[2].plot(input_P)
        ax[2].set_title("P")
        for i in range(3):
            ax[i].set_ylim([-0.01, 1.01])
        plt.suptitle("Input curves for %s stimulation" % name)
        plt.savefig("%s/input_curves_%s.png" % (directory, name))
        plt.close()

    ## Simulate model
    t_eval = np.linspace(0, stim_time+120, stim_time+120+1)

    # Get steady state
    states0 = get_steady_state(states0, pars, stim_data_ss, t_eval)
    print("Steady state IFNb values for %s: %s" % (name, states0), flush=True)

    # Integrate model
    states = solve_ivp(IFN_model, [0, t_eval[-1]], states0, t_eval=t_eval, args=(pars, stim_data))
    return states, t_eval, stim_data

# def plot_model(states, labels, t, filename, title="", xlabel="Time", ylabel="Concentration"):
#     color = plt.cm.viridis(np.linspace(0,1,len(states)))
#     # color = np.flip(color, axis=0)
#     fig = plt.figure()
#     for state, label in zip(states, labels):
#         plt.plot(t, state, label=label, color=color[labels.index(label)])
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.title(title)
#     fig.legend(bbox_to_anchor=(1.2,0.5))
#     plt.savefig("%s.png" % filename, bbox_inches="tight")
