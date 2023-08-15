import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from ifnar_module import change_equations as ifnar_change_equations
from isg_module import change_equations as isg_change_equations
from ifnb_module import change_equations as ifnb_change_equations
plt.style.use("~/IFN_paper/src/theme_bw.mplstyle")

def get_input(amp, times, t):
    input = 0
    if times == "All":
        input = amp
    else:
        for i in range(len(times)):
            if (i % 2 == 0):
                if (t > times[i]) and (t < times[i+1]):
                    input = amp
    return input

def get_inputs(N, N_times, I, I_times, P, P_times, t):
    inputs = {}
    if N_times == "All":
        inputs["nfkb"] = N
    elif N_times == "None":
        inputs["nfkb"] = 0
    else:
        inputs["nfkb"] = get_input(N, N_times, t)
    if I_times == "All":
        inputs["irf"] = I
    elif I_times == "None":
        inputs["irf"] = 0
    else:
        inputs["irf"] = get_input(I, I_times, t)
    if P_times == "All":
        inputs["p50"] = P
    elif P_times == "None":
        inputs["p50"] = 0
    else:
        inputs["p50"] = get_input(P, P_times, t)
    return inputs

   
def get_params(file):
    params = {}
    with open(file) as f:
        for line in f:
            (key, val) = line.split(",")
            params[key] = float(val)
    return params

def IFN_model(t, states, params, stim_data):
    if stim_data is None:
        N, N_times, I, I_times, P, P_times = 0, "None", 0, "None", 0, "None"
    else:
        N, N_times, I, I_times, P, P_times = stim_data
    # print("N=%s, I=%s, P=%s" % (N, I, P))
    # print("N_times=%s, I_times=%s, P_times=%s" % (N_times, I_times, P_times))
    inputs = get_inputs(N, N_times, I, I_times, P, P_times, t)
    # print("Inputs: " + str(inputs))

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

def get_steady_state(t_span, states0, params, stim_data=None):
    states = run_model(t_span, states0, params, stim_data=stim_data)
    difference = 1
    i = 1
    while difference > 0.005:
        states = run_model(t_span, states.y[:,-1], params, stim_data=stim_data)
        difference = np.max(np.abs(states.y[0:-2,-1] - states.y[0:-2,-2]))
        # print("Change in ISG mRNA: %.4f" % (states.y[-1,-1] - states.y[-1,-2]))
        i += 1
        if i > 100:
            max_diff_state = np.argmax(np.abs(states.y[0:-2,-1] - states.y[0:-2,-2]))
            print("No steady state found; max difference = %.4f, occuring at state #%d" % (difference, max_diff_state))
            break
    final_time = t_span[1]*i
    return states.y[:,-1], final_time

def plot_model(states, labels, t, filename, title="", xlabel="Time", ylabel="Concentration"):
    color = plt.cm.viridis(np.linspace(0,1,len(states)))
    # color = np.flip(color, axis=0)
    fig = plt.figure()
    for state, label in zip(states, labels):
        plt.plot(t, state, label=label, color=color[labels.index(label)])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    fig.legend(bbox_to_anchor=(1.2,0.5))
    plt.savefig("%s.png" % filename, bbox_inches="tight")
