import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from ifnar_module import change_equations as ifnar_change_equations
from isg_module import change_equations as isg_change_equations
from ifnb_module import change_equations as ifnb_change_equations
plt.style.use("~/IFN_paper/src/theme_bw.mplstyle")

def get_input(curve, t):
    # Curve is a list of values where position in list is time in minutes
    # t is the current time
    # interpolate between values in curve to get input at time t
    if t < 0:
        return 0
    elif t > len(curve) - 1:
        return 0.01
    else:
        f = interp1d(range(len(curve)), curve)
        val = f([t])[0]
    # if val > 0:
    #     print("Input at t=%.4f is %.4f" % (t, val))
    return val

def get_inputs(N_curve, I_curve, P_curve, t):
    inputs = {}
    inputs["nfkb"] = get_input(N_curve, t)
    inputs["irf"] = get_input(I_curve, t)
    inputs["p50"] = get_input(P_curve, t)
    # for key, val in inputs.items():
    #     if val > 0:
    #         print("%s = %.4f at t=%.4f" % (key, val, t))
    return inputs

   
def get_params(file):
    params = {}
    with open(file) as f:
        for line in f:
            (key, val) = line.split(",")
            params[key] = float(val)
    return params

def IFN_model(t, states, params, stim_data):
    N_curve, I_curve, P_curve = stim_data
    inputs = get_inputs(N_curve, I_curve, P_curve, t)
    # print any non-zero inputs
    # for key, val in inputs.items():
    #     if val > 0:
    #         print("%s = %.4f at t=%.4f" % (key, val, t))

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
    # print("Max NFkB value: %.4f" % np.max(stim_data[0]))
    state_names = ["IFNb", "IFNAR", "IFNAR*", "ISGF3", "ISGF3*", "ISG mRNA"]
    # print("Starting IFNb value: %.4f" % states0[0])
    states = run_model(t_span, states0, params, stim_data=stim_data)
    states_ss = states.y
    states_t = states.t
    difference = 1
    i = 1
    # print("First round IFNb value: %.4f" % states.y[0,-1])
    while difference > 0.01:
        states = run_model(t_span, states.y[:,-1], params, stim_data=stim_data)
        states_ss = np.concatenate((states_ss, states.y), axis=1)
        states_t = np.concatenate((states_t, states.t + states_t[-1]), axis=0)
        difference = np.max(np.abs(states.y[0:-2,-1] - states.y[0:-2,-2]))
        # print("Change in ISG mRNA: %.4f" % (states.y[-1,-1] - states.y[-1,-2]))
        # print("IFNb value: %.4f" % states.y[0,-1])
        i += 1
        if i > 200:
            max_diff_state = np.argmax(np.abs(states.y[0:-2,-1] - states.y[0:-2,-2]))
            print("No steady state found after %.2f hours. Max difference = %.4f, occuring for %s" % (t_span[1]*i/60, difference, state_names[max_diff_state]))
            break
    final_time = t_span[1]*i

    return states.y[:,-1], final_time, states_t, states_ss

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
