import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from ifnar_module import change_equations as ifnar_change_equations
from isg_module import change_equations as isg_change_equations
from ifnb_module import change_equations as ifnb_change_equations
plt.style.use("~/IFN_paper/src/theme_bw.mplstyle")

def get_inputs(N, I, P):
    inputs = {}
    inputs["nfkb"] = N
    inputs["irf"] = I
    inputs["p50"] = P
    return inputs

def get_params(file):
    params = {}
    with open(file) as f:
        for line in f:
            (key, val) = line.split(",")
            params[key] = float(val)
    return params

def IFN_model(t, states, inputs, params):
    ifnb_module = ifnb_change_equations(t, states[0:1], params, inputs)
    inputs["ifnb"] = states[0]
    ifnar_module = ifnar_change_equations(t, states[2:5], params, inputs)
    inputs["isgf3"] = states[4]
    isg_module = isg_change_equations(t, states[5:6], params, inputs)


def run_model(N, I, P, t_span, states0, params, t_eval=None):
    inputs = get_inputs(N,I,P)
    states = solve_ivp(IFN_model, t_span, states0, args=(inputs, params), t_eval=t_eval)
    return states

def get_steady_state(ifnb, t_span, states0, params):
    states = run_model(ifnb, t_span, states0, params)
    difference = 1
    i = 1
    while difference < 0.005:
        states = run_model(ifnb, t_span, states.y[:,-1], params)
        difference = np.min(np.abs(states.y[:,-1] - states.y[:,-2]))
        i += 1
        if i > 100:
            print("No steady state found")
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