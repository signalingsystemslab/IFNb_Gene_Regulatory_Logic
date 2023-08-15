import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import os
from ifnar_module import change_equations
plt.style.use("~/IFN_paper/src/theme_bw.mplstyle")

results_dir = "./results/ifnar_model/"
os.makedirs(results_dir, exist_ok=True)

def get_inputs(ifnb):
    inputs = {}
    inputs["ifnb"] = ifnb
    return inputs

def get_params(file):
    params = {}
    with open(file) as f:
        for line in f:
            (key, val) = line.split(",")
            params[key] = float(val)
    return params

def IFN_model(t, states, inputs, params):
    return change_equations(t, states, params, inputs)

def run_model(ifnb, t_span, states0, params, t_eval=None):
    inputs = get_inputs(ifnb)
    states = solve_ivp(IFN_model, t_span, states0, args=(inputs, params), t_eval=t_eval)
    return states

def get_steady_state(ifnb, t_span, states0, params):
    states = run_model(ifnb, t_span, states0, params)
    difference = 1
    i = 1
    while difference > 0.005:
        states = run_model(ifnb, t_span, states.y[:,-1], params)
        difference = np.max(np.abs(states.y[:,-1] - states.y[:,-2]))
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
    plt.close()

def main():
    params = get_params("ifnar_params.csv")
    t = [0,500]
    states0 = [50, 0, 50, 0]
    states_ss, t_ss = get_steady_state(0, t, states0, params)
    print("Steady state: %s \t Time: %s min" % (states_ss, t_ss))

    states = run_model(5, t, states_ss, params)
    state_list = [states.y[i,:] for i in range(states.y.shape[0])]
    labels = [i for i in range(states.y.shape[0])]
    plot_model(state_list, labels, states.t, "ifnar_model_ifnb5")

    t_eval = np.linspace(t[0], t[1], 1000)
    labels = []
    ifnb = [0.1, 0.3, 1, 3, 10, 30, 100]
    state_list_isgf3 = np.zeros((len(t_eval), len(ifnb)))
    state_list_ifnar = np.zeros((len(t_eval), len(ifnb)))
    for i in range(len(ifnb)):
        beta = ifnb[i]
        states = run_model(beta, t, states_ss, params,t_eval)
        # active isgf3 is the 4th state
        state_list_isgf3[:,i] = states.y[3,:]
        # active ifnar is the 2nd state
        state_list_ifnar[:,i] = states.y[1,:]
        labels.append("IFNB = %s nM" % beta)

    print("Max ISGF3 for each IFNB: %s" % np.max(state_list_isgf3, axis=0))

    state_list = [state_list_isgf3[:,i] for i in range(state_list_isgf3.shape[1])]
    plot_model(state_list, labels, t_eval, "%sifnar_model_wide_range_isgf3" % results_dir, r"IFNAR model results with varying IFN$\beta$",
                "Time", "Active ISGF3 (nM)")
    
    state_list = [state_list_ifnar[:,i] for i in range(state_list_ifnar.shape[1])]
    plot_model(state_list, labels, t_eval, "%sifnar_model_wide_range_ifnar" % results_dir, r"IFNAR model results with varying IFN$\beta$",
                "Time", "Active IFNAR (nM)")
    
    ifnb = [14, 62]
    ifnb = np.round(ifnb, 0)
    labels = []
    state_list_isgf3 = np.zeros((len(t_eval), len(ifnb)))
    state_list_ifnar = np.zeros((len(t_eval), len(ifnb)))
    for i in range(len(ifnb)):
        beta = ifnb[i]
        states = run_model(beta, t, states_ss, params,t_eval)
        # active isgf3 is the 4th state
        state_list_isgf3[:,i] = states.y[3,:]
        # active ifnar is the 2nd state
        state_list_ifnar[:,i] = states.y[1,:]
        labels.append("IFNB = %s nM" % beta)

    print("Max ISGF3 for each IFNB: %s" % np.max(state_list_isgf3, axis=0))

    state_list = [state_list_isgf3[:,i] for i in range(state_list_isgf3.shape[1])]
    plot_model(state_list, labels, t_eval, "%sifnar_model_isgf3_14-62" % results_dir, r"IFNAR model results with varying IFN$\beta$",
                "Time", "Active ISGF3 (nM)")
    
    state_list = [state_list_ifnar[:,i] for i in range(state_list_ifnar.shape[1])]
    plot_model(state_list, labels, t_eval, "%sifnar_model_ifnar_14-62" % results_dir, r"IFNAR model results with varying IFN$\beta$",
                "Time", "Active IFNAR (nM)")
    

if __name__ == "__main__":
    main()