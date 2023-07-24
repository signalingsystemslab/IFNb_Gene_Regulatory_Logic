import numpy as np

def change_equations(t, states, pars, inputs):
    # Unpack states
    isg = states[0]

    # Unpack pars
    p_syn_isg = pars["p_syn_isg"]
    p_deg_isg = pars["p_deg_isg"]

    # Unpack inputs
    isgf3 = inputs["isgf3"]

    # Calculate derivatives
    disg = p_syn_isg * isgf3 - p_deg_isg * isg
    return [disg]