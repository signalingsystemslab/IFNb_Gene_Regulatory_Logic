import numpy as np

def change_equations(t, states, pars, inputs):
   # Unpack states
    ifnb_prot = states[0]

    # Unpack pars
    p_trans_ifnb = pars["p_trans_ifnb"]
    t_half_ifnb_prot = pars["t_half_ifnb_prot"]
    p_deg_ifnb_prot = np.log(2) / t_half_ifnb_prot

    # Unpack inputs
    ifnb = inputs['ifnb_rna']

    # Calculate derivatives
    difnb_prot = p_trans_ifnb * ifnb - p_deg_ifnb_prot * ifnb_prot
    return [difnb_prot]