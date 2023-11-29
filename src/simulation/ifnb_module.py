import numpy as np
from ifnb_model import *


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
    t_half_ifnb = pars["t_half_ifnb"]
    p_deg_ifnb = np.log(2) / t_half_ifnb

    # Unpack inputs
    nfkb = inputs["nfkb"]
    irf = inputs["irf"]
    p50 = inputs["p50"]

    # Calculate derivatives
    f = get_f(t_pars, K, C, nfkb, irf, p50)
    difnb = f * p_syn_ifnb - p_deg_ifnb * ifnb
    return [difnb]