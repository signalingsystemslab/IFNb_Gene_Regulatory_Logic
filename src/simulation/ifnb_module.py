import numpy as np
from ifnb_model import *


def change_equations(t, states, pars, inputs):
    # Unpack states
    ifnb = states[0]

    # Unpack pars
    t_pars = pars["t_pars"]
    K = pars["K"]
    C = pars["C"]
    scale = pars["scale"]
    p_deg_ifnb = pars["p_deg_ifnb"]

    # Unpack inputs
    nfkb = inputs["nfkb"]
    irf = inputs["irf"]
    p50 = inputs["p50"]


    # Calculate derivatives
    f = get_f(t_pars, K, C, nfkb, irf, p50, scaling=scale)
    difnb = f * ifnb - p_deg_ifnb * ifnb
    return [difnb]