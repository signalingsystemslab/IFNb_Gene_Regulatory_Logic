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
    p_deg_ifnb = 1/pars["p_tau_ifnb"]

    # Unpack inputs
    nfkb = inputs["nfkb"]
    irf = inputs["irf"]
    p50 = inputs["p50"]


    # Calculate derivatives
    f = get_f(t_pars, K, C, nfkb, irf, p50)
    difnb = f * p_syn_ifnb - p_deg_ifnb * ifnb
    # if t< 60:
    #     print("T_pars: %s\tK: %s\tC: %s\tNFKB: %s\tIRF: %s\tP50: %s" % (t_pars, K, C, nfkb, irf, p50))
    #     print("IFNb production rate: %.4f at t=%.4f" % (f, t))
    #     print("dIFNb/dt: %.4f" % difnb)
    return [difnb]