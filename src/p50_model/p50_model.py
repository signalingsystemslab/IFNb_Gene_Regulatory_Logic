import numpy as np
import matplotlib.pyplot as plt

class Modelp50:
    def __init__(self, t_pars, k_pars=None, c_par=None, h_pars=None):
        self.parsT = t_pars
        
        if len(t_pars) != 5:
            # print("Got %d pars when expected %d" % (len(t_pars), 5))
            # print("pars = " + str(t_pars))
            raise ValueError("Got %d pars when expected %d. Pars = %s" % (len(t_pars), 5, str(t_pars)))

        if k_pars is not None:
            if len(k_pars) != 4:
                print("Got %d pars when expected %d" % (len(k_pars), 4))
                print("pars = " + str(k_pars))
                raise ValueError("Incorrect number of k parameters in input")

            self.k1 = k_pars[0]
            self.k2 = k_pars[1]
            self.kn = k_pars[2]
            self.kp = k_pars[3]
        else:
            self.k1 = 1
            self.k2 = 1
            self.kn = 1
            self.kp = 1

        if c_par is not None:
            # functional cooperativity between IRF and NFkB
            self.c = c_par
        else:
            self.c = 1

        if h_pars is not None:
            # Hill coefficient for each IRF binding event
            if type(h_pars) in [int, float]:
                self.h1 = h_pars - 1
                self.h2 = h_pars - 1
            else:
                if len(h_pars) != 2:
                    print("Got %d pars when expected %d" % (len(h_pars), 2))
                    print("pars = " + str(h_pars))
                    raise ValueError("Incorrect number of h parameters in input")
                self.h1 = h_pars[0] - 1
                self.h2 = h_pars[1] - 1


        # States
        # 1, I, Ig, I*P, N, N*P, P, I*Ig, I*N, Ig*N, I*N*P, I*Ig*N
        # I*N is sum of I and N

        self.t = np.array([0.0 for i in range(12)])
        self.t[1] = self.parsT[0] # IRF - t1
        self.t[2] = self.parsT[1] # IRF_G - t2
        self.t[3] = self.parsT[0] # IRF + p50 - t1
        self.t[4] = self.parsT[2] # NFkB - t3
        self.t[5] = self.parsT[2] # NFkB + p50 - t3
        # 6 is zero
        self.t[7] = self.parsT[3] # IRF + IRF_G - t4
        self.t[8] = self.parsT[0] + self.parsT[2] # IRF + NFkB - t1 + t3
        self.t[9] = self.parsT[4] # IRF_G + NFkB - t5
        self.t[10] = self.parsT[0] + self.parsT[2] # IRF + NFkB + p50 - t1 + t3
        self.t[11] = 1


    def calculateBeta(self, I):
        self.beta = np.array([1.0 for i in range(12)])
        
        self.beta[1] = self.k1 * (I ** self.h1)
        self.beta[2] = self.k2 * (I ** self.h2)
        self.beta[3] = self.k1 * (I ** self.h1) * self.kp
        self.beta[4] = self.kn
        self.beta[5] = self.kn * self.kp
        self.beta[6] = self.kp
        self.beta[7] = self.k1 * (I ** self.h1) * self.k2 * (I ** self.h2) * self.c
        self.beta[8] = self.k1 * (I ** self.h1) * self.kn
        self.beta[9] = self.k2 * (I ** self.h2) * self.kn
        self.beta[10] = self.k1 * (I ** self.h1) * self.kn * self.kp
        self.beta[11] = self.k1 * (I ** self.h1) * self.k2 * (I ** self.h2) * self.kn * self.c

    def calculateState(self, N, I, P=1):
        if not hasattr(self, 'beta'):
            self.calculateBeta(I)
        self.state = np.array([1, I, I, I*P, N, N*P, P, I**2, I*N, I*N, I*N*P, I**2*N])

    def calculateProb(self):
        self.prob = self.state * self.beta / np.dot(np.transpose(self.state), self.beta)

    # def calculateF(self):
    #     self.f = np.dot(np.transpose(self.state),(self.beta * self.t)) / np.dot(np.transpose(self.state), self.beta)

    def calculateF(self):
        self.f = np.dot(np.transpose(self.prob), self.t)

def get_f(t_pars, k_pars, N, I, P, c_par=None, h_pars=None, scaling=False):
   model = Modelp50(t_pars, k_pars, c_par=c_par, h_pars=h_pars)
   model.calculateState(N, I, P)
   model.calculateProb()
   model.calculateF()
   if scaling:
       m2 = Modelp50(t_pars, k_pars)
       m2.calculateState(0.75, 0.5, 1) # Normalize to WT pIC value
       m2.calculateF()
       return model.f / m2.f
   else:
       return model.f

def get_state_prob(t_pars, k_pars, N, I, P, c_par=None, h_pars=None):
    model = Modelp50(t_pars, k_pars, c_par=c_par, h_pars=h_pars)
    model.calculateState(N, I, P)
    model.calculateProb()
    probabilities = model.prob
    
    # 1, I, Ig, I*P, N, N*P, P, I*Ig, I*N, Ig*N, I*N*P, I*Ig*N
    state_names = ["none", r"$IRF$", r"$IRF_G$", r"$IRF\cdot p50$", r"$NF\kappa B$", 
                r"$NF\kappa B\cdot p50$", r"$p50$", r"$IRF\cdot IRF_G$", 
                r"$IRF\cdot NF\kappa B$", r"$IRF_G\cdot NF\kappa B$", r"$IRF\cdot NF\kappa B\cdot p50$", 
                r"$IRF\cdot IRF_G\cdot NF\kappa B$"]

    return probabilities, state_names

def get_contribution(t_pars, k_pars, N, I, P=1, c_par=None, h_pars=None):
    # Returns relative contribution of each state to f
    model = Modelp50(t_pars, k_pars, c_par=c_par, h_pars=h_pars)
    probabilties, state_names = get_state_prob(t_pars, k_pars, N, I, P, c_par=c_par, h_pars=h_pars)
    f_contributions = model.t * probabilties
    return f_contributions, state_names

# def calc_irf_state_prob(I,k1,k2,kp,P):
#     irf_prob = (I*k1)/(1+I*k1)
#     irfg_prob = (I*k2)/(1+I*k2+kp*P)
#     irf_tot_prob = I*(k1+k2+I*k1*k2+k1*kp*P)/((1+I*k1)*(1+I*k2+kp*P))
#     return irf_prob, irfg_prob, irf_tot_prob

# def calc_irf_state_prob_hill(I,k1,k2,kp,P,h):
#     # print("I: %f, k1: %f, k2: %f, kp: %f, P: %f, h: %f" % (I, k1, k2, kp, P, h))
#     irf_prob=(I**(1+h)*k1*(1+I*k2+kp*P))/(1+I**(2+h)*k1*k2+kp*P+I**(1+h)*(k1+k2+k1*kp*P))
#     irfg_prob=(I**(1+h)*(1+I*k1)*k2)/(1+I**(2+h)*k1*k2+kp*P+I**(1+h)*(k1+k2+k1*kp*P))
#     irf_tot_prob=(I**(1+h)*(k1+k2+I*k1*k2+k1*kp*P))/(1+I**(2+h)*k1*k2+kp*P+I**(1+h)*(k1+k2+k1*kp*P))
#     return irf_prob, irfg_prob, irf_tot_prob

# def get_irf_state_prob(t_pars, k_pars, N, I, P, c_par=None, h_pars=None):
#     if c_par is None and h_pars is None:
#         irf_prob, irfg_prob, irf_tot_prob = calc_irf_state_prob(I, k_pars[0], k_pars[1], k_pars[3], P)
#     elif c_par is None and h_pars is not None:
#         irf_prob, irfg_prob, irf_tot_prob = calc_irf_state_prob_hill(I, k_pars[0], k_pars[1], k_pars[3], P, h_pars[0])
#     else:
#         # Return probabilities of each IRF state
#         model = Modelp50(t_pars, k_pars)
#         model.calculateState(N, I, P)
#         binding_amount = model.state * model.beta
#         probabilities = binding_amount / np.sum(binding_amount)
#         irfg_prob = np.sum(probabilities[[2, 7, 9, 11]])
#         irf_prob = np.sum(probabilities[[1, 3, 7, 8, 10, 11]])
#         irf_tot_prob = irf_prob + irfg_prob - probabilities[11] - probabilities[7]
#         return irf_prob, irfg_prob, irf_tot_prob


def plot_contour(f_values, model_name, I, N, dir, name, condition ="", normalize=True, cbar_scaling=True):
    if normalize:
        f_values = f_values / np.max(f_values)
    fig=plt.figure()
    if cbar_scaling:
        plt.contourf(I,N, f_values, 100, cmap="RdYlBu_r", vmin=0, vmax=1)
    else:
        plt.contourf(I,N, f_values, 100, cmap="RdYlBu_r")
    plt.grid(False)
    plt.colorbar(format="%.1f")
    plt.title("Model %s contour plot %s" % (model_name, condition))
    fig.gca().set_ylabel(r"$NF\kappa B$")
    fig.gca().set_xlabel(r"$IRF$")
    plt.savefig("%s/contour_plot_%s.png" % (dir,name))
    plt.close()

def p50_objective(pars, *args):
    """Minimization objective function for the three site model.
    Args:
    pars: array of parameters
    args: tuple of (N, I, IFNb_data)
    kwargs: additional parameters (c, h)
    """

    N, I, P, beta, h_pars = args
    t_pars = pars[0:6]
    k_pars = pars[6:10]

    num_pts = len(N)
    
    f_list = [get_f(t_pars, k_pars, N[i], I[i], P[i], h_pars=h_pars) for i in range(num_pts)] 
    residuals = np.array(f_list) - beta
    
    rmsd = np.sqrt(np.mean(residuals**2))
    return rmsd

