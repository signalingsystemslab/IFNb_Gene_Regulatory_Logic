# Three site model, no p50 competition
import numpy as np
import matplotlib.pyplot as plt

class three_site:
    def __init__(self, t_pars, k_pars=None, c_par=None, h_pars=None):
        self.parsT = t_pars
        
        if len(t_pars) != 4:
            # print("Got %d t pars when expected %d" % (len(t_pars), 5))
            # print("pars = " + str(t_pars))
            raise ValueError("Incorrect number of t parameters in input (%d instead of 4)" % len(t_pars))

        if k_pars is not None:
            if len(k_pars) != 3:
                # print("Got %d k pars when expected %d" % (len(k_pars), 3))
                # print("pars = " + str(k_pars))
                raise ValueError("Incorrect number of k parameters in input (%d instead of 3)" % len(k_pars))

            self.k1 = k_pars[0]
            self.k2 = k_pars[1]
            self.kn = k_pars[2]
        else:
            self.k1 = 1
            self.k2 = 1
            self.kn = 1

        if c_par is not None:
            # functional cooperativity between IRF and NFkB
            self.c = c_par
        else:
            self.c = 1

        self.h1 = 0
        self.h2 = 0
        self.hn = 0

        if h_pars is not None:
            # Hill coefficient for each IRF binding event
            if type(h_pars) in [int, float]:
                self.h1 = h_pars - 1
                self.h2 = h_pars - 1
            else:
                if len(h_pars) == 2:
                    self.h1 = h_pars[0] - 1
                    self.h2 = h_pars[1] - 1
                elif len(h_pars) == 3:
                    self.h1 = h_pars[0] - 1
                    self.h2 = h_pars[1] - 1
                    self.hn = h_pars[2] - 1
                else:
                    print("Got %d pars when expected %d" % (len(h_pars), 2))
                    print("pars = " + str(h_pars))
                    raise ValueError("Incorrect number of h parameters in input")


        # States
        # 1, I, Ig, N,  I*Ig, I*N, Ig*N, I*Ig*N
        # t-vals
        # 0, t1, t1, t2, t3, t1+t2, t4, 1

        # Force t for both IRFs to be the same
        self.t = np.array([0.0 for i in range(8)])
        self.t[1] = self.parsT[0] # t1 - IRF
        self.t[2] = self.parsT[0] # t1 - IRF_G
        self.t[3] = self.parsT[1] # t2 - NFkB
        self.t[4] = self.parsT[2] # t3 - IRF + IRF_G
        self.t[5] = self.parsT[0] + self.parsT[1] # t1 + t2 - IRF + NFkB
        self.t[6] = self.parsT[3] # t4 - IRF_G + NFkB
        self.t[7] = 1 #  IRF + IRF_G + NFkB


    def calculateBeta(self, I, N=None):
        self.beta = np.array([1.0 for i in range(8)])
        
        self.beta[1] = self.k1 * (I ** self.h1)
        self.beta[2] = self.k2 * (I ** self.h2)
        self.beta[3] = self.kn * (N ** self.hn)
        self.beta[4] = self.k1 * (I ** self.h1) * self.k2 * (I ** self.h2)
        self.beta[5] = self.k1 * (I ** self.h1) * self.kn * (N ** self.hn)
        self.beta[6] = self.k2 * (I ** self.h2) * self.kn * (N ** self.hn)
        self.beta[7] = self.k1 * (I ** self.h1) * self.k2 * (I ** self.h2) * self.kn * (N ** self.hn)

    def calculateState(self, N, I):
        if not hasattr(self, 'beta'):
            self.calculateBeta(I, N)
        self.state = np.array([1, I, I, N, I*I, I*N, I*N, I*I*N])

    def calculateProb(self):
        self.prob = self.state * self.beta / np.dot(np.transpose(self.state), self.beta)


    def calculateF(self):
        self.f = np.dot(np.transpose(self.prob), self.t)

def get_f(t_pars, k_pars, N, I, c_par=None, h_pars=None, scaling=False):
    model = three_site(t_pars, k_pars, c_par=c_par, h_pars=h_pars)
    model.calculateState(N, I)
    model.calculateProb()
    model.calculateF()
    if scaling:
        m2 = three_site(t_pars, k_pars)
        m2.calculateState(0.75, 0.5, 1) # Normalize to WT pIC value
        m2.calculateF()
        return model.f / m2.f
    else:
        return model.f

def get_state_prob(t_pars, k_pars, N, I, c_par=None, h_pars=None):
    model = three_site(t_pars, k_pars, c_par=c_par, h_pars=h_pars)
    model.calculateState(N, I)
    model.calculateProb()
    probabilities = model.prob
    
    # 1, I, Ig, N,  I*Ig, I*N, Ig*N, I*Ig*N
    state_names = ["none", r"$IRF$", r"$IRF_G$", r"$NF\kappa B$", r"$IRF\cdot IRF_G$", 
                   r"$IRF\cdot NF\kappa B$", r"$IRF_G\cdot NF\kappa B$", r"$IRF\cdot IRF_G\cdot NF\kappa B$"]

    return probabilities, state_names

def get_contribution(t_pars, k_pars, N, I, c_par=None, h_pars=None):
    # Returns relative contribution of each state to f
    model = three_site(t_pars, k_pars, c_par=c_par, h_pars=h_pars)
    probabilties, state_names = get_state_prob(t_pars, k_pars, N, I, c_par=c_par, h_pars=h_pars)
    f_contributions = model.t * probabilties
    return f_contributions, state_names

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

def three_site_objective(pars, *args):
    """Minimization objective function for the three site model.
    Args:
    pars: array of parameters
    args: tuple of (N, I, IFNb_data)
    kwargs: additional parameters (c, h)
    """

    N, I, beta, c_par, h_pars = args
    t_pars = pars[0:5]
    k_pars = pars[5:8]

    num_pts = len(N)
    
    f_list = [get_f(t_pars, k_pars, N[i], I[i], c_par=c_par, h_pars=h_pars) for i in range(num_pts)] 
    residuals = np.array(f_list) - beta
    
    rmsd = np.sqrt(np.mean(residuals**2))
    rmsd_scaled = rmsd * 1000
    return rmsd_scaled