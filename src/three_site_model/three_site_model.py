# Three site model, no p50 competition
import numpy as np
import matplotlib.pyplot as plt

class three_site:
    def __init__(self, t_pars, k_pars=None, c_par=None, h_pars=None):
        self.parsT = t_pars
        
        if len(t_pars) != 5:
            print("Got %d pars when expected %d") % (len(t_pars), 5)
            print("pars = " + str(t_pars))
            raise ValueError("Incorrect number of t parameters in input")

        if k_pars is not None:
            if len(k_pars) != 3:
                print("Got %d pars when expected %d" % (len(k_pars), 3))
                print("pars = " + str(k_pars))
                raise ValueError("Incorrect number of k parameters in input")

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
        # 1, I, Ig, N,  I*Ig, I*N, Ig*N, I*Ig*N
        # t-vals
        # 0, t1, t2, t3, t4, t5, t6, 1

        self.t = np.array([0.0 for i in range(8)])
        self.t[1] = self.parsT[0]
        self.t[2] = self.parsT[1]
        self.t[3] = self.parsT[2]
        self.t[4] = self.parsT[3]
        self.t[5] = self.parsT[0] + self.parsT[2]
        self.t[6] = self.parsT[4]
        self.t[7] = 1


    def calculateBeta(self, I):
        self.beta = np.array([1.0 for i in range(8)])
        
        self.beta[1] = self.k1 * (I ** self.h1)
        self.beta[2] = self.k2 * (I ** self.h2)
        self.beta[3] = self.kn
        self.beta[4] = self.k1 * (I ** self.h1) * self.k2 * (I ** self.h2)
        self.beta[5] = self.k1 * (I ** self.h1) * self.kn
        self.beta[6] = self.k2 * (I ** self.h2) * self.kn
        self.beta[7] = self.k1 * (I ** self.h1) * self.k2 * (I ** self.h2) * self.kn

    def calculateState(self, N, I):
        if not hasattr(self, 'beta'):
            self.calculateBeta(I)
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
    args: tuple of (N, I, beta, par_type)
    par_type: string of length 1-4, where each character is one of "k", "c", "h", "t"
    "k" = k_pars, length 3
    "c" = c_par, length 1
    "h" = h_pars, length 2
    "t" = t_pars only
    """

    N, I, beta, par_type = args
    t_pars = pars[0:6]
    c_par = None
    h_pars = None
    k_pars = None

    if par_type not in ["k", "c", "kc", "ck", "t", "h", "kh", "hk"]:
        print("Accepted par_types: k, c, kc, ck, t, h, kh, hk")
        raise ValueError("par_type %s not recognized" % par_type)
    
    if par_type != "t":
        num_options = len(par_type)
        startindex = 6
        if len(pars) < startindex:
            print("Not enough parameters after startindex %d" % startindex)
            raise ValueError("Not enough parameters")

        for i in range(num_options):
            if par_type[i] == "k":
                if len(pars[startindex:]) < 3:
                    print("Not enough parameters for k_pars")
                    raise ValueError("Not enough parameters for k_pars")
                k_pars = pars[startindex:startindex+4]
                startindex += 4
            elif par_type[i] == "c":
                c_par = pars[startindex]
                startindex += 1
            elif par_type[i] == "h":
                if len(pars[startindex:]) < 2:
                    print("Not enough parameters for h_pars")
                    raise ValueError("Not enough parameters for h_pars")
                h_pars = pars[startindex:startindex+2]
                startindex += 2

    num_pts = len(N)
    
    f_list = [get_f(t_pars, k_pars, N[i], I[i], c_par=c_par, h_pars=h_pars) for i in range(num_pts)] 
    residuals = np.array(f_list) - beta
    
    rmsd = np.sqrt(np.mean(residuals**2))
    return rmsd