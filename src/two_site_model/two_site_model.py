import numpy as np
import matplotlib.pyplot as plt

class two_site:
    def __init__(self, model=None, t_vals=None, k=None, h_pars=None, C=1):
        t_pars = {"IRF":[0,1,0,1],
                  "NFkB":[0,0,1,1],
                  "AND":[0,0,0,1],
                  "OR":[0,1,1,1]}
        self.ki = 1
        self.kn = 1
        if model in t_pars.keys():
            self.t = t_pars[model]
        else:
            self.t = np.zeros(4)
            self.t[1] = t_vals[0]
            self.t[2] = t_vals[1]
            self.t[3] = 1
            if k is not None:
                self.ki = k[0]
                self.kn = k[1]

        if h_pars is not None:
            if type(h_pars) in [int, float]:
                self.hi = h_pars - 1
                self.hn = 0
            else:
                if len(h_pars) != 2:
                    print("Got %d pars when expected %d" % (len(h_pars), 2))
                    print("pars = " + str(h_pars))
                    raise ValueError("Incorrect number of h parameters in input")
                self.hi = h_pars[0] - 1
                self.hn = h_pars[1] - 1
        self.C = C

    def calculateBeta(self, I, N=None):
        if self.hn ==0:
            N = 1
        elif N is None:
            raise ValueError("Need to provide N if model has k_n != 0")
        self.beta = np.ones(4)
        self.beta[1] = self.ki * (I ** self.hi)
        self.beta[2] = self.kn * (N ** self.hn)
        self.beta[3] = self.ki * (I ** self.hi) * self.kn * (N ** self.hn) * self.C
    def calculateState(self, N, I):
        self.state = np.array([1, I, N, I*N])
        # print(self.state)
    def calculateF(self):
        self.f = np.dot(np.transpose(self.state),(self.beta * self.t)) / np.dot(np.transpose(self.state), self.beta)
        # print(self.f)
    def calculateProb(self):
        self.prob = self.state * self.beta / np.dot(np.transpose(self.state), self.beta)

def get_f(N, I, model=None, t=None, k=None, h=0, C=1):
    m = two_site(model, t, k, h, C)
    m.calculateBeta(I, N)
    m.calculateState(N, I)
    m.calculateF()
    return m.f

def plot_contour(f_values, model_name, I, N, dir, name, condition ="", normalize=True):
    if normalize:
        f_values = f_values / np.max(f_values)
    fig=plt.figure()
    plt.contourf(I,N, f_values, 100, cmap="RdYlBu_r")
    plt.grid(False)
    plt.colorbar(format="%.1f")
    plt.title("Model %s contour plot %s" % (model_name, condition))
    fig.gca().set_ylabel(r"$NF\kappa B$")
    fig.gca().set_xlabel(r"$IRF$")
    plt.savefig("%s/contour_plot_%s.png" % (dir,name))
    plt.close()


def get_state_prob(N, I, model=None, t=None, k=None, h=0, C=1):
    model = two_site(model, t, k, h, C)
    model.calculateBeta(I, N)
    model.calculateState(N, I)
    model.calculateProb()
    probabilities = model.prob
    
    state_names = ["none", "IRF", r"$NF\kappa B$", r"IRF and $NF\kappa B$"]

    return probabilities, state_names

def get_contribution(N, I, model=None, t=None, k=None, h=0, C=1):
    model = two_site(model, t, k, h, C)

    probabilties, state_names = get_state_prob(N, I, model=model, t=t, k=k, h=h, C=C)
    f_contributions = model.t * probabilties
    return f_contributions, state_names