import numpy as np
import matplotlib.pyplot as plt

class three_site_model:
    def __init__(self, pars, model):
        i=0
        self.K=1
        self.C=1
        if model == "B2":
            self.K = pars[0]
            i+=1
        elif model == "B3":
            self.C = pars[0]
            i+=1
        elif model == "B4":
            self.K = pars[0]
            self.C = pars[1]
            i+=2

        if len(pars[i:len(pars)]) != 6:
            print("model= " + model)
            print("i= " + str(i))
            print("pars = " + str(pars), "length= " + str(len(pars)))
            print("K= " + str(self.K))
            raise ValueError("Incorrect number of parameters in input")

        self.parsT = pars[i:len(pars)]
        if len(self.parsT) != 6:
            print("i= " + str(i))
            print("parsT = " + str(self.parsT))
            print("K= " + str(self.K))
            raise ValueError("Incorrect number of parameters for class")
	
        self.beta = np.array([1 for i in range(8)])

        self.beta[2] = self.K
        self.beta[4] = self.K * self.C
        self.beta[6] = self.K
        self.beta[7] = self.K * self.C

        self.t = np.hstack((0, self.parsT, 1))

    def calculateState(self, N, I):
        self.state = np.array([1, I, I, N, I**2, I*N, I*N, I**2*N])

    def calculateF(self):
        self.f = np.dot(np.transpose(self.state),(self.beta * self.t)) / np.dot(np.transpose(self.state), self.beta)

def explore_model_three_site(pars, N, I, model):
    model = three_site_model(pars, model)
    model.calculateState(N, I)
    model.calculateF()
    return model.f

def plot_contour(f_values, model_name, I, N, dir,name, normalize=True):
    if normalize:
        f_values = f_values / np.max(f_values)
    fig=plt.figure()
    plt.contourf(I,N, f_values, 100, cmap="RdYlBu_r")
    plt.grid(False)
    plt.colorbar(format="%.1f")
    plt.title("Model %s best fit" % model_name)
    fig.gca().set_ylabel(r"$NF\kappa B$")
    fig.gca().set_xlabel(r"$IRF$")
    plt.savefig("%s/contour_plot_%s.png" % (dir,name))
    plt.close()

def calculateFvalues(model_name, pars, I, N):
    f_values = np.zeros((len(N), len(I)))
    for n in range(len(N)):
        for i in range(len(I)):
            f_values[n,i] = explore_model_three_site(pars,N[n], I[i],model_name)
    return f_values