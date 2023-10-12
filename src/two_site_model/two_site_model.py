import numpy as np
import matplotlib.pyplot as plt

class two_site:
    def __init__(self, C, model):
        t_pars = {"IRF":[0,1,0,1],
                  "NFkB":[0,0,1,1],
                  "AND":[0,0,0,1],
                  "OR":[0,1,1,1],}
        self.t = t_pars[model]
        self.beta = np.array([1 for i in range(4)])
        self.beta[3] = C
    def calculateState(self, N, I):
        self.state = np.array([1, I, N, I*N])
        # print(self.state)
    def calculateF(self):
        self.f = np.dot(np.transpose(self.state),(self.beta * self.t)) / np.dot(np.transpose(self.state), self.beta)
        # print(self.f)

def explore_two_site(C, model, N, I):
    m = two_site(C, model)
    m.calculateState(N, I)
    m.calculateF()
    return m.f

def get_f(C, N, I, model_name):
    m = two_site(C, model_name)
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