import numpy as np
import matplotlib.pyplot as plt

class Modelp50:
	def __init__(self, t_pars, k1=1, k2=1, kp=1):
		self.k1=k1
		self.k2=k2
		self.kp=kp
		num_t_pars = 2
		
		if len(t_pars) != num_t_pars:
			print("Model got %d pars when expected %d" % (len(t_pars), num_t_pars))
			print("pars = " + str(t_pars))
			raise ValueError("Incorrect number of parameters in input")

		self.parsT = t_pars

		# beta will contain: 1, k_1, k_2, k_p, k1*k_p, k1*k2
		self.beta = np.array([1.0 for i in range(6)])
		self.beta[1] = self.k1
		self.beta[2] = self.k2
		self.beta[3] = self.kp
		self.beta[4] = self.k1*self.kp
		self.beta[5] = self.k1*self.k2
		
		
		# Self t array will contain: 0, t1, t2, 0, t1, 1
		# where t1 is the 0th element of self.parsT, etc
		self.t = np.array([0.0 for i in range(6)])
		self.t[1] = self.parsT[0]
		self.t[2] = self.parsT[1]
		self.t[4] = self.parsT[0]
		self.t[5] = 1.0

	def calculateState(self, N, I, P=0):
		# state is 1, I, I, p, I*p, I*I
		self.state = np.array([1, I, I, P, I*P, I*I])

	def calculateF(self):
		self.f = np.dot(np.transpose(self.state),(self.beta * self.t)) / np.dot(np.transpose(self.state), self.beta)
		
def get_state_prob(t_pars, k_pars, N, I, P):
	k1, k2, kp = k_pars
	model = Modelp50(t_pars, k1, k2, kp)
	model.calculateState(N, I, P)
	binding_amount = model.state * model.beta
	probabilities = binding_amount / np.sum(binding_amount)

	state_names = ["none", "IRF", r"$IRF_G$", r"p50", r"$IRF$ \cdot p50", r"$IRF \cdot IRF_G$"]

	return probabilities, state_names

def get_f(t_pars, k_pars, N, I, P, scaling=False):
	k1, k2, kp = k_pars
	model = Modelp50(t_pars, k1, k2, kp)
	model.calculateState(N, I, P)
	model.calculateF()
	if scaling:
		m2 = Modelp50(t_pars, k1, k2, kp)
		m2.calculateState(0.75, 0.5, 1) # Normalize to WT pIC value
		m2.calculateF()
		return model.f / m2.f
	else:
		return model.f
	
def plot_contour(f_values, I, N, dir, name, condition ="", normalize=True, cbar_scaling=True):
	model_name = "p50"
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