import numpy as np
import matplotlib.pyplot as plt

class Modelp50:
	def __init__(self, t_pars, k_pars=None):
		self.parsT = t_pars

		if len(t_pars) != 6:
			print("Got %d pars when expected %d") % (len(t_pars), 6)
			print("pars = " + str(t_pars))
			raise ValueError("Incorrect number of t parameters in input")

		if k_pars is not None:
			if len(k_pars) != 4:
				print("Got %d pars when expected %d" % (len(k_pars), 4))
				print("pars = " + str(k_pars))
				raise ValueError("Incorrect number of k parameters in input")

			self.k1 = k_pars[0]
			self.k2 = k_pars[1]
			self.kn = k_pars[2]
			self.kp = k_pars[3]

		# beta will contain values from this matrix: \begin{pmatrix} 
		# \begin{pmatrix} 
		# 1 \\ k_{I1} \\ k_{I2} \\ k_{I1} k_p \\ k_N \\ k_N k_p \\ k_p \\ k_{I1} k_{12} \\ k_{I1} k_N \\ k_{I2} k_N \\ k_{I1} k_N k_p \\ k_{I1} k_{12} k_N
		# \end{pmatrix}
		self.beta = np.array([1.0 for i in range(12)])
		
		self.beta[1] = self.k1
		self.beta[2] = self.k2
		self.beta[3] = self.k1 * self.kp
		self.beta[4] = self.kn
		self.beta[5] = self.kn * self.kp
		self.beta[6] = self.kp
		self.beta[7] = self.k1 * self.k2
		self.beta[8] = self.k1 * self.kn
		self.beta[9] = self.k2 * self.kn
		self.beta[10] = self.k1 * self.kn * self.kp
		self.beta[11] = self.k1 * self.k2 * self.kn


		# Self t array will contain: 0, t1, t2, t1, t3, t3, 0, t4, t5, t6, t5, 1 where t1 is 
		# the 0th element of self.parsT, etc
		self.t = np.array([0.0 for i in range(12)])
		self.t[1] = self.parsT[0]
		self.t[2] = self.parsT[1]
		self.t[3] = self.parsT[0]
		self.t[4] = self.parsT[2]
		self.t[5] = self.parsT[2]
		self.t[7] = self.parsT[3]
		self.t[8] = self.parsT[4]
		self.t[9] = self.parsT[5]
		self.t[10] = self.parsT[4]
		self.t[11] = 1


	def calculateState(self, N, I, P=0):
		self.state = np.array([1, I, I, I*P, N, N*P, P, I**2, I*N, I*N, I*N*P, I**2*N])

	def calculateF(self):
		self.f = np.dot(np.transpose(self.state),(self.beta * self.t)) / np.dot(np.transpose(self.state), self.beta)

def get_f(t_pars, k_pars, N, I, P, scaling=False):
	model = Modelp50(t_pars, k_pars)
	model.calculateState(N, I, P)
	model.calculateF()
	if scaling:
		m2 = Modelp50(t_pars, k_pars)
		m2.calculateState(0.75, 0.5, 1) # Normalize to WT pIC value
		m2.calculateF()
		return model.f / m2.f
	else:
		return model.f

def get_state_prob(t_pars, k_pars, N, I, P):
	model = Modelp50(t_pars, k_pars)
	model.calculateState(N, I, P)
	binding_amount = model.state * model.beta
	probabilities = binding_amount / np.sum(binding_amount)
	
	state_names = ["none", "IRF", r"$IRF_G$", r"p50", r"$IRF \cdot p50$", r"$IRF \cdot IRF_G$"]

	return probabilities, state_names

def get_f(t_pars, k_pars, N, I, P, scaling=False):
	model = Modelp50(t_pars, k_pars)
	model.calculateState(N, I, P)
	model.calculateF()
	if scaling:
		m2 = Modelp50(t_pars, k_pars)
		m2.calculateState(0.75, 0.5, 1) # Normalize to WT pIC value
		m2.calculateF()
		return model.f / m2.f
	else:
		return model.f
	
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