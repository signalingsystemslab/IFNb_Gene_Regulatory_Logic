import numpy as np
import matplotlib.pyplot as plt

class Modelp50:
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
			print("Model %s: got %d pars when expected %d" % (model, len(pars), 6 + i))
			# print("i= " + str(i))
			print("pars = " + str(pars))
			# print("K= " + str(self.K))
			raise ValueError("Incorrect number of parameters in input")

		self.parsT = pars[i:len(pars)]
		if len(self.parsT) != 6:
			print("i= " + str(i))
			print("parsT = " + str(self.parsT))
			print("K= " + str(self.K))
			raise ValueError("Incorrect number of parameters for class")

		# beta will contain values from this matrix: \begin{pmatrix} 
		# 1 \\ k_{I1} \\ k_{I2} \\ k_{I1} k_p \\ k_N \\ k_N k_p \\ k_p \\ k_{I1} k_{12} \\
		#  k_{I1} k_N \\ k_{I2} k_N \\ k_{I1} k_N k_p \\ k_{I1} k_{12} k_N
		# \end{pmatrix}
		self.beta = np.array([1.0 for i in range(12)])
		
		self.beta[2] = self.K
		self.beta[7] = self.K * self.C
		self.beta[9] = self.K
		self.beta[11] = self.K * self.C
		
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

# def explore_modelp50(parsT, N, I, P, model_name):
# 	model = Modelp50(parsT, model_name)
# 	# print(model.parsT)
# 	# print(model.t)
# 	# print("beta = ", model.beta)
# 	model.calculateState(N, I, P)
# 	# print(model.state)
# 	model.calculateF()
# 	# print(model.f)
# 	return model.f

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

def get_f(t_pars, K, C, N, I, P, model_name="B2", scaling=1):
	if model_name == "B1":
		other_pars = []
	elif model_name == "B2":
		other_pars = [K]
	elif model_name == "B3":
		other_pars = [C]
	elif model_name == "B4":
		other_pars = [K, C]

	model = Modelp50(np.concatenate((other_pars, t_pars)), model_name)
	model.calculateState(N, I, P)
	model.calculateF()
	return model.f * scaling

def get_product(t_pars, K, C, model_name="B2"):
	# Return product of beta and t
	if model_name == "B1":
		other_pars = []
	elif model_name == "B2":
		other_pars = [K]
	elif model_name == "B3":
		other_pars = [C]
	elif model_name == "B4":
		other_pars = [K, C]

	model = Modelp50(np.concatenate((other_pars, t_pars)), model_name)
	state_names = ["none", "IRF", r"$IRF_G$", r"IRF + p50", r"NF$\kappa$B", r"NF$\kappa$B + p50", "p50", r"IRF + $IRF_G$",
		r"IRF + NF$\kappa$B", r"$IRF_G$ + NF$\kappa$B", r"IRF + NF$\kappa$B + p50", r"IRF + $IRF_G$ + NF$\kappa$B"]
	return model.beta * model.t, state_names

def get_state_prob(t_pars, K, C, N, I, P, model_name="B2"):
	# Return probability of each state
	if model_name == "B1":
		other_pars = []
	elif model_name == "B2":
		other_pars = [K]
	elif model_name == "B3":
		other_pars = [C]
	elif model_name == "B4":
		other_pars = [K, C]

	model = Modelp50(np.concatenate((other_pars, t_pars)), model_name)
	model.calculateState(N, I, P)
	probabilities = model.state  / np.sum(model.state)

	state_names = ["none", "IRF", r"$IRF_G$", r"IRF + p50", r"NF$\kappa$B", r"NF$\kappa$B + p50", "p50", r"IRF + $IRF_G$",
		r"IRF + NF$\kappa$B", r"$IRF_G$ + NF$\kappa$B", r"IRF + NF$\kappa$B + p50", r"IRF + $IRF_G$ + NF$\kappa$B"]

	return probabilities, state_names

def get_f_contribution(t_pars, K, C, N, I, P, model_name="B2"):
	# Return fraction of f contributed by each state
	if model_name == "B1":
		other_pars = []
	elif model_name == "B2":
		other_pars = [K]
	elif model_name == "B3":
		other_pars = [C]
	elif model_name == "B4":
		other_pars = [K, C]

	model = Modelp50(np.concatenate((other_pars, t_pars)), model_name)
	model.calculateState(N, I, P)

	f_contributions = model.beta * model.t * model.state / np.dot(np.transpose(model.state), model.beta)
	state_names = ["none", "IRF", r"$IRF_G$", r"IRF + p50", r"NF$\kappa$B", r"NF$\kappa$B + p50", "p50", r"IRF + $IRF_G$",
		r"IRF + NF$\kappa$B", r"$IRF_G$ + NF$\kappa$B", r"IRF + NF$\kappa$B + p50", r"IRF + $IRF_G$ + NF$\kappa$B"]
	
	return f_contributions, state_names