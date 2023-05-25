import numpy as np
import matplotlib.pyplot as plt

class Modelp50:
	def __init__(self, pars, model= "B2"):
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

def explore_modelp50(parsT, N, I, P, model_name):
	model = Modelp50(parsT, model_name)
	# print(model.parsT)
	# print(model.t)
	# print("beta = ", model.beta)
	model.calculateState(N, I, P)
	# print(model.state)
	model.calculateF()
	# print(model.f)
	return model.f

def plot_contour(f_values, model_name, I, N, dir,name, p50=1, normalize=True):
    if normalize:
        f_values = f_values / np.max(f_values)
    fig=plt.figure()
    plt.contourf(I,N, f_values, 100, cmap="RdYlBu_r")
    plt.grid(False)
    plt.colorbar(format="%.1f")
    plt.title("Model "+model_name+" best fit, p50 = %s" % p50)
    fig.gca().set_ylabel(r"$NF\kappa B$")
    fig.gca().set_xlabel(r"$IRF$")
    plt.savefig("%s/contour_plot_%s.png" % (dir,name))
    plt.close()

def calculateFvalues(model_name, pars, I, N, p50):
    f_values = np.zeros((len(N), len(I)))
    for n in range(len(N)):
        for i in range(len(I)):
            f_values[n,i] = explore_modelp50(pars,N[n], I[i],p50[i], model_name)
    return f_values