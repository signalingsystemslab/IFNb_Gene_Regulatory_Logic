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
			print("pars = " + str(pars))
			raise ValueError("Incorrect number of parameters in input")

		self.parsT = pars[i:len(pars)]
		if len(self.parsT) != 6:
			print("i= " + str(i))
			print("parsT = " + str(self.parsT))
			print("K= " + str(self.K))
			raise ValueError("Incorrect number of parameters for class")

		self.beta = np.array([1.0 for i in range(12)])
		self.beta[2] = self.K
		self.beta[7] = self.K * self.C
		self.beta[9] = self.K
		self.beta[11] = self.K * self.C
		
		self.t = np.array([0.0 for i in range(12)])
		self.t[0] = 0
		self.t[1] = self.parsT[0]
		self.t[2] = self.parsT[1]
		self.t[3] = self.parsT[0]
		self.t[4] = self.parsT[2]
		self.t[5] = self.parsT[2]
		self.t[6] = 0
		self.t[7] = self.parsT[3]
		self.t[8] = self.parsT[4]
		self.t[9] = self.parsT[5]
		self.t[10] = self.parsT[4]
		self.t[11] = 1


	def calculateState(self, N, I, P=0):
		self.state = np.array([1, I, I, I*P, N, N*P, P, I**2, I*N, I*N, I*N*P, I**2*N])

	def calculateF(self):
		# print(self.state)
		self.f = np.dot(np.transpose(self.state),(self.beta * self.t)) / np.dot(np.transpose(self.state), self.beta)

def explore_modelp50(parsT, N, I, P, model_name):
	model = Modelp50(parsT, model_name)
	model.calculateState(N, I, P)
	model.calculateF()
	return model.f

def get_f(t_pars, K, C, N, I, P, model_name="B2", scaling=1):
	if model_name == "B1":
		other_pars = []
	elif model_name == "B2":
		other_pars = [K]
	elif model_name == "B3":
		other_pars = [C]
	elif model_name == "B4":
		other_pars = [K, C]

	model = Modelp50(other_pars + t_pars, model_name)
	model.calculateState(N, I, P)
	model.calculateF()
	return model.f * scaling