import numpy as np

class Modelp50:
	def __init__(self, pars, model= "B2"):
		i=0
		if model == "B1":
			self.K = 1
		elif model == "B2":
			self.K = pars[0]
			i+=1
		else:
			raise ValueError("Incorrect model type")

		if len(pars[i:len(pars)]) != 6:
			print("i= " + str(i))
			print("pars = " + str(pars), "lengt= " + str(len(pars)))
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
		# Multiply positions containing k_{I2} by self.K
		self.beta[2] = self.K
		self.beta[7] = self.K
		self.beta[9] = self.K
		self.beta[11] = self.K
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

def explore_modelp50(parsT, N, I, P, model_name= "B2"):
	model = Modelp50(parsT, model_name)
	# print(model.parsT)
	# print(model.t)
	# print("beta = ", model.beta)
	model.calculateState(N, I, P)
	# print(model.state)
	model.calculateF()
	# print(model.f)
	return model.f

# arr1 = [1,2,3,4]
# arr2 = np.array([5,6,7,8])
# arr1[0] = arr2[0]
# print(arr1)

# class test():
# 	def __init__(self,pars):
# 		self.parsT = pars[1:len(pars)]
# 		print(self.parsT)
# 		self.t = np.array([0 for i in range(12)]).astype(float)
# 		self.t[1] = 200
# 		self.t[2] = self.parsT[1]
# 		self.t[3] = pars[3]
# 		print(self.parsT[1])
# 		print(self.t[2])


# # set arr1 = [2.00000000e+00 2.22044838e-14 2.22044838e-14 2.22044748e-14
# #  1.00000000e+00 3.43062417e-01 5.17380217e-01]
# arr1 = [2.0, 1.5,2.5,3.5]

# x=test(arr1)
# print(x.t)