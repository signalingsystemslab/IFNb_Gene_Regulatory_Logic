# Author: Allison Schiffman
# https://github.com/aschiffman/


# import numpy
import numpy as np
# start a class for the model called ModelB1.
# Define the class and the init method with the following parameters: parsT is equal to the input params, beta is equal to list of ones length 8, and t is equal to [0 self.parsT 1]
class ModelB1:	
	def __init__(self, parsT):
		self.parsT = parsT
		self.beta = np.array([1 for i in range(8)])
		# make self.t a list with 0, the elements from self.parsT, and 1
		self.t = np.array([0] + self.parsT + [1])
		# print(self.t)

# Define the method called calculateState for the model called modelB1. The method should take the following parameters: N and I. The method should create a class variable called state with a list with 1, I, I, N, I squared, I times N, I times N, and I squared times N.
	def calculateState(self, N, I):
		self.state = np.array([1, I, I, N, I**2, I*N, I*N, I**2*N])

# Define a method called calculateF. The method should take no parameters.
# The method should create a class variable called f which is equal to the transpose of objs.state times the multiple of obj.beta and obj.t, all divided by the the multiple of obj.state transpose and obj.beta.
	def calculateF(self):
		self.f = np.dot(np.transpose(self.state),(self.beta * self.t)) / np.dot(np.transpose(self.state), self.beta)

def explore_modelB1(parsT, N, I):
    model = ModelB1(parsT)
    model.calculateState(N, I)
    model.calculateF()
    return model.f

# def main():
# 	# Create an instance of the class modelB1 called p50 with the following parameters: [ 0.20650806 0.20199513 0.04456881 0.82492396 0.56675847 0.60181564]
# 	p50 = ModelB1([ 0.20650806, 0.20199513, 0.04456881, 0.82492396, 0.56675847, 0.60181564])
# 	# Call the method calculateState for the instance p50 with the following parameters: I=0.25, N=0
# 	p50.calculateState(0, 0.25)
# 	#Calculate f
# 	p50.calculateF()
# 	# Print the class variable f for the instance p50
# 	# Create a dictionary called p50_dict with the following key-value pairs: 'B1': p50.f
# 	p50_dict = {'B1': p50.f}
# 	# Save the dictionary p50_dict to a file called p50_dict.npy
# 	np.save('p50_dict.npy', p50_dict)
# 	print(p50.f)

# if __name__ == '__main__':
# 	main()


	