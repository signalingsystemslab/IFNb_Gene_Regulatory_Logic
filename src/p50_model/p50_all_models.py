# Copy classes ModelB1, ModelB2, ModelB3, and ModelB4 from p50_modelB1.py, p50_modelB2.py, p50_modelB3.py, and p50_modelB4.py, respectively.
#
import numpy as np
import matplotlib.pyplot as plt

class ModelB1:	
	def __init__(self, parsT):
		self.parsT = parsT
		self.beta = [1 for i in range(8)]
		# make self.t a list with 0, the elements from self.parsT, and 1
		self.t = [0] + self.parsT + [1]

	def calculateState(self, N, I):
		self.state = [1, I, I, N, I**2, I*N, I*N, I**2*N]
                
	def calculateF(self):
		self.f = np.transpose(self.state) * np.dot(self.beta, self.t) / np.dot(np.transpose(self.state), self.beta)

class ModelB2:
    def __init__(self, parsT):
        self.K = parsT[0]
        self.parsT = parsT[1:len(parsT)]
        self.beta = [1 for i in range(8)]
        # beta position 0 refers to none, beta position 1 refers to Ki1, beta position 2 refers to Ki2, beta position 3 refers to kN,
        # beta position 4 refers to Ki1Ki2, beta position 5 refers to Ki1kN, beta position 6 refers to Ki2kN, beta position 7 refers to Ki1Ki2kN
        # 
        # multiply beta by K at positions referring to Ki2, Ki1Ki2, Ki2kN, and Ki1Ki2kN
        # multiply beta by C at positions referring to Ki1Ki2, Ki1Ki2kN
        self.beta[2] = self.K
        self.beta[4] = self.K
        self.beta[6] = self.K
        self.beta[7] = self.K

        self.t = [0] + self.parsT + [1]

    def calculateState(self, N, I):
        self.state = [1, I, I, N, I**2, I*N, I*N, I**2*N]

    def calculateF(self):
        self.f = np.transpose(self.state) * np.dot(self.beta, self.t) / np.dot(np.transpose(self.state), self.beta)

class ModelB3:
    def __init__(self, parsT):
        self.C = parsT[0]
        self.parsT = parsT[1:len(parsT)]
        self.beta = [1 for i in range(8)]
        # beta position 0 refers to none, beta position 1 refers to Ki1, beta position 2 refers to Ki2, beta position 3 refers to kN,
        # beta position 4 refers to Ki1Ki2, beta position 5 refers to Ki1kN, beta position 6 refers to Ki2kN, beta position 7 refers to Ki1Ki2kN
        # multiply beta by C at positions referring to Ki1Ki2, Ki1Ki2kN
        self.beta[4] = self.C
        self.beta[7] = self.C
        
        self.t = [0] + self.parsT + [1]

    def calculateState(self, N, I):
        self.state = [1, I, I, N, I**2, I*N, I*N, I**2*N]

    def calculateF(self):
        self.f = np.transpose(self.state) * np.dot(self.beta, self.t) / np.dot(np.transpose(self.state), self.beta)

class ModelB4:
    def __init__(self, parsT):
        self.K = parsT[0]
        self.C = parsT[1]
        self.parsT = parsT[2:len(parsT)]
        self.beta = [1 for i in range(8)]
        # beta position 0 refers to none, beta position 1 refers to Ki1, beta position 2 refers to Ki2, beta position 3 refers to kN,
        # beta position 4 refers to Ki1Ki2, beta position 5 refers to Ki1kN, beta position 6 refers to Ki2kN, beta position 7 refers to Ki1Ki2kN
        # 
        # multiply beta by K at positions referring to Ki2, Ki1Ki2, Ki2kN, and Ki1Ki2kN
        # multiply beta by C at positions referring to Ki1Ki2, Ki1Ki2kN
        self.beta[2] = self.K
        self.beta[4] = self.K * self.C
        self.beta[6] = self.K
        self.beta[7] = self.K * self.C

        self.t = [0] + self.parsT + [1]

    def calculateState(self, N, I):
        self.state = [1, I, I, N, I**2, I*N, I*N, I**2*N]

    def calculateF(self):
        self.f = np.transpose(self.state) * np.dot(self.beta, self.t) / np.dot(np.transpose(self.state), self.beta)

def explore_modelB1(parsT, N, I):
    model = ModelB1(parsT)
    model.calculateState(N, I)
    model.calculateF()
    return model.f

def explore_modelB2(parsT, N, I):
    model = ModelB2(parsT)
    model.calculateState(N, I)
    model.calculateF()
    return model.f

def explore_modelB3(parsT, N, I):
    model = ModelB3(parsT)
    model.calculateState(N, I)
    model.calculateF()
    return model.f

def explore_modelB4(parsT, N, I):
    model = ModelB4(parsT)
    model.calculateState(N, I)
    model.calculateF()
    return model.f

# Test model B2 for relacrel mutant with K_i2 = 1.51644916, parsT = k_12   0.16063854   0.12538234   0.03271724   0.83962291   0.46298051   0.49638786 ,
# N = 0, I = 0.25
k_i2 = 1.51644916
relacrel = explore_modelB2([k_i2, 0.16063854, 0.12538234, 0.03271724, 0.83962291, 0.46298051, 0.49638786], 0, 0.25)

# Test model B2 for relarelbcrel mutant with parsT = k_i2*2  0.16063854   0.12538234   0.03271724   0.83962291   0.46298051   0.49638786 ,
# N = 0, I = 0.25
relarelbcrel = explore_modelB2([k_i2/2, 0.16063854, 0.12538234, 0.03271724, 0.83962291, 0.46298051, 0.49638786], 0, 0.25)

# Test model B2 for relacrelnfkb mutant with parsT = k_i2/2  0.16063854   0.12538234   0.03271724   0.83962291   0.46298051   0.49638786 ,
# N = 0, I = 0.25
relacrelnfkb = explore_modelB2([k_i2*2, 0.16063854, 0.12538234, 0.03271724, 0.83962291, 0.46298051, 0.49638786], 0, 0.25)


# Plot the results 
plt.plot(relacrel, 'g.', label = 'relacrel (control))')
plt.plot(relarelbcrel, 'r.', label = 'relarelbcrel (high p50)')
plt.plot(relacrelnfkb, 'b.', label = 'relacrelnfkb (low p50))')
plt.legend(loc = 'upper right')
plt.xlabel('State')
plt.ylabel('IFNb mRNA')
plt.title('Model B2 example genotypes')
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7], ['None', 'I1', 'I2', 'N', 'I1I2', 'I1N', 'I2N', 'I1I2N'])

# Save the plot to a png file in ./figures/ called "modelB2_test_genotypes_all_states.png"
plt.savefig('./figures/modelB2_test_genotypes_all_states.png')

# Test ModelB2 with k_i2 scaled by all values in the range 0.5 to 2.0
k_scale = 1/np.linspace(0.1, 10.0, 100)
# Create a matrix where k_scale is the first column and the next 8 columns are the corresponding f value
f = np.zeros((len(k_scale), 9))

for i in range(len(k_scale)):
    f[i, 0] = 1/k_scale[i]
    f[i, 1:9] = explore_modelB2([k_scale[i]*k_i2, 0.16063854, 0.12538234, 0.03271724, 0.83962291, 0.46298051, 0.49638786], 0, 0.25)


# Initialize new figure
plt.figure()
# Plot the results
plt.plot(f[:, 0], f[:, 2], 'r', label = 'I1')
plt.plot(f[:, 0], f[:, 3], 'b', label = 'I2')
plt.plot(f[:, 0], f[:, 4], 'y', label = 'N')
plt.plot(f[:, 0], f[:, 5], 'c', label = 'I1I2')
plt.plot(f[:, 0], f[:, 6], 'm', label = 'I1N')
plt.plot(f[:, 0], f[:, 7], 'k', label = 'I2N')
plt.plot(f[:, 0], f[:, 8], 'w', label = 'I1I2N')
plt.legend(loc = 'upper right')
plt.xlabel('p50')
plt.ylabel('IFNb mRNA')
plt.title('Model B2 p50 scaling')

# make x-ticks in a linspace from the first element of the first column of f to the last element of the first column of f
plt.xticks(np.linspace(f[0, 0], f[-1, 0], 10))

# Save the plot to a png file in ./figures/ called "modelB2_test_p50_scaling.png"
plt.savefig('./figures/modelB2_test_p50_scaling.png')

