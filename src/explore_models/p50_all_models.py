# Copy classes ModelB1, ModelB2, ModelB3, and ModelB4 from p50_modelB1.py, p50_modelB2.py, p50_modelB3.py, and p50_modelB4.py, respectively.
#
import numpy as np
import matplotlib.pyplot as plt

from modelB1 import *
from modelB2 import *
from modelB3 import *
from modelB4 import *


# write a function for plotting the results that takes in the numpy array, the title, and the filename
def plot_model_results(f, title, filename):
    plt.figure()
    # Plot the results
    plt.plot(f[:, 0], f[:, 1], color="purple", linewidth=2.5, linestyle="-")
    # plt.legend(loc = 'upper right')
    plt.xlabel('p50')
    plt.ylabel('IFNb mRNA')
    plt.title(title)
    plt.xticks(np.linspace(f[0, 0], f[-1, 0], 10))
    plt.savefig(filename)


# # Test model B2 for relacrel mutant with K_i2 = 1.51644916, t= [0.16063854, 0.12538234, 0.03271724, 0.83962291, 0.46298051, 0.49638786], parsT = k_12, t
# # # N = 0, I = 0.25
k_i2 = 1.51644916
t = [0.16063854, 0.12538234, 0.03271724, 0.83962291, 0.46298051, 0.49638786]
C=2
# print("Test model B2 for relacrel mutant with parsT = k_i2, t")
# relacrel = explore_modelB2([k_i2] + t, 0, 0.25)
# print(relacrel)

# # Test model B2 for relarelbcrel mutant with parsT = k_i2*2  0.16063854   0.12538234   0.03271724   0.83962291   0.46298051   0.49638786 ,
# # N = 0, I = 0.25
# print("Test model B2 for relarelbcrel mutant with parsT = k_i2/2")
# relarelbcrel = explore_modelB2([k_i2/2] + t, 0, 0.25)
# print(relarelbcrel)

# # Test model B2 for relacrelnfkb mutant with parsT = k_i2/2  0.16063854   0.12538234   0.03271724   0.83962291   0.46298051   0.49638786 ,
# # N = 0, I = 0.25
# print("Test model B2 for relacrelnfkb mutant with parsT = k_i2*2")
# relacrelnfkb = explore_modelB2([k_i2*2] + t, 0, 0.25)
# print(relacrelnfkb)

# # Test model B1 for relacrel mutant with parsT = t
# print("Test model B1 for relacrel mutant with parsT = t")
# relacrel = explore_modelB1(t, 0, 0.25)
# print(relacrel)

# # Test model B3 for relacrel mutant with C=2, parsT = t
# print("Test model B3 for relacrel mutant with C=2, parsT = t")
# C=2
# relacrel = explore_modelB3([C] + t, 0, 0.25)
# print(relacrel)

# # Test model B4 for relacrel mutant with C=2, K_i2 = 1.51644916, parsT = t
# print("Test model B4 for relacrel mutant with C=2, K_i2 = 1.51644916, parsT = t")
# relacrel = explore_modelB4([k_i2,C] + t, 0, 0.25)
# print(relacrel)


# Test ModelB2 with k_i2 scaled by all values in the range 0.5 to 2.0
k_scale = 1/np.linspace(0.1, 3.0, 5000)
# Create a matrix where inverse of k_scale is the first column and the second column is the corresponding f value
f = np.zeros((len(k_scale), 2))
for i in range(len(k_scale)):
    f[i, 0] = 1/k_scale[i]
    f[i, 1] = explore_modelB2([k_scale[i]*k_i2] +t, 0, 0.25)

# Plot the results using the function plot_model_results
plot_model_results(f, 'Model B2 +p50 scaling', './figures/modelB2_test_p50_scaling.png')

# Test model B1 with k_i2 scaled by all values in the range 0.5 to 2.0
for i in range(len(k_scale)):
    f[i, 0] = 1/k_scale[i]
    f[i, 1] = explore_modelB1(t, 0, 0.25)

# Plot the results using the function plot_model_results
plot_model_results(f, 'Model B1 +p50 scaling', './figures/modelB1_test_p50_scaling.png')


# Test model B3 with k_i2 scaled by all values in the range 0.5 to 2.0
for i in range(len(k_scale)):
    f[i, 0] = 1/k_scale[i]
    f[i, 1] = explore_modelB3([C] + t, 0, 0.25)

# Plot the results using the function plot_model_results
plot_model_results(f, 'Model B3 +p50 scaling', './figures/modelB3_test_p50_scaling.png')

# Test model B4 with k_i2 scaled by all values in the range 0.5 to 2.0
for i in range(len(k_scale)):
    f[i, 0] = 1/k_scale[i]
    f[i, 1] = explore_modelB4([k_scale[i]*k_i2, C] + t , 0, 0.25)

# Plot the results using the function plot_model_results
plot_model_results(f, 'Model B4 +p50 scaling', './figures/modelB4_test_p50_scaling.png')

# write a function called comp that takes in the input p50 which is a float and calculates a value k with a competitive inhibition term


p50 = np.linspace(0.01, 1, 1000)
k = comp(p50, 1, 0.25)

f = np.zeros((len(k), 2))
for i in range(len(k)):
    f[i, 0] = k[i]
    f[i, 1] = explore_modelB2([k[i]] + t, 0, 0.25)

# Plot the results using the function plot_model_results
plot_model_results(f, "Model B2 +p50 competition", "./figures/modelB2_test_p50_competition.png")
