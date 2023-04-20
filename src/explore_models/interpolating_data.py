import numpy as np
import scipy
import matplotlib.pyplot as plt
from modelB2 import *
from modelB1 import *
from modelB3 import *
from modelB4 import *

# Load exp_matrix_norm.csv
exp_matrix = np.loadtxt("../data/exp_matrix_norm.csv", delimiter=",")
# First column is irf, second column is nfkb, third column is ifnb
irf = exp_matrix[:,0]
nfkb = exp_matrix[:,1]
ifnb = exp_matrix[:,2]

I = np.linspace(0, 1, 100)
N = np.linspace(0, 1, 100)
I, N = np.meshgrid(I, N)
interp = scipy.interpolate.LinearNDInterpolator(list(zip(irf, nfkb)), ifnb)
B = interp(I, N)

plt.figure()
plt.pcolormesh(I, N, B, cmap="RdYlBu_r")
plt.scatter(irf, nfkb, c=ifnb, cmap="RdYlBu_r", edgecolors="k")
plt.colorbar()
plt.xlabel("IRF")
plt.ylabel(r"NF$\kappa$B")
plt.title(r"Interpolated data: IFN$\beta$ vs IRF and NF$\kappa$B")
plt.savefig("../explore_models/figures/interpolated_data.png")

# # Create synthetic data with interpolated values
# # for each row in exp_matrix, randomly select a value of nfkb and irf from a normal distribution with mean of the value in exp_matrix and standard deviation of 0.05
# npts=100
# sterr = 0.05
# synthetic_data = np.zeros(exp_matrix.shape+(npts,))
# cov = np.array([sterr**2, 0, 0, sterr**2]).reshape(2,2)
# for p in range(0, npts):
#     rng = np.random.default_rng(seed=42)
#     x = [scipy.stats.multivariate_normal(mean=[exp_matrix[j,0], exp_matrix[j,1]], cov=cov).rvs() for j in range(0, exp_matrix.shape[0])]
#     nfkb = [x[j][0] for j in range(0, exp_matrix.shape[0])]
#     irf = [x[j][1] for j in range(0, exp_matrix.shape[0])]
#     ifnb = interp(irf, nfkb)
#     synthetic_data[:,:,p] = np.array([irf, nfkb, ifnb]).T

# print(synthetic_data[:,:,1])
# plt.figure()
# # plt.pcolormesh(I, N, B, cmap="RdYlBu_r")
# plt.scatter(synthetic_data[0,:,:], synthetic_data[1,:,:], c=synthetic_data[2,:,:], cmap="RdYlBu_r")
# plt.colorbar()
# plt.xlabel("IRF")
# plt.ylabel(r"NF$\kappa$B")
# plt.title(r"Synthetic data: IFN$\beta$ vs IRF and NF$\kappa$B")
# plt.savefig("../explore_models/figures/synthetic_data.png")

# Compare model predicted t values to interpolated data
model1_params = np.loadtxt("../data/ModelB1_parameters.csv", delimiter=",")
model2_params = np.loadtxt("../data/ModelB2_parameters.csv", delimiter=",")
model3_params = np.loadtxt("../data/ModelB3_parameters.csv", delimiter=",")
model4_params = np.loadtxt("../data/ModelB4_parameters.csv", delimiter=",")

# Model 1
model1_f = np.zeros((model1_params.shape[0], exp_matrix.shape[0]))

for row in range(0, model1_params.shape[0]):
    for j in range(0, exp_matrix.shape[0]):
        f = explore_modelB2(model1_params[row,:], nfkb[j], irf[j])
        model1_f[row,j] = f