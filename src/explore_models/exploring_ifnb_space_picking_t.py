from modelB2 import *
from modelB1 import *
from modelB3 import *
from modelB4 import *
import matplotlib.pyplot as plt


## Exploring model B2
# initialize a vector n and a vector i from 0 to 1 with 100 elements
N = np.linspace(0, 1, 100)
I = np.linspace(0, 1, 100)

f_values = np.zeros((len(N), len(I)))
k_i2 = 2

# make vector t = [2.22e-14 2.22e-14 2.22e-14 1.00 0.343 0.517]
t = [2.22e-14, 2.22e-14, 2.22e-14, 1.00, 0.343, 0.517]

colors = {"min":"#042940", "max":"#042940", "between":"#005C53"}

for n in range(len(N)):
    for i in range(len(I)):
        f = explore_modelB2([k_i2] + t, N[n], I[i])
        f_values[n,i] = f


f_values = f_values / np.max(f_values)
f_b2 = f_values.copy()

# Make a contour plot of the with a colormap from blue to red
fig=plt.figure()
plt.contourf(I,N, f_values, 100, cmap="RdYlBu_r")
plt.colorbar(format="%.1f")
plt.title("Model B2 best fit results")
fig.gca().set_ylabel(r"$NF\kappa B$")
fig.gca().set_xlabel(r"$IRF$")
plt.savefig("./figures/picking_t/modelB2_best_fit_results.png")

# for each value of n, calculate the minimum value of f and the maximum value of f and save them in a new matrix
f_plot = np.zeros((len(N),2))
for n in range(len(N)):
    f_plot[n,0] = np.min(f_values[n,:])
    f_plot[n,1] = np.max(f_values[n,:])



mid = int(np.round(len(N)/2))

# plot the results and color between the min and max values
plt.figure()
plt.plot(N, f_plot[:,0], color=colors["min"], label="min")
plt.plot(N, f_plot[:,1], color=colors["max"], label="max")
plt.fill_between(N, f_plot[:,0], f_plot[:,1], color=colors["between"], alpha=0.2)
plt.xlabel("N")
plt.ylabel("IFNb fraction of max")
plt.text(0.5, f_plot[mid,0]+0.05*f_plot[mid,1], "min", color=colors["min"])
plt.text(0.5, f_plot[mid,1]*1.05, "max", color=colors["max"])
plt.title("IFNb vs N Model B2")
plt.ylim(0,1)
plt.savefig("./figures/picking_t/ifnb_vs_N_modelB2.png")

# for each value of i, calculate the minimum value of f and the maximum value of f and save them in a new matrix
f_plot = np.zeros((len(I),2))
for i in range(len(I)):
    f_plot[i,0] = np.min(f_values[:,i])
    f_plot[i,1] = np.max(f_values[:,i])

mid = int(np.round(len(I)/2))

# plot the results and color between the min and max values
plt.figure()
plt.plot(I, f_plot[:,0], color=colors["min"], label="min")
plt.plot(I, f_plot[:,1], color=colors["max"], label="max")
plt.fill_between(I, f_plot[:,0], f_plot[:,1], color=colors["between"], alpha=0.2)
plt.xlabel("I")
plt.ylabel("IFNb fraction of max")
plt.text(0.5, f_plot[mid,0]+0.1*f_plot[mid,1], "min", color=colors["min"])
plt.text(0.5, f_plot[mid,1]*1.1, "max", color=colors["max"])
plt.title("IFNb vs I Model B2")
plt.ylim(0,1)
plt.savefig("./figures/picking_t/ifnb_vs_I_modelB2.png")

## Exploring model B1  
# make vector t = [0.225 0.241 1.43e- 3 4.77e- 1 0.904 0.404]
t = [0.225, 0.241, 1.43e-3, 4.77e-1, 0.904, 0.404]

f_values = np.zeros((len(N), len(I)))
for n in range(len(N)):
    for i in range(len(I)):
        f = explore_modelB1(t, N[n], I[i])
        f_values[n,i] = f


f_values = f_values / np.max(f_values)
f_b1 = f_values.copy()

# Make a contour plot
fig=plt.figure()
plt.contourf(I,N, f_values, 100, cmap="RdYlBu_r")
plt.colorbar(format="%.1f")
plt.title("Model B1 best fit results")
fig.gca().set_ylabel(r"$NF\kappa B$")
fig.gca().set_xlabel(r"$IRF$")
plt.savefig("./figures/picking_t/modelB1_best_fit_results.png")


# for each value of n, calculate the minimum value of f and the maximum value of f and save them in a new matrix
f_plot = np.zeros((len(N),2))
for n in range(len(N)):
    f_plot[n,0] = np.min(f_values[n,:])
    f_plot[n,1] = np.max(f_values[n,:])

mid = int(np.round(len(N)/2))

# plot the results and color between the min and max values
plt.figure()
plt.plot(N, f_plot[:,0], color=colors["min"], label="min")
plt.plot(N, f_plot[:,1], color=colors["max"], label="max")
plt.fill_between(N, f_plot[:,0], f_plot[:,1], color=colors["between"], alpha=0.2)
plt.xlabel("N")
plt.ylabel("IFNb fraction of max")
plt.text(0.5, f_plot[mid,0]+0.05*f_plot[mid,1], "min", color=colors["min"])
plt.text(0.5, f_plot[mid,1]*1.05, "max", color=colors["max"])
plt.title("IFNb vs N Model B1")
plt.ylim(0,1)
plt.savefig("./figures/picking_t/ifnb_vs_N_modelB1.png")

# for each value of i, calculate the minimum value of f and the maximum value of f and save them in a new matrix
f_plot = np.zeros((len(I),2))
for i in range(len(I)):
    f_plot[i,0] = np.min(f_values[:,i])
    f_plot[i,1] = np.max(f_values[:,i])

mid = int(np.round(len(I)/2))

# plot the results and color between the min and max values
plt.figure()
plt.plot(I, f_plot[:,0], color=colors["min"], label="min")
plt.plot(I, f_plot[:,1], color=colors["max"], label="max")
plt.fill_between(I, f_plot[:,0], f_plot[:,1], color=colors["between"], alpha=0.2)
plt.xlabel("I")
plt.ylabel("IFNb fraction of max")
plt.text(0.5, f_plot[mid,0]+0.1*f_plot[mid,1], "min", color=colors["min"])
plt.text(0.5, f_plot[mid,1]*1.1, "max", color=colors["max"])
plt.title("IFNb vs I Model B1")
plt.ylim(0,1)
plt.savefig("./figures/picking_t/ifnb_vs_I_modelB1.png")

# [plt.close(f) for f in plt.get_fignums()]
# compare f_b2 and f_b1
plt.figure()
plt.plot(N, f_b2[:,0], label="Model B2 0")
plt.plot(N, f_b2[:,mid], label="Model B2 mid")
plt.plot(N, f_b2[:,len(N)-1], label="Model B2 1")
plt.plot(N, f_b1[:,0], label="Model B1 0")
plt.plot(N, f_b1[:,mid], label="Model B1")
plt.plot(N, f_b1[:,len(N)-1], label="Model B1 1")
plt.xlabel("N")
plt.ylabel("IFNb fraction of max")
plt.title("IFNb vs N Model B1 vs B2")
plt.legend()
plt.savefig("./figures/picking_t/ifnb_vs_N_modelB1_vs_B2.png")

## Exploring model B3
# make a vector t with the following values: 3.98e- 2 0.00175 2.23e-14 2.22e-14 7.09e- 2 7.09e- 2
t = [3.98e-2, 0.00175, 2.23e-14, 2.22e-14, 7.09e-2, 7.09e-2]
C= 0.124

f_values = np.zeros((len(N), len(I)))
for n in range(len(N)):
    for i in range(len(I)):
        f = explore_modelB3([C]+t, N[n], I[i])
        f_values[n,i] = f


f_values = f_values / np.max(f_values)
f_b3 = f_values.copy()

# Make a contour plot
fig=plt.figure()
plt.contourf(I,N, f_values, 100, cmap="RdYlBu_r")
plt.colorbar(format="%.1f")
plt.title("Model B3 best fit results")
fig.gca().set_ylabel(r"$NF\kappa B$")
fig.gca().set_xlabel(r"$IRF$")
plt.savefig("./figures/picking_t/modelB3_best_fit_results.png")


# for each value of n, calculate the minimum value of f and the maximum value of f and save them in a new matrix
f_plot = np.zeros((len(N),2))
for n in range(len(N)):
    f_plot[n,0] = np.min(f_values[n,:])
    f_plot[n,1] = np.max(f_values[n,:])

mid = int(np.round(len(N)/2))

# plot the results and color between the min and max values
plt.figure()
plt.plot(N, f_plot[:,0], color=colors["min"], label="min")
plt.plot(N, f_plot[:,1], color=colors["max"], label="max")
plt.fill_between(N, f_plot[:,0], f_plot[:,1], color=colors["between"], alpha=0.2)
plt.xlabel("N")
plt.ylabel("IFNb fraction of max")
plt.text(0.5, f_plot[mid,0]+0.05*f_plot[mid,1], "min", color=colors["min"])
plt.text(0.5, f_plot[mid,1]*1.05, "max", color=colors["max"])
plt.title("IFNb vs N Model B3")
plt.ylim(0,1)
plt.savefig("./figures/picking_t/ifnb_vs_N_modelB3.png")

# for each value of i, calculate the minimum value of f and the maximum value of f and save them in a new matrix
f_plot = np.zeros((len(I),2))
for i in range(len(I)):
    f_plot[i,0] = np.min(f_values[:,i])
    f_plot[i,1] = np.max(f_values[:,i])

mid = int(np.round(len(I)/2))

# plot the results and color between the min and max values
plt.figure()
plt.plot(I, f_plot[:,0], color=colors["min"], label="min")
plt.plot(I, f_plot[:,1], color=colors["max"], label="max")
plt.fill_between(I, f_plot[:,0], f_plot[:,1], color=colors["between"], alpha=0.2)
plt.xlabel("I")
plt.ylabel("IFNb fraction of max")
plt.text(0.5, f_plot[mid,0]+0.1*f_plot[mid,1], "min", color=colors["min"])
plt.text(0.5, f_plot[mid,1]*1.1, "max", color=colors["max"])
plt.title("IFNb vs I Model B3")
plt.ylim(0,1)
plt.savefig("./figures/picking_t/ifnb_vs_I_modelB3.png")


## Exploring model B4
# make a list t with the following values:  0.114 0.00262 0 1 1 1
t = [0.114, 0.00262, 0, 1, 1, 1]
k_i2 = 2
C= 1.81

f_values = np.zeros((len(N), len(I)))
for n in range(len(N)):
    for i in range(len(I)):
        f = explore_modelB4([k_i2, C]+t, N[n], I[i])
        f_values[n,i] = f


f_values = f_values / np.max(f_values)
f_b4 = f_values.copy()

# Make a contour plot
fig=plt.figure()
plt.contourf(I,N, f_values, 100, cmap="RdYlBu_r")
plt.colorbar(format="%.1f")
plt.title("Model B4 best fit results")
fig.gca().set_ylabel(r"$NF\kappa B$")
fig.gca().set_xlabel(r"$IRF$")
plt.savefig("./figures/picking_t/modelB4_best_fit_results.png")


# for each value of n, calculate the minimum value of f and the maximum value of f and save them in a new matrix
f_plot = np.zeros((len(N),2))
for n in range(len(N)):
    f_plot[n,0] = np.min(f_values[n,:])
    f_plot[n,1] = np.max(f_values[n,:])

mid = int(np.round(len(N)/2))

# plot the results and color between the min and max values
plt.figure()
plt.plot(N, f_plot[:,0], color=colors["min"], label="min")
plt.plot(N, f_plot[:,1], color=colors["max"], label="max")
plt.fill_between(N, f_plot[:,0], f_plot[:,1], color=colors["between"], alpha=0.2)
plt.xlabel("N")
plt.ylabel("IFNb fraction of max")
plt.text(0.5, f_plot[mid,0]+0.05*f_plot[mid,1], "min", color=colors["min"])
plt.text(0.5, f_plot[mid,1]*1.05, "max", color=colors["max"])
plt.title("IFNb vs N Model B4")
plt.ylim(0,1)
plt.savefig("./figures/picking_t/ifnb_vs_N_modelB4.png")

# for each value of i, calculate the minimum value of f and the maximum value of f and save them in a new matrix
f_plot = np.zeros((len(I),2))
for i in range(len(I)):
    f_plot[i,0] = np.min(f_values[:,i])
    f_plot[i,1] = np.max(f_values[:,i])

mid = int(np.round(len(I)/2))

# plot the results and color between the min and max values
plt.figure()
plt.plot(I, f_plot[:,0], color=colors["min"], label="min")
plt.plot(I, f_plot[:,1], color=colors["max"], label="max")
plt.fill_between(I, f_plot[:,0], f_plot[:,1], color=colors["between"], alpha=0.2)
plt.xlabel("I")
plt.ylabel("IFNb fraction of max")
plt.text(0.5, f_plot[mid,0]+0.1*f_plot[mid,1], "min", color=colors["min"])
plt.text(0.5, f_plot[mid,1]*1.1, "max", color=colors["max"])
plt.title("IFNb vs I Model B4")
plt.ylim(0,1)
plt.savefig("./figures/picking_t/ifnb_vs_I_modelB4.png")
