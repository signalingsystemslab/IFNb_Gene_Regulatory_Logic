from modelB2 import *
from modelB1 import *
from modelB3 import *
from modelB4 import *
import matplotlib.pyplot as plt
import pandas as pd
# import time
import os

plt.style.use("~/IFN_paper/src/theme_bw.mplstyle")
os.makedirs("./figures/ifnb_space/", exist_ok=True)


# Function for plotting the results of the fit in a contour plot taking in the following arguments: array with F values, name of the model
def plot_contour(f_values, model_name, t, normalize=True):
    if normalize:
        f_values = f_values / np.max(f_values)
    fig=plt.figure()
    plt.contourf(I,N, f_values, 100, cmap="RdYlBu_r")
    plt.grid(False)
    plt.colorbar(format="%.1f")
    plt.title("Model "+model_name+" best fit results"+", dset "+t)
    fig.gca().set_ylabel(r"$NF\kappa B$")
    fig.gca().set_xlabel(r"$IRF$")
    plt.savefig("./figures/ifnb_space/model"+model_name+"_best_fit_results_dset" + t +".png")
    plt.close()

# Function for plotting IFNb vs N or I taking in the following arguments: array with F values, name of the model, name of the variable
def plot_ifnb_vs_either(f_values, model_name, var_name, t, normalize=True):
    if normalize:
        f_values = f_values / np.max(f_values)

    if var_name == "N":
        var = N
        f_plot = np.zeros((len(var),2))
        for n in range(len(var)):
            f_plot[n,0] = np.min(f_values[n,:])
            f_plot[n,1] = np.max(f_values[n,:])
    elif var_name == "I":
        var = I
        f_plot = np.zeros((len(var),2))
        for i in range(len(var)):
            f_plot[i,0] = np.min(f_values[:,i])
            f_plot[i,1] = np.max(f_values[:,i])

    mid = int(np.round(len(var)/2))
    labs = {"N": r"$NF\kappa B$", "I": r"$IRF$"}
    colors = {"min":"#042940", "max":"#042940", "between":"#005C53"}
    # plot the results and color between the min and max values
    plt.figure()
    plt.grid(False)
    plt.plot(var, f_plot[:,0], color=colors["min"], label="min")
    plt.plot(var, f_plot[:,1], color=colors["max"], label="max")
    plt.fill_between(var, f_plot[:,0], f_plot[:,1], color=colors["between"], alpha=0.2)
    plt.xlabel(labs[var_name])
    plt.ylabel(r"$IFN\beta$ fraction of max")
    plt.text(0.5, f_plot[mid,0]+0.1*f_plot[mid,1], "min", color=colors["min"])
    plt.text(0.5, f_plot[mid,1]*1.1, "max", color=colors["max"])
    plt.title("IFNb vs "+var_name+" Model "+model_name+", dset "+t)
    plt.ylim(0,1)
    plt.xlim(0,1)
    plt.savefig("./figures/ifnb_space/ifnb_vs_"+var_name+"_model"+model_name+"dset" + t + ".png")
    plt.close()

def calculateFvalues(model_fun, pars, N, I):
    f_values = np.zeros((len(N), len(I)))
    for n in range(len(N)):
        for i in range(len(I)):
            f_values[n,i] = model_fun(pars,N[n], I[i])
    return f_values


N = np.linspace(0, 1, 100)
I = np.linspace(0, 1, 100)
t_names = ["t1", "t2", "t3", "t4", "t5", "t6"]
parnames = {"B1": t_names, "B2": np.hstack(["K_i2", t_names]), "B3": np.hstack(["C", t_names]), "B4": np.hstack(["K_i2", "C", t_names])}
model_funs = {"B1": explore_modelB1, "B2": explore_modelB2, "B3": explore_modelB3, "B4": explore_modelB4}
for model_name, model_fun in model_funs.items():
    print(model_name)
    pars_df = pd.read_csv("../data/params_fits_" + model_name + "_representative.csv", index_col=0)

    pars_arr = np.array(pars_df.loc[:,parnames[model_name]].astype(float))

    # plot top 5 t values
    
    for row in range(pars_arr.shape[0]):
        d = pars_df.loc[row,"dset"]
        print("param #%d, dataset = %d" % (row, d))
        pars = list(pars_arr[row,:])
        f_values = calculateFvalues(model_fun, pars, N, I)
        plot_contour(f_values, model_name, str(d))
        plot_ifnb_vs_either(f_values, model_name,"N", str(d))
        plot_ifnb_vs_either(f_values, model_name, "I", str(d))

pars = list(np.loadtxt("../data/modelB1_best_fit.csv", delimiter=",")[0,:])
f_values = calculateFvalues(explore_modelB1, pars, N, I)
# plot_contour(f_values, "B1", "test")
plot_ifnb_vs_either(f_values, "B1","N", "test")