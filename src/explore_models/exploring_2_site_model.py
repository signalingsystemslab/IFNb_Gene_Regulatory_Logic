from model2site import *
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from modelB1 import *
from modelB2 import *
from modelB3 import *
from modelB4 import *

plt.style.use("~/IFN_paper/src/theme_bw.mplstyle")

colors = ["#071540","#B3203B"]
cmap = LinearSegmentedColormap.from_list("cmap", colors)

N = np.linspace(0, 1, 100)
I = np.linspace(0, 1, 100)

def plot_contour(f_values, model_name, C, normalize=True):
    if normalize:
        f_values = f_values / np.max(f_values)
    fig=plt.figure()
    plt.grid(False)
    plt.contourf(I,N, f_values, 100, cmap=cmap)
    plt.colorbar(format="%.1f")
    plt.title("Model "+model_name+" best fit results")
    fig.gca().set_ylabel(r"$NF\kappa B$")
    fig.gca().set_xlabel(r"$IRF$")
    plt.savefig("../2-site-model/figs/contour/model_"+model_name+"_best_fit_results_C%.2f.pdf" % C)

def calculateFvalues2(pars, model, N, I):
    f_values = np.zeros((len(N), len(I)))
    for n in range(len(N)):
        for i in range(len(I)):
            f_values[n,i] = explore_model2site(pars, model, N[n], I[i])
    return f_values

def calculateFvalues(model_fun, pars, N, I):
    f_values = np.zeros((len(N), len(I)))
    for n in range(len(N)):
        for i in range(len(I)):
            f_values[n,i] = model_fun(pars,N[n], I[i])
    return f_values

exp_data = pd.read_csv("../data/exp_data_mins.csv", header=0)
model_names = {3: "AND",
               2: "NFkB",
               1: "IRF",
               4: "OR"}
exp_data["model"] = exp_data["model"].replace(model_names)

exp_matrix = np.loadtxt("../data/exp_matrix_norm.csv", delimiter=",")
irf = exp_matrix[:,0]
nfkb = exp_matrix[:,1]
ifnb = exp_matrix[:,2]
ifnb_predicted = {}

for model in model_names.values():
    C = exp_data.loc[exp_data["model"] == model, "bestC"].values[0]
    print("Model: %s" % model)
    print("C = %.2f" % C)

    f = calculateFvalues2(C, model,N, I)
    plot_contour(f, model, C)
    
    f_values = []
    for i, n in zip(irf, nfkb):
        f = explore_model2site(C, model, n, i)
        f_values.append(f)
    # print(f_values)
    # print("max = %.2f" % np.max(f_values))
    if np.max(f_values) != 0:
        f_values = f_values / np.max(f_values)

    ifnb_predicted[model] = f_values

# for key, value in ifnb_predicted.items():
#     print(key, value)

# 3-site residuals calculation
model_funs = {"B1": explore_modelB1, "B2": explore_modelB2, "B3": explore_modelB3, "B4": explore_modelB4}
for model_name, fun in model_funs.items():
    pars_arr = np.loadtxt("../data/model"+model_name+"_best_fit.csv", delimiter=",")
    pars = list(pars_arr[0,:])
    
    f_values = []
    for i, n in zip(irf, nfkb):
        f = fun(pars, n, i)
        f_values.append(f)

    if np.max(f_values) != 0:
        f_values = f_values / np.max(f_values)

    ifnb_predicted[model_name] = f_values


# make empty dataframe called residuals
residuals = pd.DataFrame(columns=["model", "residuals"])
for model in model_names.values():
    # residuals = residuals.append({"model": model,
    #                                 "residuals": ifnb_predicted[model] - ifnb},
    #                                 ignore_index=True)
    residuals = pd.concat([residuals, pd.DataFrame({"model": model,
                                                            "residuals": [ifnb_predicted[model] - ifnb]})],
                                                            ignore_index=True)

for model in model_funs.keys():
    residuals = pd.concat([residuals, pd.DataFrame({"model": model,
                                                            "residuals": [ifnb_predicted[model] - ifnb]})],
                                                            ignore_index=True)

# print(residuals)

# Add column to data frame with maximum residual
residuals["residuals_squared"] = residuals["residuals"].apply(lambda x: x**2)
residuals["max_residual"] = residuals["residuals_squared"].apply(lambda x: np.max(x))
# Add column to data frame with RMSD
residuals["RMSD"] = residuals["residuals_squared"].apply(lambda x: np.sqrt(np.mean(x)))

print(residuals)
# Plot in a bar chart
fig=plt.figure()
plt.bar(residuals["model"], residuals["max_residual"])
plt.title("Maximum residual for experimental data, 2 and 3 site models")
fig.gca().set_ylabel("Maximum residual squared")
fig.gca().set_xlabel("Model")
plt.savefig("../2-site-model/figs/max_residuals_2_3_site_models.png")
