# Using previous data and new p50 data, grid search for global minimum RMSD and locally minimize using least squares
from p50_model import *
import matplotlib.pyplot as plt
import pandas as pd
import os
import scipy.optimize as opt
import time
plt.style.use("~/IFN_paper/src/theme_bw.mplstyle")

result_dir = "./figures/grid_search/"
os.makedirs(result_dir, exist_ok=True)

print("Loading data")
training_data = pd.read_csv("../data/p50_training_data.csv")
num_pts = training_data.shape[0]
print(training_data)

def p50_objective(pars, *args):
    N, I, P, beta, model_name = args
    if model_name == "B2":
        pars = np.hstack([pars[6], pars[0:6]])
    elif model_name == "B3":
        pars = np.hstack([pars[6], pars[0:6]])
    elif model_name == "B4":
        pars = np.hstack([pars[6:8], pars[0:6]])
    f_list = [explore_modelp50(pars, N[i], I[i], P[i], model_name) for i in range(num_pts)]
    residuals = np.array(f_list) - beta
    rmsd = np.sqrt(np.mean(residuals**2))
    return rmsd

def optimize_model(N, I, P, beta, model_name):
    print("Optimizing model ", model_name)
    start = time.time()
    print("Starting brute force optimization at ", time.ctime())
    trgs = [slice(0, 1, 0.1) for i in range(6)]
    par_rgs = []
    if model_name == "B2":
        par_rgs.append(slice(0, 2, 0.1))
    elif model_name == "B3":
        par_rgs.append((10**-2, 10**2))
    elif model_name == "B4":
        par_rgs.append(slice(0, 2, 0.1))
        par_rgs.append((10**-2, 10**2))
    rgs = tuple(trgs+par_rgs)
    # print(rgs)
    res = opt.brute(p50_objective, rgs, args=(N, I, P, beta, model_name), Ns=10, full_output=True, finish=None,
                    workers=20)
    end = time.time()
    print("Optimized parameters:\n", res[0])
    t = end - start
    if t < 60:
        print("Time elapsed: %.2f seconds" % t)
    elif t < 3600:
        print("Time elapsed: %.2f minutes" % (t/60))
    else:
        print("Time elapsed: %.2f hours" % (t/3600))
    return res[0], res[1], res[3]

def p50_objective_local(pars, *args):
    N, I, P, beta, model_name = args
    if model_name == "B2":
        pars = np.hstack([pars[6], pars[0:6]])
    elif model_name == "B3":
        pars = np.hstack([pars[6], pars[0:6]])
    elif model_name == "B4":
        pars = np.hstack([pars[6:8], pars[0:6]])
    f_list = [explore_modelp50(pars, N[i], I[i], P[i], model_name) for i in range(num_pts)]
    residuals = np.array(f_list) - beta
    return residuals

def optimize_model_local(N, I, P, beta, model_name, pars):
    start = time.time()
    if True:
        method = "trf"
    else:
        method = "lm"
    m_name = {"lm": "Levenberg-Marquardt", "trf": "Trust Region Reflective"}
    print("Using %s method to locally optimize model %s" % (m_name[method], model_name))
    print("Starting local optimization at ", time.ctime())
    upper = np.array([1 for i in range(6)])
    lower = np.array([0 for i in range(6)])
    if model_name == "B2":
        upper = np.append(upper, 2)
        lower = np.append(lower, 0)
    elif model_name == "B3":
        upper = np.append(upper, 10**2)
        lower = np.append(lower, 10**-2)
    elif model_name == "B4":
        upper = np.append(upper, [2, 10**2])
        lower = np.append(lower, [0, 10**-2])

    res = opt.least_squares(p50_objective_local, pars, args=(N, I, P, beta, model_name),
                             method=method, bounds=(lower, upper), loss = "linear")
    end = time.time()
    print("Optimized parameters:\n", res.x)
    t = end - start
    if t < 60:
        print("Time elapsed: %.2f seconds" % t)
    elif t < 3600:
        print("Time elapsed: %.2f minutes" % (t/60))
    else:
        print("Time elapsed: %.2f hours" % (t/3600))
    return res.x, res.cost, res.fun

res_title = ["t1", "t2", "t3", "t4", "t5", "t6","K_i2", "C","rmsd"]
results = pd.DataFrame(columns=res_title)
pars, rho, jout = optimize_model(training_data["NFkB"], training_data["IRF"], training_data["p50"], training_data["IFNb"], "B1")
# pars, rmsd = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1], 0.11
results.loc["B1"] = np.hstack([pars, np.nan,np.nan,rho])
np.save("../data/p50_grid_search_rho_values_B1.npy", jout)

pars, rho , jout = optimize_model(training_data["NFkB"], training_data["IRF"], training_data["p50"], training_data["IFNb"], "B2")
# pars, rmsd = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2], 0.22
results.loc["B2"] = np.hstack([pars, np.nan,rho])
np.save("../data/p50_grid_search_rho_values_B2.npy", jout)

pars, rho, jout = optimize_model(training_data["NFkB"], training_data["IRF"], training_data["p50"], training_data["IFNb"], "B3")
# pars, rmsd = [0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.3], 0.33
results.loc["B3"] = np.hstack([pars[0:6], np.nan, pars[6], rho])
np.save("../data/p50_grid_search_rho_values_B3.npy", jout)

pars, rho, jout = optimize_model(training_data["NFkB"], training_data["IRF"], training_data["p50"], training_data["IFNb"], "B4")
# pars, rmsd = [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4], 0.44
results.loc["B4"] = np.hstack([pars, rho])
np.save("../data/p50_grid_search_rho_values_B4.npy", jout)

# Save results
print("Saving results to ../data/p50_grid_search_minimum_results.csv")
results.to_csv("../data/p50_grid_search_minimum_results.csv")

# Local optimization
grid_search_results = pd.read_csv("../data/p50_grid_search_minimum_results.csv", index_col=0)
print("Grid search results:\n", grid_search_results)
b1_starting_pars = grid_search_results.loc["B1"].values[0:6]
b2_starting_pars = grid_search_results.loc["B2"].values[0:7]
b3_starting_pars = grid_search_results.loc["B3"].values[[0,1,2,3,4,5,7]]
b4_starting_pars = grid_search_results.loc["B4"].values[0:8]
print("Starting parameters for local optimization:\n B1: %s\n B2: %s\n B3: %s\n B4: %s" % (b1_starting_pars, b2_starting_pars, b3_starting_pars, b4_starting_pars))

res_title = ["t1", "t2", "t3", "t4", "t5", "t6","K_i2", "C","rho"] + ["res_%d" % i for i in range(num_pts)]

results = pd.DataFrame(columns=res_title)
pars, rho, residuals = optimize_model_local(training_data["NFkB"], training_data["IRF"], training_data["p50"],
                                            training_data["IFNb"], "B1", b1_starting_pars)
results.loc["B1"] = np.hstack([pars, np.nan, np.nan, rho, residuals])

pars, rho, residuals = optimize_model_local(training_data["NFkB"], training_data["IRF"], training_data["p50"],
                                            training_data["IFNb"], "B2", b2_starting_pars)
results.loc["B2"] = np.hstack([pars, np.nan, rho, residuals])

pars, rho, residuals = optimize_model_local(training_data["NFkB"], training_data["IRF"], training_data["p50"],
                                            training_data["IFNb"], "B3", b3_starting_pars)
results.loc["B3"] = np.hstack([pars[0:6], np.nan, pars[6], rho, residuals])

pars, rho, residuals = optimize_model_local(training_data["NFkB"], training_data["IRF"], training_data["p50"],
                                            training_data["IFNb"], "B4", b4_starting_pars)
results.loc["B4"] = np.hstack([pars, rho, residuals])

# Save results
print("Saving results to ../data/p50_local_optimization_results.csv")
results.to_csv("../data/p50_local_optimization_results.csv")

# Make contour plots from best fit parameters
results = pd.read_csv("../data/p50_local_optimization_results.csv", index_col=0)
pars = {"B1": results.loc["B1"].values[0:6],
        "B2": np.hstack([results.loc["B2"].values[6], results.loc["B2"].values[0:6]]),
        "B3": np.hstack([results.loc["B3"].values[7], results.loc["B3"].values[0:6]]),
        "B4": np.hstack([results.loc["B4"].values[6:8], results.loc["B4"].values[0:6]])}
I = np.linspace(0, 1, 100)
N= I.copy()
P = np.array([1 for i in range(100)])

for model in ["B1", "B2", "B3", "B4"]:
    f = calculateFvalues(model, pars[model], I, N)
    title = "grid_opt_best_fit_%s" % model
    plot_contour(f, model, I, N, result_dir, title)

#  Plot best fit parameters
t_pars = {"B1": results.loc["B1"].values[0:6],
          "B2": results.loc["B2"].values[0:6],
          "B3": results.loc["B3"].values[0:6],
          "B4": results.loc["B4"].values[0:6]}
fig = plt.figure()
i=0
for model in ["B1", "B2", "B3", "B4"]:
    x= np.arange(i, i+6)
    plt.plot(x, t_pars[model], label=model, marker="o", linestyle="none")
    i+=0.1
plt.legend(bbox_to_anchor=(1.2,0.5))
plt.ylabel("Transcription capability (t)")
plt.xlabel("Parameter")
plt.grid(False)
# plt.xticks(range(6), list(results)[0:6])
plt.xticks(np.arange(0.2,6.2), [r"IRF_1", r"IRF_2", r"NF$\kappa$B", r"IRF_1 IRF_2", r"IRF_1 NF$\kappa$B", r"IRF_2 NF$\kappa$B"], rotation=45)
plt.title("Best fit parameters with grid search for 3-site model")
plt.savefig(os.path.join(result_dir, "grid_opt_best_fit_parameters.png"), bbox_inches="tight")

# Compare predicted IFNb to testing data
test_data = pd.read_csv("../data/p50_testing_data.csv")
testing_rmsd = {}
f_vals = test_data.copy()
for model in ["B1", "B2", "B3", "B4"]:
    f = [explore_modelp50(pars[model], test_data["NFkB"][i], test_data["IRF"][i], test_data["p50"][i], model) for i in range(test_data.shape[0])]
    # f = f / f[1] * f_vals["IFNb"][1]
    f_vals["IFNb_%s" % model] = f
    rmsd = np.sqrt(np.mean((f / f[1] * f_vals["IFNb"][1] - test_data["IFNb"])**2))
    testing_rmsd[model] = rmsd
print("Testing RMSD:\n", testing_rmsd)


p50_vals = np.linspace(0, 2, 100)
fig = plt.figure()
for model_name in ["B1", "B2", "B3", "B4"]:
    ifnb = [explore_modelp50(pars[model_name], test_data["NFkB"][0], test_data["IRF"][0], p50, model_name) for p50 in p50_vals]
    # print("IFNb at p50=0: %.2f, at p50=1: %.2f, at p50=2: %.2f" % (ifnb[0], ifnb[50], ifnb[99]))
    #  Normalize to p50 = 1
    ifnb = ifnb / f_vals["IFNb_%s" % model_name][1] * f_vals["IFNb"][1] 
    print("IFNb at p50=0: %.2f, at p50=1: %.2f, at p50=2: %.2f" % (ifnb[0], ifnb[50], ifnb[99]))
    plt.plot(p50_vals, ifnb, label=model_name)
plt.scatter(test_data["p50"], test_data["IFNb"], s=50, label="Testing Data", zorder =2, color="black")
plt.xlabel("p50")
plt.ylabel(r"IFN$\beta$")
plt.ylim([0,1])
plt.title("Model predictions for all models")
plt.grid(False)
fig.legend(bbox_to_anchor=(1.15,0.5))
plt.savefig("%s/grid_opt_predictions.png" % result_dir, bbox_inches="tight")