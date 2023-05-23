# Using previous data and new p50 data, optimize the model with scipy.optimize,least_squares
from p50_model import *
import matplotlib.pyplot as plt
import pandas as pd
import os
import scipy.optimize as opt
import time
plt.style.use("~/IFN_paper/src/theme_bw.mplstyle")

result_dir = "./figures/lsqnonlin/"
os.makedirs(result_dir, exist_ok=True)

print("Loading data")
training_data = pd.read_csv("../data/p50_training_data.csv")
num_pts = training_data.shape[0]

def p50_objective(pars, *args):
    N, I, P, beta, model_name = args
    f_list = [explore_modelp50(pars, N[i], I[i], P[i], model_name) for i in range(num_pts)]
    residuals = np.array(f_list) - beta
    rmsd = np.sqrt(np.mean(residuals**2))
    return rmsd

def optimize_model(N, I, P, beta, model_name):
    print("Optimizing model ", model_name)
    start = time.time()
    trgs = [slice(0, 1, 0.1) for i in range(6)]
    par_rgs = []
    if model_name == "B2":
        par_rgs.append(slice(0, 2, 0.1))
    elif model_name == "B3":
        par_rgs.append((10**-2, 10**2))
    elif model_name == "B4":
        par_rgs.append(slice(0, 2, 0.1))
        par_rgs.append((10**-2, 10**2))
    rgs = tuple(trgs + par_rgs)
    # print(rgs)
    res = opt.brute(p50_objective, rgs, args=(N, I, P, beta, model_name), Ns=10, full_output=True, finish=opt.fmin,
                    workers=20)
    end = time.time()
    t = end - start
    print("Time elapsed: %.2f seconds" % t)
    return res[0], res[1]

res_title = ["t1", "t2", "t3", "t4", "t5", "t6","K_i2", "C","rmsd"]
results = pd.DataFrame(columns=res_title)
pars, rmsd = optimize_model(training_data["NFkB"], training_data["IRF"], training_data["p50"], training_data["IFNb"], "B1")
# pars, rmsd = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1], 0.11
results.loc["B1"] = np.hstack([pars, np.nan,np.nan,rmsd])

pars, rmsd = optimize_model(training_data["NFkB"], training_data["IRF"], training_data["p50"], training_data["IFNb"], "B2")
# pars, rmsd = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2], 0.22
results.loc["B2"] = np.hstack([pars, np.nan,rmsd])

pars, rmsd = optimize_model(training_data["NFkB"], training_data["IRF"], training_data["p50"], training_data["IFNb"], "B3")
# pars, rmsd = [0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.3], 0.33
results.loc["B3"] = np.hstack([pars[0:6], np.nan, pars[6], rmsd])

pars, rmsd = optimize_model(training_data["NFkB"], training_data["IRF"], training_data["p50"], training_data["IFNb"], "B4")
# pars, rmsd = [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4], 0.44
results.loc["B4"] = np.hstack([pars, rmsd])

# Save results
print("Saving results to ../data/p50_grid_search_minimum_results.csv")
results.to_csv("../data/p50_grid_search_minimum_results.csv")