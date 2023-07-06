# Optimize 3-site model by randomly selecting initial parameters

from three_site_model import *
import matplotlib.pyplot as plt
import pandas as pd
import os
import scipy.optimize as opt
import time
from multiprocessing import Pool
plt.style.use("~/IFN_paper/src/theme_bw.mplstyle")

result_dir = "./figures/grid_search_no_CpG/"
os.makedirs(result_dir, exist_ok=True)

print("Loading data")
training_data = pd.read_csv("../data/training_data_noCpG.csv")
num_pts = training_data.shape[0]
print(training_data)

def three_site_objective(pars, *args):
    N, I, beta, model_name = args
    if model_name == "B2":
        pars = np.hstack([pars[6], pars[0:6]])
    elif model_name == "B3":
        pars = np.hstack([pars[6], pars[0:6]])
    elif model_name == "B4":
        pars = np.hstack([pars[6:8], pars[0:6]])
    f_list = [explore_model_three_site(pars, N[i], I[i], model_name) for i in range(num_pts)]
    residuals = np.array(f_list) - beta
    rmsd = np.sqrt(np.mean(residuals**2))
    return rmsd

def select_params(num_params, par_length, seed=5):
    params = np.zeros(num_params, par_length)
    np.random.seed(seed)
    for i in range(num_params):
        params[i] = np.random.uniform(0, 1, par_length)
    return params

def optimize_model(N, I, beta, model_name, params, num_threads=40):
    print("Optimizing model ", model_name)
    start = time.time()
    print("Starting random optimization at %s" % time.ctime())
    par_length = params.shape[1]
    rmsd = np.zeros(par_length)
    for i in range(par_length):
        with Pool(num_threads) as p:
            rmsd[i] = p.map(three_site_objective, params[:, i], args=(N, I, beta, model_name))
    end = time.time()
    print("Finished optimization at %s" % time.ctime())
    t = end - start
    if t < 60:
        print("Time elapsed: %.2f seconds" % t)
    elif t < 3600:
        print("Time elapsed: %.2f minutes" % (t/60))
    else:
        print("Time elapsed: %.2f hours" % (t/3600))