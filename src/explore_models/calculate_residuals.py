import numpy as np
import pandas as pd
from modelB1 import *
from modelB2 import *
from modelB3 import *
from modelB4 import *

ifnb_predicted = pd.DataFrame(columns=["model", "pars", "dset","residuals","rmsd"])

exp_matrix = np.loadtxt("../data/exp_matrix_norm.csv", delimiter=",")
irf = exp_matrix[:,0]
nfkb = exp_matrix[:,1]
ifnb = exp_matrix[:,2]
model_funs = {"B1": explore_modelB1, "B2": explore_modelB2, "B3": explore_modelB3, "B4": explore_modelB4}
for model_name, fun in model_funs.items():
    pars_arr = np.loadtxt("../data/Model"+model_name+"_parameters.csv", delimiter=",")
    pars_list = [pars_arr[i,:-1] for i in range(len(pars_arr))]
    dset = pars_arr[:,-1].astype(int)
    rmsd_list = []
    res_list = []

    for pars in pars_list:
        # print(pars)
        f_values = []
        for i, n in zip(irf, nfkb):
            f = fun(pars, n, i)
            f_values.append(f)

        if np.max(f_values) != 0:
            f_values = f_values / np.max(f_values)

        residuals = f_values - ifnb
        res_list.append(residuals)
        rmsd = np.sqrt(np.mean(residuals**2))
        rmsd_list.append(rmsd)

    ifnb_predicted = pd.concat([ifnb_predicted, pd.DataFrame({"model": [model_name]*len(pars_list),
                                                                "pars": pars_list,
                                                                "dset": dset,
                                                                "residuals": res_list,
                                                                "rmsd": rmsd_list})])
    
ifnb_predicted.to_csv("../data/params_fits_3site_models.csv", index=False)
