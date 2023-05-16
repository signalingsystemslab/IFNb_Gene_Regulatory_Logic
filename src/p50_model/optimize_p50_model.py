# For each model (B1, B2, B3, B4), identify parameter sets that best fit the data
# TBD: Maybe I won't use this
from p50_model import *
import matplotlib.pyplot as plt
import pandas as pd
import os
plt.style.use("~/IFN_paper/src/theme_bw.mplstyle")

os.makedirs("./figures/optimized/", exist_ok=True)

t_names = ["t1", "t2", "t3", "t4", "t5", "t6"]
parnames_dict = {"B1": t_names, "B2": np.hstack(["K_i2", t_names]), "B3": np.hstack(["C", t_names]), "B4": np.hstack(["K_i2", "C", t_names])}

def load_parameters(model_name):
    pars_df = pd.read_csv("../data/params_fits_3site_models.csv")
    pars_df = pars_df.copy().loc[pars_df["model"] == model_name]
    parnames = parnames_dict[model_name]

    print("\nRMSD for all params for model %s:" % model_name)
    print(pars_df["rmsd"].describe())

    pars_df = pars_df.loc[pars_df["rmsd"] < 0.15]
    print("%d parameter sets with RMSD < 0.15" % pars_df.shape[0])

    pars_df[parnames] = pars_df.loc[:,"pars"].str.strip("[]").str.split(expand=True).astype(float)
    pars_arr = np.array(pars_df[parnames].astype(float))
    pars_df = pars_df.drop("pars", axis=1)

    if pars_df.shape[0] != pars_arr.shape[0]:
        raise ValueError("pars_df has %d rows and pars_arr has %d rows" % (pars_df.shape[0], pars_arr.shape[0]))

    # print("pars_df with t columns and no pars column:")
    # print(pars_df)
    
    return parnames, pars_arr, pars_df

def plot_contour(f_values, model_name, I, N, p50,normalize=True):
    if normalize:
        f_values = f_values / np.max(f_values)
    fig=plt.figure()
    plt.contourf(I,N, f_values, 100, cmap="RdYlBu_r")
    plt.grid(False)
    plt.colorbar(format="%.1f")
    plt.title("Model "+model_name+" best fit, p50 = %s" % p50)
    fig.gca().set_ylabel(r"$NF\kappa B$")
    fig.gca().set_xlabel(r"$IRF$")
    plt.savefig("./figures/optimized/model%s_contour_p50_%s.png" % (model_name, p50))
    plt.close()

def calculateFvalues(model_name, pars, I, N, p50):
    f_values = np.zeros((len(N), len(I)))
    for n in range(len(N)):
        for i in range(len(I)):
            f_values[n,i] = explore_modelp50(pars,N[n], I[i],p50, model_name)
    return f_values

def find_best_parameters(model_name):
    parnames, pars_arr, pars_df = load_parameters(model_name)
    data = pd.read_csv("../data/p50_data.csv").drop("note", axis=1)
    
    f_values = np.zeros((pars_arr.shape[0], data.shape[0]))
    for row in range(pars_arr.shape[0]):
        f_values[row,:] = [explore_modelp50(pars_arr[row,:], data["NFkB"][i], data["IRF"][i], 
                                            data["p50"][i], model_name) for i in range(data.shape[0])]

    # # Normalize f_values so highest in each set is 0.4
    # row_max = np.max(f_values, axis=1)
    # f_values = f_values / row_max[:,None] * np.max(data["IFNb"])
    # rmsd = np.sqrt(np.mean((f_values - data["IFNb"].values[None,:])**2, axis=1))
    # best_params = pars_arr[np.argmin(rmsd),:]
    # best_dset = pars_df.iloc[np.argmin(rmsd), pars_df.columns.get_loc("dset")]

    #  Normalize f_values so lowest in each set is same as lowest in data
    row_min = np.min(f_values, axis=1)
    f_values = f_values / row_min[:,None] * np.min(data["IFNb"])
    rmsd = np.sqrt(np.mean((f_values - data["IFNb"].values[None,:])**2, axis=1))
    best_params = pars_arr[np.argmin(rmsd),:]
    best_dset = pars_df.iloc[np.argmin(rmsd), pars_df.columns.get_loc("dset")]

    I = np.linspace(0,1,100)
    N = I.copy()
    plot_contour(calculateFvalues(model_name, best_params, N, I, data["p50"][0]), model_name, I, N, data["p50"][0])
    plot_contour(calculateFvalues(model_name, best_params, N, I, data["p50"][1]), model_name, I, N, data["p50"][1])

    # #  Testing
    # m = Modelp50(best_params, model_name)
    # print("Beta for model %s: %s" % (model_name, m.beta))
    # elems = [1, "ki1", "ki2","ki1 kp", "kn", "kn kp", "kp", "ki1 ki2", "ki1 kn", "ki2 kn", "ki1 kp kn", "ki1 ki2 kn"]
    # for val, elem in zip(m.beta, elems):
    #     print("%s: %s" % (elem, val))
    return best_params, best_dset, rmsd[np.argmin(rmsd)]

def plot_ifnb_vs_p50(model_name, pars, I, N, plims=[0,2],condition=""):
    dir = "./figures/optimized/"
    os.makedirs(dir, exist_ok=True)
    model = pd.DataFrame(columns=["p50", "ifnb"])
    num_pvals = 100
    model["p50"] = np.linspace(plims[0], plims[1], num_pvals)


    fig6_data = pd.read_csv("../data/fig_6_p50_data.csv")
    # fig6_data["p50"] = [plims[0], model.iloc[50,model.columns.get_loc("p50")], plims[1]]
    # norm_p50 = np.rint((fig6_data["p50"][1]/fig6_data["p50"][2])*num_pvals).astype(int)
    norm_p50=0
    p50_value = model.iloc[norm_p50,model.columns.get_loc("p50")]
    # scale_factor = fig6_data.iloc[1, fig6_data.columns.get_loc("ifnb_norm")]

    print("Normalizing to p50 position %s, value %.3f" % (norm_p50, p50_value))
    model["ifnb"] = [explore_modelp50(pars, N, I, p50, model_name) for p50 in model["p50"]]
    model["ifnb"] = model["ifnb"] / model.iloc[norm_p50,model.columns.get_loc("ifnb")] * fig6_data.iloc[norm_p50, fig6_data.columns.get_loc("ifnb_norm")]
    
    fig = plt.figure()
    plt.plot(model["p50"], model["ifnb"], color="black", label="Model prediction")
    plt.scatter(fig6_data["p50"], fig6_data["ifnb_norm"], s=50, label="Data", zorder =2)
    plt.xlabel("p50")
    plt.ylabel(r"IFN$\beta$")
    plt.title("Model "+model_name+" "+condition+r" predicted IFN$\beta$ vs p50")
    fig.legend(bbox_to_anchor=(1.2,0.5))
    plt.savefig(dir+model_name+"_"+condition+".png")

    rmsd = np.sqrt(np.mean((model.iloc[[0,norm_p50,num_pvals-1],model.columns.get_loc("ifnb")].values - fig6_data["ifnb_norm"].values)**2))
    return rmsd

def plot_all_models(plims=[0,2]):
    dir = "./figures/optimized/"
    os.makedirs(dir, exist_ok=True)
    model = pd.DataFrame(columns=["p50", "ifnb"])
    model["p50"] = np.linspace(plims[0], plims[1], 101)

    fig6_data = pd.read_csv("../data/fig_6_p50_data.csv")
    # fig6_data["p50"] = [plims[0], model.iloc[50,model.columns.get_loc("p50")], plims[1]]
    # norm_p50 = np.rint((fig6_data["p50"][1]/fig6_data["p50"][2])*100).astype(int)
    norm_p50=0


    fig = plt.figure()
    for model_name in ["B1", "B2", "B3", "B4"]:
        best_params = find_best_parameters(model_name)[0]
        model["ifnb"] = [explore_modelp50(best_params, 0, 0.25, p50, model_name) for p50 in model["p50"]]
        model["ifnb"] = model["ifnb"] / model.iloc[norm_p50,model.columns.get_loc("ifnb")] * fig6_data.iloc[norm_p50, fig6_data.columns.get_loc("ifnb_norm")]
        plt.plot(model["p50"], model["ifnb"], label=model_name)
    plt.scatter(fig6_data["p50"], fig6_data["ifnb_norm"], s=50, label="Data", zorder =2, color="black")
    plt.xlabel("p50")
    plt.ylabel(r"IFN$\beta$")
    plt.ylim([0,1])
    plt.title("Model predictions for all models")
    fig.legend(bbox_to_anchor=(1.1,0.5))
    plt.savefig(dir+"all_models%d-%f.png" % (plims[0], plims[1]))


results = pd.DataFrame(columns=["model", "condition", "rmsd"])
params = pd.DataFrame(columns=["model", "params"])
for model_name in ["B1", "B2", "B3", "B4"]:

    best_params, best_dset, rmsd = find_best_parameters(model_name)
    print("Best parameters for model "+model_name+" are: ", best_params)
    params = pd.concat([params, pd.DataFrame({"model": model_name, "params": [best_params]})])
    print("RMSD for model "+model_name+" is: ", rmsd)

    # r = plot_ifnb_vs_p50(model_name, best_params, I=0.25, N=0, plims=[0,1], condition="NFkB_KO_0-1")
    # results = pd.concat([results, pd.DataFrame({"model": [model_name], "condition": ["NFkB_KO_0-1"], "rmsd": [r]})])

    # r = plot_ifnb_vs_p50(model_name, best_params, I=0.25, N=0, condition="NFkB_KO_0-2")
    # results = pd.concat([results, pd.DataFrame({"model": [model_name], "condition": ["NFkB_KO_0-2"], "rmsd": [r]})])

    r = plot_ifnb_vs_p50(model_name, best_params, I=0.25, N=0, plims=[0,2.5], condition="NFkB_KO_0-2.5")
    results = pd.concat([results, pd.DataFrame({"model": [model_name], "condition": ["NFkB_KO_0-2.5"], "rmsd": [r]})])

    # r = plot_ifnb_vs_p50(model_name, best_params, I=0.25, N=0, plims=[0,5], condition="NFkB_KO_0-5")
    # results = pd.concat([results, pd.DataFrame({"model": [model_name], "condition": ["NFkB_KO_0-5"], "rmsd": [r]})])

    # r = plot_ifnb_vs_p50(model_name, best_params, I=0.25, N=0, plims=[0,20], condition="NFkB_KO_0-20")
    # results = pd.concat([results, pd.DataFrame({"model": [model_name], "condition": ["NFkB_KO_0-20"], "rmsd": [r]})])

params.to_csv("../data/p50_params.csv")

print(results)

fig = plt.figure()
for condition in results["condition"].unique():
    plt.scatter(results[results["condition"]==condition]["model"], results[results["condition"]==condition]["rmsd"], label=condition)
# plt.scatter(results["model"], results["rmsd"], s=50)
plt.xlabel("Model")
plt.ylabel("RMSD")
plt.ylim([0,0.3])
fig.legend(bbox_to_anchor=(1.2,0.5))
plt.title(r"RMSD for model predictions of IFN$\beta$ vs p50")
plt.savefig("./figures/optimized/rmsd.png")

plot_all_models([0,2.5])

#  Plot scatter plot of all data
fig6_data = pd.read_csv("../data/fig_6_p50_data.csv")
fig6_data = fig6_data.rename(columns={"ifnb_norm": "IFNb"})
p50_data = pd.read_csv("../data/p50_data.csv")
exp_data = pd.read_csv("../data/exp_matrix_norm.csv", names=["IRF", "NFkB", "IFNb"])
exp_data["p50"] = 1

all_data = pd.concat([exp_data, fig6_data.loc[:, ["IRF", "NFkB", "IFNb", "p50"]],p50_data.loc[:, ["IRF", "NFkB", "IFNb","p50"]]], axis=0, ignore_index=True)
# print(all_data)

data_p50_1 = all_data[all_data["p50"]==1]

fig = plt.figure()
plt.scatter(data_p50_1["IRF"], data_p50_1["NFkB"], c=data_p50_1["IFNb"], cmap="RdYlBu_r", s=50)
plt.xlabel("IRF")
plt.ylabel(r"NF$\kappa$B")
plt.title(r"All data points with p50=1")
plt.colorbar(label=r"IFN$\beta$")
plt.savefig("./figures/optimized/all_data_p50_1.png")

data_p50_0 = all_data[all_data["p50"]==0]

fig = plt.figure()
plt.scatter(data_p50_0["IRF"], data_p50_0["NFkB"], c=data_p50_0["IFNb"], cmap="RdYlBu_r", s=50)
plt.xlabel("IRF")
plt.ylabel(r"NF$\kappa$B")
plt.xlim([0,1])
plt.ylim([-0.02,1])
plt.clim([0,1])
plt.title(r"All data points with p50=0")
plt.colorbar(label=r"IFN$\beta$")
plt.savefig("./figures/optimized/all_data_p50_0.png")


# b1_t_pars = [0.21992227, 0.75186586, 0.10526895, 0.9766144,  1., 1.]
# b2_t_pars = [0.01995081, 0.50188124, 0.13282681, 1., 1., 1.]
# x1=np.arange(6)-0.4
# x2=np.arange(6)

# fig = plt.figure()
# plt.bar(x1, b1_t_pars, label="B2", width=0.4, align="edge")
# plt.bar(x2, b2_t_pars, label="B4", width=0.4, align="edge")
# fig.legend(bbox_to_anchor=(1.05,0.5))
# plt.xticks(np.arange(6), [r"IRF_1", r"IRF_2", r"NF$\kappa$B", r"IRF_1 IRF_2", r"IRF_1 NF$\kappa$B", r"IRF_2 NF$\kappa$B"], rotation=45)
# plt.ylabel("Transcription capability (t)")
# plt.title("p50-p50 best fit parameters")
# plt.savefig("./figures/optimized/p50-p50_best_fit_tpars.png")
