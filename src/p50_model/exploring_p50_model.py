from p50_model import *
import matplotlib.pyplot as plt
import pandas as pd
import itertools
from multiprocessing import Pool
import os
import time
plt.style.use("~/IFN_paper/src/theme_bw.mplstyle")


def make_p50_plots(N, I, P, filename):
    colors_b2 = ["#E24A33","#348ABD","#988ED5"]
    best_fit_color="#EF4A5F"
    t_B2 = np.loadtxt("../data/model"+"B2"+"_best_fit.csv", delimiter=",")
    f_values = np.zeros((len(P), t_B2.shape[0]))
    f_max =1
    ki2 = np.zeros(t_B2.shape[0])
    for i in range(len(P)):
        for j in range(t_B2.shape[0]):
            t = t_B2[j,:]
            if i ==0:
                ki2[j] = t[0]
            # print(" t = ", t)
            # print("p50 = ", P[i])
            try:
                f = explore_modelp50(t, N, I, P[i])
            except ValueError:
                print("i = ", i, "\n j = ", j)
                raise ValueError("t = ", t, "p50 = ", P[i])

            f_values[i,j] = f
            


    f_max = explore_modelp50(t_B2[0,:], 1, 1, P[-1])
    f_values = f_values / f_max
    ratio_B2 = f_values[0,0]/f_values[-1,0]

    indices = np.argwhere(ki2<1).flatten()
    low=f_values[:,indices]
    indices = np.argwhere(ki2>1).flatten()
    high=f_values[:,indices]
    indices = np.argwhere(ki2==1).flatten()
    equal=f_values[:,indices]

    # rows of f are p50 values, columns are model parameter sets
    # Plot f as a function of p50 for all parameter sets combined
    plt.figure()
    plt.plot(P, low, color=colors_b2[0], alpha=0.5, label="ki2 < ki1")
    plt.plot(P, high, color=colors_b2[1], alpha=0.5, label="ki2 > ki1")
    plt.plot(P, equal, color=colors_b2[2], alpha=0.5, label="ki2 = ki1")
    plt.xlabel("p50")
    plt.ylabel(r"IFN$\beta$ mRNA relative to p50=0, I=%.2f, N=%.2f" % (I, N))
    plt.title(r"IFN$\beta$ mRNA as a function of p50, model B2")
    handles, labels = plt.gca().get_legend_handles_labels()
    newLabels, newHandles = [], []
    for handle, label in zip(handles, labels):
        if label not in newLabels:
            newLabels.append(label)
            newHandles.append(handle)
    plt.legend(newHandles, newLabels)
    plt.savefig("../p50_model/figures/model_B2_p50_" + filename + ".png")
    plt.close()

    # Create a boxplot of f values at 5 p50 values
    p50_values = [int(np.round(len(P)/5)*i) for i in range(5)] + [len(P)-1]
    f_values_p50 = f_values[p50_values,:]
    plt.figure()
    plt.boxplot(f_values_p50.T)
    plt.xticks([1,2,3,4,5,6], [("%.2f" % P[i]) for i in p50_values])
    plt.xlabel("p50")
    plt.ylabel(r"IFN$\beta$ mRNA")
    plt.title(r"IFN$\beta$ mRNA at different p50 values, model B2")
    plt.savefig("../p50_model/figures/model_B2_p50_boxplot" + filename + ".png")
    plt.close()


    # Normalize f_values by p50=0
    f_values = f_values / f_values[0,:]
    indices = np.argwhere(ki2<1).flatten()
    low=f_values[:,indices]
    indices = np.argwhere(ki2>1).flatten()
    high=f_values[:,indices]
    indices = np.argwhere(ki2==1).flatten()
    equal=f_values[:,indices]

    plt.figure()
    plt.plot(P, low, color=colors_b2[0], alpha=0.5, label="ki2 < ki1")
    plt.plot(P, high, color=colors_b2[1], alpha=0.5, label="ki2 > ki1")
    plt.plot(P, equal, color=colors_b2[2], alpha=0.5, label="ki2 = ki1")
    plt.xlabel("p50")
    plt.ylabel(r"Change in IFN$\beta$ mRNA with addition of p50 vs p50=0")
    plt.title(r"Change in IFN$\beta$ mRNA as a function of p50, model B2")
    handles, labels = plt.gca().get_legend_handles_labels()
    newLabels, newHandles = [], []
    for handle, label in zip(handles, labels):
        if label not in newLabels:
            newLabels.append(label)
            newHandles.append(handle)
    plt.legend(newHandles, newLabels)
    plt.savefig("../p50_model/figures/model_B2_p50_normalized" + filename + ".png")
    plt.close()


    t_B1 = np.loadtxt("../data/model"+"B1"+"_best_fit.csv", delimiter=",")
    f_values = np.zeros((len(P), t_B1.shape[0]))
    f_max =1

    for i in range(len(P)):
        for j in range(t_B1.shape[0]):
            t = t_B1[j,:]
            try:
                f = explore_modelp50(t, N, I, P[i], model_name="B1")
            except ValueError:
                print("i = ", i, "\n j = ", j)
                raise ValueError("t = ", t, "\np50 = ", P[i])
            # f = explore_modelp50(t, N, I, P[i], model_name="B1")
            f_values[i,j] = f

    f_max = explore_modelp50(t_B1[0,:], 1, 1, P[-1], model_name="B1")
    f_values = f_values / f_max
    ratio_B1 = f_values[0,0]/f_values[-1,0]

    plt.figure()
    plt.plot(P, f_values[:,1:-1], color="black")
    plt.plot(P, f_values[:,0], color=best_fit_color, label="best fit")
    plt.xlabel("p50")
    plt.ylabel(r"IFN$\beta$ mRNA relative to p50=0, I=%.2f, N=%.2f" % (I, N))
    plt.title(r"IFN$\beta$ mRNA as a function of p50, model B1")
    plt.legend()
    plt.savefig("../p50_model/figures/model_B1_p50" + filename + ".png")
    plt.close()

    f_values_p50 = f_values[p50_values,:]
    plt.figure()
    plt.boxplot(f_values_p50.T)
    plt.xticks([1,2,3,4,5,6], [("%.2f" % P[i]) for i in p50_values])
    plt.xlabel("p50")
    plt.ylabel(r"IFN$\beta$ mRNA")
    plt.title(r"IFN$\beta$ mRNA at different p50 values, model B1")
    plt.savefig("../p50_model/figures/model_B1_p50_boxplot" + filename + ".png")
    plt.close()

    f_values = f_values / f_values[0,:]

    plt.figure()
    plt.plot(P, f_values[:,1:-1], color="black")
    plt.plot(P, f_values[:,0], color=best_fit_color, label="best fit")
    plt.xlabel("p50")
    plt.ylabel(r"Change in IFN$\beta$ mRNA with addition of p50 vs p50=0")
    plt.title(r"Change in IFN$\beta$ mRNA as a function of p50, model B1")
    plt.legend()
    plt.savefig("../p50_model/figures/model_B1_p50_normalized" + filename + ".png")
    plt.close()

    return ratio_B1, ratio_B2

# # Model f as a function of p50
# P = np.linspace(0, 2, 50)

# # Make a data frame of I, N, and P values and filenames
# df = pd.DataFrame(columns=["I", "N", "P", "filename", "ratio_B1"])
# vals = np.logspace(-3, 0, 10)


# i=0
# for comb in itertools.combinations_with_replacement(vals, 2):
#     I, N = comb
#     df.loc[i] = [I, N, P, "I%.3f_N%.3f" % (I, N),0]
#     rB1, rB2 = make_p50_plots(I, N, P, "I%.3f_N%.3f" % (I, N))
#     df.loc[i, "ratio_B1"] = rB1
#     if I != N:
#         N, I = comb
#         df.loc[i+1] = [I, N, P, "I%.3f_N%.3f" % (I, N),0]
#         rB1, rB2 = make_p50_plots(I, N, P, "I%.3f_N%.3f" % (I, N))
#         df.loc[i+1, "ratio_B1"] = rB1
#         i+=2
#     else:
#         i+=1
#     print("Finished I=%.3f, N=%.3f and reverse" % (I, N))

# print(df["filename"])
# pd.DataFrame.to_csv(df, "../data/model_B1_p50_ratios.csv")
# df = pd.read_csv("../data/model_B1_p50_ratios.csv")
# print(df)

# # Plot I vs N
# plt.figure()
# plt.scatter(df["I"], df["N"], c=df["ratio_B1"], cmap="viridis")
# # Set colormap boundaries to include 1.0
# plt.clim(1.0, np.max(df["ratio_B1"]))
# plt.xlabel(r"IRF")
# plt.ylabel(r"NF$\kappa$B")
# plt.title(r"Fold change IFN$\beta$ w/ p50 KO, model B1")
# plt.colorbar(label=r"Fold change")
# plt.savefig("../p50_model/figures/model_B1_p50_ratio.png")

# # remove duplicates for I and N
# df = df.drop_duplicates(subset=["I", "N"])

# # Make an array of ratio B1 values where rows are I and columns are N
# df_sorted = df.sort_values(by=["I", "N"])
# vals = np.unique(df_sorted["I"])

# ratio_B1 = np.zeros((len(vals), len(vals)))
# for i in range(len(vals)):
#     ratio_B1[i,:] = df_sorted.loc[df_sorted["I"]==vals[i], "ratio_B1"].values
#     # print(vals[i])
#     # print(df_sorted.loc[df_sorted["I"]==vals[i], "ratio_B1"].values)


# plt.figure()
# levels = np.linspace(1.0, np.max(df["ratio_B1"]), 8)
# levels = np.round(levels, 2)
# plt.contourf(vals, vals, ratio_B1, cmap="viridis", levels=levels)
# plt.colorbar(label=r"Fold change")
# plt.xlabel(r"IRF")
# plt.ylabel(r"NF$\kappa$B")
# plt.title(r"Fold change IFN$\beta$ w/ p50 KO, model B1")
# plt.savefig("../p50_model/figures/model_B1_p50_ratio_contour.png")

# With final parameters, get f values for a range of NFkB and IRF values in WT and KO
dir = "results/comparing_genotypes/"
os.makedirs(dir, exist_ok=True)

# Get final parameters
def get_params(file):
    params = {}
    with open(file) as f:
        for line in f:
            (key, val) = line.split(",")
            params[key] = float(val)
    return params

def return_N(N,I):
    return N

def return_I(N,I):
    return I

print("Getting parameters")
pars = get_params("../p50_model/results/random_opt/ifnb_best_params_random_global.csv")
# Unpack pars
t1 = pars["t1"]
t2 = pars["t2"]
t3 = pars["t3"]
t4 = pars["t4"]
t5 = pars["t5"]
t6 = pars["t6"]
t_pars = [t1, t2, t3, t4, t5, t6]
K = pars["K_i2"]
C = pars["C"]

# N_list = np.linspace(0, 1, 100)
# I_list = np.linspace(0, 1, 100)
# P_list = {"WT": 1, "KO": 0}
# f_values = np.zeros((len(N_list)*len(I_list), len(P_list)))
# N_values = np.zeros((len(N_list)*len(I_list), len(P_list)))
# I_values = np.zeros((len(N_list)*len(I_list), len(P_list)))

# # x=[return_N_I_P(N, I) for N in N_list for I in I_list]
# # print("Shape of x: %s, type of x: %s" % (str(np.shape(x)), str(type(x))))
# # f_test = np.zeros((len(N_list), len(I_list)))
# # print(x)
# # x = np.array(x).reshape((len(N_list), len(I_list)))
# # print("Shape of x: %s, type of x: %s" % (str(np.shape(x)), str(type(x))))
# # print(x)
# print("Starting multiprocessing")
# start = time.time()
# with Pool(30) as pl:
#     for i in range(len(P_list)):
#         p = list(P_list.keys())[i]
#         print("Calculating f values for %s" % p)
#         # get_f(t_pars, K, C, N, I, P, model_name="B2", scaling=1)
#         f_values[:,i] = pl.starmap(get_f, [(t_pars, K, C, N, I, P_list[p], "B2", 1) for N in N_list for I in I_list])
#         N_values[:,i] = pl.starmap(return_N, [(N, I) for N in N_list for I in I_list])
#         I_values[:,i] = pl.starmap(return_I, [(N, I) for N in N_list for I in I_list])
# end = time.time()
# print("Finished multiprocessing in %.2f minutes" % ((end-start)/60))

# f_values = f_values.reshape((len(N_list), len(I_list), len(P_list)))
# N_values = N_values.reshape((len(N_list), len(I_list), len(P_list)))
# I_values = I_values.reshape((len(N_list), len(I_list), len(P_list)))

# # Save all values
# np.save("%s/f_values.npy" % dir, f_values)
# np.save("%s/N_values.npy" % dir, N_values)
# np.save("%s/I_values.npy" % dir, I_values)

# # Calculate fold change KO vs WT
# f_values = np.load("%s/f_values.npy" % dir)
# N_values = np.load("%s/N_values.npy" % dir)
# I_values = np.load("%s/I_values.npy" % dir)

# # Replace 0 values in :,:,0 with 10e-10 to avoid divide by 0 error
# f_values[:,:,0][f_values[:,:,0]==0] = 10e-10
# fold_change = f_values[:,:,1] / f_values[:,:,0]

# # print N and I values that maximize fold change (top 5)
# print("Top 5 fold change values:")
# print(np.sort(fold_change.flatten())[-5:])
# print("N values:")
# print(N_values[:,:,0].flatten()[np.argsort(fold_change.flatten())[-5:]])
# print("I values:")
# print(I_values[:,:,0].flatten()[np.argsort(fold_change.flatten())[-5:]])

# print("Making plots")
# # Plot fold change
# plt.figure()
# plt.contourf(I_values[:,:,0], N_values[:,:,0], fold_change, 100, cmap="RdYlBu_r")
# plt.colorbar(label=r"Fold change IFN$\beta$ in KO vs WT")
# plt.xlabel(r"IRF")
# plt.ylabel(r"NF$\kappa$B")
# plt.title(r"Fold change IFN$\beta$ w/ p50 KO, model B2")
# plt.savefig("%s/fold_change_heatmap.png" % dir)
# plt.close()

# # Plot f values for WT and KO
# plt.figure()
# plt.contourf(I_values[:,:,0], N_values[:,:,0], f_values[:,:,0], 100, cmap="RdYlBu_r")
# plt.colorbar(label=r"IFN$\beta$ mRNA")
# plt.xlabel(r"IRF")
# plt.ylabel(r"NF$\kappa$B")
# plt.title(r"IFN$\beta$ mRNA in WT, model B2")
# plt.savefig("%s/WT_heatmap.png" % dir)
# plt.close()

# plt.figure()
# plt.contourf(I_values[:,:,1], N_values[:,:,1], f_values[:,:,1], 100, cmap="RdYlBu_r")
# plt.colorbar(label=r"IFN$\beta$ mRNA")
# plt.xlabel(r"IRF")
# plt.ylabel(r"NF$\kappa$B")
# plt.title(r"IFN$\beta$ mRNA in KO, model B2")
# plt.savefig("%s/KO_heatmap.png" % dir)

# # Calculate log 2 fold change
# log2_fold_change = np.log2(fold_change)
# plt.figure()
# plt.contourf(I_values[:,:,0], N_values[:,:,0], log2_fold_change, 100, cmap="RdYlBu_r")
# plt.colorbar(label=r"Log$_2$ fold change IFN$\beta$ in KO vs WT")
# plt.xlabel(r"IRF")
# plt.ylabel(r"NF$\kappa$B")
# plt.title(r"Log$_2$ fold change IFN$\beta$ w/ p50 KO, model B2")
# plt.savefig("%s/log2_fold_change_heatmap.png" % dir)

# # print("Shape = %s" % str(np.shape(f_values)))
# # print("\nF values for WT:")
# # print(f_values[:,:,0])
# # print("\nN values for WT:")
# # print(N_values[:,:,0])
# # print("\nI values for WT:")
# # print(I_values[:,:,0])

# ## How much do t parameters affect ifnb expression?
# # Determine sensitivity to parameters t1-t6
# t_pars = np.linspace(0.1, 1, 10)
# t_pars = np.meshgrid(t_pars, t_pars, t_pars, t_pars, t_pars, t_pars)
# t_pars = np.array(t_pars).T.reshape(-1, 6)
# k_pars = [2 for i in range(len(t_pars))]
# # Append k_pars to t_pars
# t_pars = np.hstack((t_pars, np.array(k_pars).reshape(-1,1)))

# print(t_pars.shape)
# print(len(t_pars))
# print([t_pars[i,:] for i in range(5)])

# # f_values = np.zeros((len(t_pars)))
# # f_values_KO = np.zeros((len(t_pars)))
# # print("Starting multiprocessing")
# # start = time.time()
# # with Pool(30) as p:
# #     f_values[:] = p.starmap(get_f, [(t_pars[i,:], K, C, 1, 1, 1, "B2", 1) for i in range(len(t_pars))])
# #     f_values_KO[:] = p.starmap(get_f, [(t_pars[i,:], K, C, 1, 1, 0, "B2", 1) for i in range(len(t_pars))])
# # end = time.time()
# # print("Finished multiprocessing in %.2f minutes" % ((end-start)/60))

# # #  get_f(t_pars, K, C, N, I, P, model_name="B2", scaling=1)

# # # Save f values
# # np.save("%s/f_values_t_pars.npy" % dir, f_values)
# # np.save("%s/f_values_KO_t_pars.npy" % dir, f_values_KO)
# # np.save("%s/t_pars.npy" % dir, t_pars)
# # print("Saved f values")

# # Plot f values
# start = time.time()
# print("Loading f values")
# f_values = np.load("%s/f_values_t_pars.npy" % dir)
# f_values_KO = np.load("%s/f_values_KO_t_pars.npy" % dir)
# t_pars = np.load("%s/t_pars.npy" % dir)
# print("Loaded f values")

# # # Plot f values for each parameter as scatter plots
# # for i in range(6):
# #     print("Plotting scatter plot for WT t%d" % (i+1))
# #     plt.figure()
# #     plt.scatter(t_pars[:,i], f_values)
# #     plt.xlabel("t%d" % (i+1))
# #     plt.ylabel(r"IFN$\beta$ mRNA")
# #     plt.title(r"IFN$\beta$ mRNA vs t%d, model B2" % (i+1))
# #     plt.savefig("%s/f_values_t%d.png" % (dir, i+1))

# # # KO
# # for i in range(6):
# #     print("Plotting scatter plot for KO t%d" % (i+1))
# #     plt.figure()
# #     plt.scatter(t_pars[:,i], f_values_KO)
# #     plt.xlabel("t%d" % (i+1))
# #     plt.ylabel(r"IFN$\beta$ mRNA")
# #     plt.title(r"IFN$\beta$ mRNA vs t%d, model B2" % (i+1))
# #     plt.savefig("%s/f_values_KO_t%d.png" % (dir, i+1))

# # # Fold change KO/WT
# fold_change = f_values_KO / f_values
# # for i in range(6):
# #     print("Plotting scatter plot for fold change vs t%d" % (i+1))
# #     plt.figure()
# #     plt.scatter(t_pars[:,i], fold_change)
# #     plt.xlabel("t%d" % (i+1))
# #     plt.ylabel(r"Fold change IFN$\beta$ mRNA")
# #     plt.title(r"Fold change IFN$\beta$ mRNA vs t%d, model B2" % (i+1))
# #     plt.savefig("%s/fold_change_t%d.png" % (dir, i+1))

# # # Plot f values with each two parameters held constant
# # fig, ax = plt.subplots(6,6, figsize=(10,10))
# # for i in range(6):
# #     for j in range(6):
# #         print("Plotting scatter plot for t%d vs t%d" % (i+1, j+1))
# #         if i != j:
# #             ax[i,j].scatter(t_pars[:,i], t_pars[:,j], c=fold_change, cmap="viridis")
# #             ax[i,j].set_xlabel("t%d" % (i+1))
# #             ax[i,j].set_ylabel("t%d" % (j+1))
# #             # ax[i,j].set_title(r"IFN$\beta$ mRNA vs t%d and t%d, model B2" % (i+1, j+1))
# #         else:
# #             ax[i,j].scatter(t_pars[:,i], f_values)
# #             ax[i,j].set_xlabel("t%d" % (i+1))
# #             ax[i,j].set_ylabel(r"IFN$\beta$ mRNA")
 
# # plt.tight_layout()
# # plt.savefig("%s/fold_change_t_pars_heatmap.png" % dir)

# # end = time.time()
# # print("Finished in %.2f minutes" % ((end-start)/60))

# # Print t_pars that maximize fold change
# print("Parameters which give the biggest fold change:")
# print(t_pars[np.argmax(fold_change),:])
# print("Fold change = %.2f" % np.max(fold_change))


# Unpack pars
t1 = 0.01
t2 = 1.0
t3 = 0.01
t4 = 1.0
t5 = 0.01
t6 = 1.0
t_pars = [t1, t2, t3, t4, t5, t6]
print("Calculating f values for t-pars %s" % str(t_pars))

# Calculate f-values for WT and KO
WT_test_f = get_f(t_pars, K, C, 1, 1, 1, "B2", 1)
KO_test_f = get_f(t_pars, K, C, 1, 1, 0, "B2", 1)
print("WT f = %.2f, KO f = %.2f, KO/WT = %.2f" % (WT_test_f, KO_test_f, KO_test_f/WT_test_f))

