from p50_model import *
import matplotlib.pyplot as plt
import pandas as pd
import itertools
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

# Model f as a function of p50
P = np.linspace(0, 2, 50)

# Make a data frame of I, N, and P values and filenames
df = pd.DataFrame(columns=["I", "N", "P", "filename", "ratio_B1"])
vals = np.logspace(-3, 0, 10)


i=0
for comb in itertools.combinations_with_replacement(vals, 2):
    I, N = comb
    df.loc[i] = [I, N, P, "I%.3f_N%.3f" % (I, N),0]
    rB1, rB2 = make_p50_plots(I, N, P, "I%.3f_N%.3f" % (I, N))
    df.loc[i, "ratio_B1"] = rB1
    if I != N:
        N, I = comb
        df.loc[i+1] = [I, N, P, "I%.3f_N%.3f" % (I, N),0]
        rB1, rB2 = make_p50_plots(I, N, P, "I%.3f_N%.3f" % (I, N))
        df.loc[i+1, "ratio_B1"] = rB1
        i+=2
    else:
        i+=1
    print("Finished I=%.3f, N=%.3f and reverse" % (I, N))

print(df["filename"])
pd.DataFrame.to_csv(df, "../data/model_B1_p50_ratios.csv")
df = pd.read_csv("../data/model_B1_p50_ratios.csv")
print(df)

# Plot I vs N
plt.figure()
plt.scatter(df["I"], df["N"], c=df["ratio_B1"], cmap="viridis")
# Set colormap boundaries to include 1.0
plt.clim(1.0, np.max(df["ratio_B1"]))
plt.xlabel(r"IRF")
plt.ylabel(r"NF$\kappa$B")
plt.title(r"Fold change IFN$\beta$ w/ p50 KO, model B1")
plt.colorbar(label=r"Fold change")
plt.savefig("../p50_model/figures/model_B1_p50_ratio.png")

# remove duplicates for I and N
df = df.drop_duplicates(subset=["I", "N"])

# Make an array of ratio B1 values where rows are I and columns are N
df_sorted = df.sort_values(by=["I", "N"])
vals = np.unique(df_sorted["I"])

ratio_B1 = np.zeros((len(vals), len(vals)))
for i in range(len(vals)):
    ratio_B1[i,:] = df_sorted.loc[df_sorted["I"]==vals[i], "ratio_B1"].values
    # print(vals[i])
    # print(df_sorted.loc[df_sorted["I"]==vals[i], "ratio_B1"].values)


plt.figure()
levels = np.linspace(1.0, np.max(df["ratio_B1"]), 8)
levels = np.round(levels, 2)
plt.contourf(vals, vals, ratio_B1, cmap="viridis", levels=levels)
plt.colorbar(label=r"Fold change")
plt.xlabel(r"IRF")
plt.ylabel(r"NF$\kappa$B")
plt.title(r"Fold change IFN$\beta$ w/ p50 KO, model B1")
plt.savefig("../p50_model/figures/model_B1_p50_ratio_contour.png")
