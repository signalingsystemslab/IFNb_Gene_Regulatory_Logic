import matplotlib.pyplot as plt
import pandas as pd
plt.style.use("~/IFN_paper/src/theme_bw.mplstyle")

font = {'size'   : 20}
plt.rc('font', **font)

# Load the data, which has columns: IRF,NFkB,IFNb,p50,Stimulus,Genotype
training_data = pd.read_csv("../data/training_data.csv")
num_pts = training_data.shape[0]

# Plot the training data with IRF and NFkB on the x and y axes, color as IFNb, and shape corresponding to receptor
receptor = {"LPS": "TLR",
            "polyIC": "RLR",
            "CpG": "TLR"}
shape = {"TLR": "o",
            "RLR": "s"}
fig = plt.figure()

for stim in receptor.keys():
    data = training_data.loc[(training_data["Stimulus"]==stim)]
    plt.scatter(data["IRF"], data["NFkB"], c=data["IFNb"], marker=shape[receptor[stim]], 
                label=receptor[stim], cmap="RdYlBu_r", vmin=0, vmax=1, s=200)
fig.gca().set_ylabel(r"$NF\kappa B$")
fig.gca().set_xlabel(r"$IRF$")
plt.colorbar(format="%.1f")
plt.title("Observed data")
plt.grid(False)

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
fig.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.2,0.5))
plt.savefig("../figs/observed_data.svg", bbox_inches="tight")