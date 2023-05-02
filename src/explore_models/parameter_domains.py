import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

plt.style.use("~/IFN_paper/src/theme_bw.mplstyle")

t_names = ["t1", "t2", "t3", "t4", "t5", "t6"]
def load_parameters(model_name, pars_df=pd.DataFrame(columns=["model", "pars"])):
    pars_df = pars_df.copy().loc[pars_df["model"] == model_name]
   
    if model_name == "B1":
        parnames = t_names
    elif model_name == "B2":
        parnames = np.hstack(["K_i2", t_names])
    elif model_name == "B3":
        parnames = np.hstack(["C", t_names])
    elif model_name == "B4":
        parnames = np.hstack(["K_i2", "C", t_names])

    pars_df[parnames] = pars_df.loc[:,"pars"].str.strip("[]").str.split(expand=True).astype(float)
    pars_arr = np.array(pars_df[parnames].astype(float))
    pars_df = pars_df.drop("pars", axis=1)
    # print("pars_df with t columns and no pars column:")
    # print(pars_df)
    
    return parnames, pars_arr, pars_df


# Trying to find representative groups of parameters
model_names=["B1", "B2", "B3", "B4"]
for model_name in model_names:
    pars_df = pd.read_csv("../data/params_fits_3site_models.csv")

    p1, pars_arr, pars_df = load_parameters(model_name, pars_df)

    ## Testing approaches
    # perform k-means clustering on parameters
    kmeans = KMeans(n_clusters=5, random_state=0).fit(pars_arr)
    pars_df["cluster"] = kmeans.labels_

    plt.figure()
    plt.scatter(pars_df["t1"], pars_df["t2"], c=pars_df["cluster"])
    plt.xlabel("t1")
    plt.ylabel("t2")
    plt.colorbar()
    plt.title("Model " + model_name + r" t_1/t_2 parameters with clusters")
    plt.savefig("./figures/parameters/clusters_" + model_name + "_t1_t2.png")

    # Plot heatmap of parameters
    pars_sorted = pars_df.copy().sort_values(by=["cluster","rmsd"])
    clusters = pars_sorted["cluster"]
    pars_plt = pars_sorted.loc[:,t_names]

    plt.figure()
    plt.imshow(pars_plt, aspect="auto", cmap="viridis")
    # Add column for clusters
    plt.hlines(np.where(np.diff(clusters))[0], *plt.xlim(), color="white")
    plt.xticks(range(len(pars_plt.columns)), pars_plt.columns, rotation=90)
    plt.colorbar()
    plt.xlabel("Parameters")
    plt.ylabel("Values")
    plt.grid(False)
    plt.title("Model " + model_name + " parameters with clusters")
    plt.savefig("./figures/parameters/heatmap_" + model_name + "_clusters.png")

    # Selecting representative parameters: take the first parameter set in each cluster (lowest rmsd)
    pars_rep = pars_sorted.groupby("cluster").first()
    print("Representative parameters for model " + model_name + ":")
    print(pars_rep)
    pars_rep.to_csv("../data/params_fits_" + model_name + "_representative.csv")