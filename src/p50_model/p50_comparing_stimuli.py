from p50_model  import *
import matplotlib.pyplot as plt
import pandas as pd
import os
plt.style.use("~/IFN_paper/src/theme_bw.mplstyle")

dir="./figures/stimuli"
os.makedirs(dir, exist_ok=True)

stimulus_data = pd.read_csv("../data/stimulus_data.csv")
params_data = pd.read_csv("../data/p50_params.csv")


# Make predictions for models B2 and B4
params_B2 = params_data.loc[params_data["model"] == "B2", "params"].values[0].strip("[]").split()
params_B2= np.array(params_B2).astype(float)
params_B4 = params_data.loc[params_data["model"] == "B4", "params"].values[0].strip("[]").split()
params_B4= np.array(params_B4).astype(float)


results_df = pd.DataFrame(columns=["model", "stimulus", "p50","ifnb"])
for i in range(len(stimulus_data)):
    row = stimulus_data.iloc[i]
    # print("row: ", row)
    stimulus_name = row["Stimulus"]
    irf = row["IRF"]
    nfkb = row["NFkB"]

    f = explore_modelp50(params_B2, nfkb, irf, row["p50"], model_name="B2")
    results_df = pd.concat([results_df, pd.DataFrame({"model": "B2", "stimulus": stimulus_name, "p50": row["p50"], "ifnb": f}, index=[0])], ignore_index=True)
    f = explore_modelp50(params_B4, nfkb, irf, row["p50"], model_name="B4")
    results_df = pd.concat([results_df, pd.DataFrame({"model": "B4", "stimulus": stimulus_name, "p50": row["p50"], "ifnb": f}, index=[0])], ignore_index=True)

print(results_df)

# Make a graph of B2 results
fig = plt.figure()
p50_labs = {1: "WT", 0: "p50 KO"}
for p50 in p50_labs.keys():
    x = results_df.loc[(results_df["model"] == "B2") & (results_df["p50"] == p50), "stimulus"]
    y = results_df.loc[(results_df["model"] == "B2") & (results_df["p50"] == p50), "ifnb"]
    plt.plot(x, y, label=p50_labs[p50], marker="o", linestyle="None")
fig.legend(bbox_to_anchor=(1.1, 0.5))
plt.xticks(rotation=90)
plt.ylabel("IFNb")
plt.xlabel("Stimulus")
plt.title(" Predictions of p50 stimulus response for model B2")
plt.savefig(f"{dir}/B2_p50_stimulus_response.png", bbox_inches="tight")

# Make a graph of B4 results
fig = plt.figure()
for p50 in p50_labs.keys():
    x = results_df.loc[(results_df["model"] == "B4") & (results_df["p50"] == p50), "stimulus"]
    y = results_df.loc[(results_df["model"] == "B4") & (results_df["p50"] == p50), "ifnb"]
    plt.plot(x, y, label=p50_labs[p50], marker="o", linestyle="None")
fig.legend(bbox_to_anchor=(1.1, 0.5))
plt.xticks(rotation=90)
plt.ylabel("IFNb")
plt.xlabel("Stimulus")
plt.title(" Predictions of p50 stimulus response for model B4")
plt.savefig(f"{dir}/B4_p50_stimulus_response.png", bbox_inches="tight")
