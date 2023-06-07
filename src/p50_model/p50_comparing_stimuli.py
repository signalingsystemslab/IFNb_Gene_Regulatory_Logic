from p50_model  import *
import matplotlib.pyplot as plt
import pandas as pd
import os
plt.style.use("~/IFN_paper/src/theme_bw.mplstyle")

dir="./figures/stimuli"
os.makedirs(dir, exist_ok=True)

stimulus_data = pd.read_csv("../data/stimulus_data.csv")
params_data = pd.read_csv("../data/p50_local_optimization_results")

results_df = pd.DataFrame(columns=["model", "stimulus", "p50","ifnb"])
for i in range(len(params_data)):
    model = params_data.index[i]
    params = params_data.iloc[i][0:6]
    if model == "B2":
        params = np.hstack([params_data.iloc[i][6], params])
    elif model == "B3":
        params = np.hstack([params_data.iloc[i][6], params])
    elif model == "B4":
        params = np.hstack([params_data.iloc[i][6:8], params])
    for i in range(len(stimulus_data)):
        row = stimulus_data.iloc[i]
        # print("row: ", row)
        stimulus_name = row["Stimulus"]
        irf = row["IRF"]
        nfkb = row["NFkB"]

        f = explore_modelp50(params, nfkb, irf, row["p50"], model_name=model)
        results_df = pd.concat([results_df, pd.DataFrame({"model": "B2", "stimulus": stimulus_name, "p50": row["p50"], "ifnb": f}, index=[0])], ignore_index=True)
        
results_df.to_csv(f"{dir}/stimulus_results.csv", index=False)

# Plot results showing fold change of p50 KO vs WT
results_df = pd.read_csv(f"{dir}/stimulus_results.csv")
results_fc =results_df.copy()
results_fc = results_fc.pivot(index=["model", "stimulus"], columns="p50", values="ifnb")
results_fc["fc"] = results_fc[0] / results_fc[1]
results_fc = results_fc.reset_index()
print(results_fc)

for i in range(len(results_df)):
    row = results_df.iloc[i]
    if row["p50"] == 0:
        results_df.loc[i, "p50"] = "WT"
    elif row["p50"] == 1:
        results_df.loc[i, "p50"] = "KO"

    fig = plt.figure()
    x = results_fc.loc[results_fc["model"] == row["model"], "stimulus"]
    y = results_fc.loc[results_fc["model"] == row["model"], "fc"]
    plt.plot(x, y, marker="o", linestyle="None")
    plt.xticks(rotation=90)
    plt.ylabel(r"IFN$\beta$ fold change")
    plt.xlabel("Stimulus")
    plt.title("Fold change of p50 KO vs WT for model B2")
    plt.savefig(f"{dir}/%s_p50_stimulus_response_fc.png" % row["model"], bbox_inches="tight")


