from two_site_model import *
import matplotlib.pyplot as plt
import pandas as pd
import os
import scipy.optimize as opt
import time
from multiprocessing import Pool
import seaborn as sns
plt.style.use("~/IFN_paper/src/theme_bw.mplstyle")

optimization_dir = "./grid_opt/results"
results_dir = "./explore_two_site_model_results/"
os.makedirs(results_dir, exist_ok=True)

def plot_state_probabilities(state_probabilities, state_names, name, figures_dir=results_dir):
        stimuli = ["basal", "LPS", "polyIC"]
        stimulus = [s for s in stimuli if s in name]

        if len(stimulus) == 0:
            stimulus = "No Stim"
            condition = "No Stim"
        elif len(stimulus) > 1:
            raise ValueError("More than one stimulus in name")
        else:
            stimulus = stimulus[0]
            # Condition is text after stimulus_
            name_parts = name.split("_")
            stim_loc = name_parts.index(stimulus)
            cond_loc = stim_loc + 1
            genotype = name_parts[cond_loc]
            condition = "%s %s" % (stimulus, genotype)
        df_state_probabilities = pd.DataFrame(state_probabilities, columns=state_names)
        df_state_probabilities["par_set"] = np.arange(len(df_state_probabilities))
        df_state_probabilities = df_state_probabilities.melt(var_name="State", value_name="Probability", id_vars="par_set")

        fig, ax = plt.subplots()
        p = sns.lineplot(data=df_state_probabilities, x = "State", y="Probability", color="black", alpha=0.5,
                            estimator=None, units="par_set", legend=False).set_title(condition)
        sns.despine()
        plt.xticks(rotation=90)
        # Save plot
        plt.savefig("%s/%s.png" % (figures_dir, name), bbox_inches="tight")
        plt.close()

def plot_predictions(ifnb_predicted, beta, conditions, name="ifnb_predictions", figures_dir=results_dir):
        df_ifnb_predicted = pd.DataFrame({"Condition":conditions, r"IFN$\beta$":ifnb_predicted, "par_set":"Predicted"})
        df_ifnb_data = pd.DataFrame({"Condition":conditions, r"IFN$\beta$":beta, "par_set":"Data"})
        df_ifnb_predicted = pd.concat([df_ifnb_predicted, df_ifnb_data], ignore_index=True)
        df_ifnb_predicted["Genotype"] = df_ifnb_predicted["Condition"].str.split("_", expand=True)[0]
        df_ifnb_predicted["Stimulus"] = df_ifnb_predicted["Condition"].str.split("_", expand=True)[1]

        fig, ax = plt.subplots()
        sns.lineplot(data=df_ifnb_predicted.loc[df_ifnb_predicted["par_set"] == "Predicted"], x="Condition", y=r"IFN$\beta$",
                     units="par_set", color="black", alpha=0.5, estimator=None, ax=ax)
        sns.scatterplot(data=df_ifnb_predicted.loc[df_ifnb_predicted["par_set"] == "Data"], x="Condition", y=r"IFN$\beta$",
                        color="red", marker="o", ax=ax, legend=False)
        sns.despine()
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig("%s/%s.png" % (figures_dir, name))
        plt.close()

def plot_parameters(pars, name, figures_dir):
    df_pars = pd.DataFrame(pars)
    df_pars["par_set"] = np.arange(len(df_pars))
    df_pars = df_pars.melt(var_name="Parameter", value_name="Value", id_vars="par_set")
    param_names = df_pars["Parameter"].unique()

    fig, ax = plt.subplots()
    sns.displot(data=df_pars, x="Value", kde=True)
    plt.title("Distribution of parameter %s" % param_names[0])
    sns.despine()
    plt.tight_layout()
    plt.savefig("%s/%s.png" % (figures_dir, name))
    plt.close()


def main():
    pars = pd.read_csv("%s/ifnb_best_params_grid_global.csv" % optimization_dir, index_col=0, header=None)
    best_model = "IRF"
    C = pars.loc["C", 1]

    # Determine relative contribution of each TF
    N = np.linspace(0, 1, 50)
    I = np.linspace(0, 1, 50)

    n, i = np.meshgrid(N, I)
    f = np.zeros((len(N), len(I)))
    for j in range(len(N)):
        for k in range(len(I)):
            f[j,k] = get_f(C, n[j,k], i[j,k], best_model)
    
    # # save f to csv
    # np.savetxt("%s/f_%s.csv" % (results_dir, best_model), f, delimiter=",")
    # np.savetxt("%s/n_%s.csv" % (results_dir, best_model), n, delimiter=",")
    # np.savetxt("%s/i_%s.csv" % (results_dir, best_model), i, delimiter=",")

    # Plot minimum and maximum ifnb for each nfkb
    fig, ax = plt.subplots()
    ax.set_prop_cycle("color", plt.cm.viridis(np.linspace(0, 1, 5)))
    ax.plot(N, np.max(f, axis=0), label=r"Maximum IFN$\beta$", linewidth=3)
    ax.fill_between(N, np.min(f, axis=0), np.max(f, axis=0), alpha=0.2, label = "Contribution of IRF")
    ax.plot(N, np.min(f, axis=0), label=r"Minimum IFN$\beta$", linewidth=3)
    ax.set_xlabel(r"$NF\kappa B$")
    ax.set_ylabel(r"IFN$\beta$")
    ax.set_title("Model %s" % best_model)
    fig.legend(bbox_to_anchor=(1.23, 0.5))
    fig.savefig("%s/nfkb_vs_min_max_ifnb_%s.png" % (results_dir, best_model))

    # Plot minimum and maximum ifnb for each irf
    fig, ax = plt.subplots()
    ax.set_prop_cycle("color", plt.cm.viridis(np.linspace(0, 1, 5)))
    ax.plot(I, np.max(f, axis=1), label=r"Maximum IFN$\beta$", linewidth=3)
    ax.fill_between(I, np.min(f, axis=1), np.max(f, axis=1), alpha=0.2, label = "Contribution of NF$\kappa$B")
    ax.plot(I, np.min(f, axis=1), label=r"Minimum IFN$\beta$", linewidth=3)
    ax.set_xlabel(r"$IRF$")
    ax.set_ylabel(r"IFN$\beta$")
    ax.set_title("Model %s" % best_model)
    fig.legend(bbox_to_anchor=(1.25, 0.5))
    fig.savefig("%s/irf_vs_min_max_ifnb_%s.png" % (results_dir, best_model))

    # Plot best fits ifnb predictions
    training_data = pd.read_csv("../data/training_data.csv")
    print(training_data)
    nfkb = training_data["NFkB"]
    irf = training_data["IRF"]
    beta = training_data["IFNb"]
    conditions = training_data["Stimulus"] + "_" + training_data["Genotype"]
    
    ifnb_predicted = np.zeros(len(conditions))
    for j in range(len(conditions)):
        ifnb_predicted[j] = get_f(C, nfkb[j], irf[j], best_model)

    plot_predictions(ifnb_predicted, beta, conditions, name="ifnb_predictions")


if __name__ == "__main__":
    main()