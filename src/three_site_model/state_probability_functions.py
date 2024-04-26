import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def get_N_I_P(data, stimulus, genotype):
    row = data.loc[(data["Stimulus"] == stimulus) & (data["Genotype"] == genotype)]
    N = row["NFkB"].values[0]
    I = row["IRF"].values[0]
    P = row["p50"].values[0]
    return N, I, P

def get_state_prob(t_pars, k_pars, N, I, c_par=None, h_pars=None):
    model = three_site(t_pars, k_pars, c_par=c_par, h_pars=h_pars)
    model.calculateState(N, I)
    model.calculateProb()
    probabilities = model.prob
    
    # 1, I, Ig, N,  I*Ig, I*N, Ig*N, I*Ig*N
    state_names = ["none", r"$IRF$", r"$IRF_G$", r"$NF\kappa B$", r"$IRF\cdot IRF_G$", 
                   r"$IRF\cdot NF\kappa B$", r"$IRF_G\cdot NF\kappa B$", r"$IRF\cdot IRF_G\cdot NF\kappa B$"]

    return probabilities, state_names

def calc_state_prob(k_pars, N, I, P, num_t=6, h_pars=None):
    # print(N, I, P, flush=True)
    t_pars = [1 for _ in range(num_t)]
    probabilities, state_names = get_state_prob(t_pars, k_pars, N, I, P, h_pars=h_pars)
    return probabilities, state_names

def plot_state_probabilities(state_probabilities, state_names, name, figures_dir):
        stimuli = ["basal", "CpG", "LPS", "polyIC"]
        stimulus = [s for s in stimuli if s in name]
        if len(stimulus) == 0:
            stimulus = "No Stim"
        elif len(stimulus) > 1:
            raise ValueError("More than one stimulus in name")
        else:
            stimulus = stimulus[0]

        condition = name.split("_")[-2:]
        condition = " ".join(condition)
        df_state_probabilities = pd.DataFrame(state_probabilities, columns=state_names)
        df_state_probabilities["par_set"] = np.arange(len(df_state_probabilities))
        df_state_probabilities = df_state_probabilities.melt(var_name="State", value_name="Probability", id_vars="par_set")

        with sns.plotting_context("talk", rc={"lines.markersize": 7}):
            fig, ax = plt.subplots(figsize=(6,5))
            p = sns.lineplot(data=df_state_probabilities, x = "State", y="Probability", color="black", alpha=0.2,
                                estimator=None, units="par_set", legend=False).set_title(condition)
            sns.scatterplot(data=df_state_probabilities, x = "State", y="Probability", color="black", alpha=0.2, ax=ax, legend=False, zorder=10)
            sns.despine()
            plt.xticks(rotation=90)
            # Save plot
            plt.savefig("%s/%s.png" % (figures_dir, name), bbox_inches="tight")
            plt.close()
