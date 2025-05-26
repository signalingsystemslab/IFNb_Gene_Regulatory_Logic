import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["figure.dpi"] = 600
results_dir = "parameter_scan_dist_syn/nice_figures/"

color_palette = sns.blend_palette(["white", "#77A5A4","#5A8A8A","#182828"], n_colors=11)
cmap = sns.blend_palette(["white", "#77A5A4","#5A8A8A","#182828"], as_cmap=True)
prob_color = color_palette[4]

state_summary = pd.read_csv("parameter_scan_dist_syn/results/p50_dist_syn_state_probabilities_summary.csv", index_col=0)
state_summary = state_summary.reset_index(names="State")
state_order = state_summary["State"].unique()
state_summary["State"] = pd.Categorical(state_summary["State"], categories=state_order, ordered=True)
state_summary = pd.melt(state_summary, id_vars=["State"], var_name="Condition", value_name="Fraction")

# Group all conditions together
state_summary_gp = state_summary.groupby("State").sum().reset_index()
state_summary_gp["Percentage"] = state_summary_gp["Fraction"] / state_summary_gp["Fraction"].sum() * 100
print(state_summary_gp)

# Plot bar plot of state probabilities
with sns.plotting_context("paper", font_scale=1.5):
    fig, ax = plt.subplots(figsize=(5,5))
    sns.barplot(data=state_summary_gp, x="State", y="Percentage", ax=ax, color=prob_color,
                linewidth=1.5, edgecolor="black", saturation=0.8, width=0.6)
    ax.set_ylabel("State Probability (%)")
    ax.set_xlabel("")
    sns.despine()
    plt.xticks(rotation=90)
    plt.ylim(0, 40)
    plt.tight_layout()
    plt.savefig("%s/state_probabilities_summary.png" % results_dir)

# Group by stimulus
state_summary["Stimulus"] = state_summary["Condition"].str.split("_").str[0]
state_summary["Stimulus"] = state_summary["Stimulus"].replace({"basal":"Basal", "polyIC": "PolyIC"})
state_summary_gp2 = state_summary.groupby(["State", "Stimulus"]).sum().reset_index()
state_summary_gp2["Percentage"] = state_summary_gp2["Fraction"] / state_summary_gp2.groupby("Stimulus")["Fraction"].transform("sum") * 100
print(state_summary_gp2)
stim_colors= {
    "Basal": "#6d6e71",
    "CpG": "#86a43e",
    "LPS": "#ae392f",
    "PolyIC": "#4e3666"
    }

state_summary_gp2["Stimulus"] = pd.Categorical(state_summary_gp2["Stimulus"], categories=stim_colors.keys(), ordered=True)

# Plot bar plot of state probabilities by stimulus
with sns.plotting_context("paper", font_scale=1.5):
    fig, ax = plt.subplots(figsize=(6,4))
    sns.barplot(data=state_summary_gp2, x="State", y="Percentage", hue="Stimulus", ax=ax, palette=stim_colors,
                linewidth=1.5, edgecolor="black", saturation=0.8)
    ax.set_ylabel("State Probability (%)")
    ax.set_xlabel("")
    sns.despine()
    plt.xticks(rotation=90)
    sns.move_legend(ax, "center left", bbox_to_anchor=(1, 0.5), title="Stimulus", frameon=False)
    plt.ylim(0, 50)
    plt.tight_layout()
    plt.savefig("%s/state_probabilities_by_stimulus.png" % results_dir)

with sns.plotting_context("paper", font_scale=1.5):
    fig, ax = plt.subplots()
    p = sns.FacetGrid(state_summary_gp2, col="Stimulus", col_wrap=1, height=2.5, sharey=True, sharex=True, aspect=2)
    p.map_dataframe(sns.barplot, x="State", y="Percentage", hue="Stimulus", palette=stim_colors,
                    linewidth=1.5, edgecolor="black", saturation=0.8)
    p.set_axis_labels("", "Probability")
    p.set_titles(col_template="{col_name}")
    p.set(ylim=(0, 50)) 
    sns.despine()
    for a in p.axes.flat:
        a.tick_params(axis='x', rotation=90)
    plt.tight_layout()
    plt.savefig("%s/state_probabilities_by_stimulus_facet.png" % results_dir)

# active states
# $IRF_1& IRF_2$
# $IRF_1& NF\kappa B$
# $IRF_2& NF\kappa B$
# $IRF_2& NF\kappa B& p50$
# $IRF_1& IRF_2& NF\kappa B$


active_states = [r"$IRF_1& IRF_2$",
                 r"$IRF_1& NF\kappa B$",
                 r"$IRF_2& NF\kappa B$",
                 r"$IRF_2& NF\kappa B& p50$",
                 r"$IRF_1& IRF_2& NF\kappa B$"]
state_summary_active_only = state_summary_gp2[state_summary_gp2["State"].isin(active_states)].copy()
state_summary_active_only["State"] = pd.Categorical(state_summary_active_only["State"].cat.remove_unused_categories())
state_summary_active_only = state_summary_active_only[state_summary_active_only["Stimulus"] != "Basal"]
state_summary_active_only["Stimulus"] = state_summary_active_only["Stimulus"].cat.remove_unused_categories()
with sns.plotting_context("paper", font_scale=1.5):
    fig, ax = plt.subplots()
    p = sns.FacetGrid(state_summary_active_only, col="Stimulus", col_wrap=1, height=1.75, sharey=True, sharex=True, aspect=2.5)
    p.map_dataframe(sns.barplot, y="State", x="Percentage", hue="Stimulus", palette=stim_colors,
                    linewidth=1.5, edgecolor="black", saturation=0.8)
    p.set_axis_labels("Percentage", "")
    p.set_titles(col_template="{col_name}")
    p.set(xlim=(0, 50)) 
    sns.despine()
    plt.tight_layout()
    plt.savefig("%s/state_probabilities_by_stimulus_active.png" % results_dir)
