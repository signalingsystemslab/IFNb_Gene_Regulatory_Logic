import numpy as np
import pandas as pd
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["figure.dpi"] = 600
mpl.rcParams["font.sans-serif"] = "Arial"
# irf_color = "#5D9FB5"
# nfkb_color = "#BA4961"
data_color = "#6F5987"

states_cmap_pars = "ch:s=2.2,r=0.75,h=0.6,l=0.8,d=0.25"
models_cmap_pars = "ch:s=-0.0,r=0.6,h=1,d=0.3,l=0.8,g=1_r"
dataset_cmap_pars = "ch:s=0.9,r=-0.75,h=0.6,l=0.8,d=0.1"

heatmap_cmap = sns.cubehelix_palette(as_cmap=True, light=0.95, dark=0, reverse=True, rot=0.4,start=-.2, hue=0.6)
# dataset_cmap = sns.cubehelix_palette(as_cmap=True, start=0.9, rot=-.75, dark=0.1, light=0.8, hue=0.6)

plot_rc_pars = {"axes.labelsize":7, "font.size":6, "legend.fontsize":6, "xtick.labelsize":6, 
                                          "ytick.labelsize":6, "axes.titlesize":7, "legend.title_fontsize":7,
                                          "lines.markersize": 3, "axes.linewidth": 0.5,
                                            "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.minor.width": 0.5,
                                            "ytick.minor.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2,
                                            "xtick.minor.size": 1, "ytick.minor.size": 1, "legend.labelspacing": 0.2,
                                            "legend.columnspacing": 0.5, "legend.handletextpad": 0.5, "legend.handlelength": 1.5}
rc_pars={"xtick.major.pad": 1, "ytick.major.pad": 1, "legend.labelspacing": 0.2}
mpl.rcParams.update(rc_pars)

def main():
    # Load synthetic data
    errors_tested = np.loadtxt("error_percentages_tested.txt")

    pal = sns.cubehelix_palette(as_cmap=True, start=.6, rot=-.4, dark=0.2, light=0.9, hue=0.6)    
    norm = plt.Normalize(vmin=0, vmax=1)

    for e in errors_tested:
        synthetic_data = pd.read_csv("../data/p50_training_data_plus_synthetic_e%.1fpct.csv" % e)
        # print(synthetic_data.loc[synthetic_data["Dataset"]=="original"], "IFNb")
        # print(synthetic_data)

        # raise ValueError("Stop here")

        with sns.plotting_context("paper", rc=plot_rc_pars):
            print("Plotting scatterplots for standard error %.2f" % e)
            # Plot all data where Genotype != p50KO, x axis is IRF, y axis is NFkB, color is IFNb
            fig, ax = plt.subplots(figsize=(1,1))
            p = sns.scatterplot(data=synthetic_data[synthetic_data["Genotype"]!="p50KO"], x="NFkB", y="IRF", hue="IFNb", ax=ax,
                palette = pal, linewidth=0, sizes=1, legend="brief", hue_norm=norm)
            p.set_xlim(0, 1)
            p.set_ylim(0, 1)
            p.set_xlabel(r"NF$\kappa$B")
            sns.move_legend(ax, bbox_to_anchor=(1, 0.5), frameon=False, loc="center left", title=r"IFN$\beta$")
            p.set_title("WT, %.1f percent" % e)
            plt.savefig("IRF_NFkB_IFNb_scatterplot_e%.1fpct.png" % e, dpi=300, bbox_inches = "tight")

            fig, ax = plt.subplots(figsize=(1,1))
            p = sns.scatterplot(data=synthetic_data[synthetic_data["Genotype"]=="p50KO"], x="NFkB", y="IRF", hue="IFNb", ax=ax,
                palette = pal, linewidth=0, sizes=1, legend="brief", hue_norm=norm)
            p.set_xlim(0, 1)
            p.set_ylim(0, 1)
            p.set_xlabel(r"NF$\kappa$B")
            sns.move_legend(ax, bbox_to_anchor=(1, 0.5), frameon=False, loc="center left", title=r"IFN$\beta$")
            p.set_title("p50KO, %.1f percent" % e)
            plt.savefig("IRF_NFkB_IFNb_scatterplot_p50KO_e%.1fpct.png" % e, dpi=300, bbox_inches = "tight")


if __name__ == "__main__":
    main()
